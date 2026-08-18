"""U-Net + ResNet34 encoder (transfer learning) + SCSE attention + coklu gorev basligi.

Mimari ozeti (girdi 3 x 256 x W):
  ENCODER (ResNet34, ImageNet on egitimli)
    x0 = Conv7x7-BN-ReLU          ->  64 x H/2  x W/2
    x1 = MaxPool + layer1         ->  64 x H/4  x W/4
    x2 = layer2                   -> 128 x H/8  x W/8
    x3 = layer3                   -> 256 x H/16 x W/16
    x4 = layer4  (bottleneck)     -> 512 x H/32 x W/32
  DECODER (her blok: Upsample -> concat(skip) -> [Conv-BN-Act]x2 -> Dropout2d -> SCSE)
    d4 (512+256 -> 256), d3 (256+128 -> 128), d2 (128+64 -> 64),
    d1 (64+64 -> 32),    d0 (32 -> 16)
  BASLIKLAR
    segmentation: Conv1x1(16 -> 4)   ->  4 x H x W  (logit, sigmoid ile olasiliga cevrilir)
    classification (yardimci): GAP(x4) -> Dropout -> Linear(512 -> 4)

Neden bu mimari?
  * U-Net: skip connection'lar encoder'daki yuksek cozunurluklu kenar bilgisini
    decoder'a tasir. Celik kusurlari ince ve uzun yapilar oldugu icin bu kritik.
  * ResNet34 encoder: 12.5k goruntu sifirdan derin bir encoder egitmek icin az.
    ImageNet ozellikleri (kenar, doku) celik yuzeyi dokusuna iyi transfer oluyor.
  * SCSE: kanal + uzamsal dikkat. Kusurun nerede oldugunu vurgulayip arka plani bastiriyor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    name = name.lower()
    return {
        "relu": nn.ReLU(inplace=True),
        "leakyrelu": nn.LeakyReLU(0.1, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "elu": nn.ELU(inplace=True),
    }[name]


class ConvBnAct(nn.Module):
    """Conv3x3 -> (BatchNorm) -> Activation. BN ablasyon icin kapatilabilir."""

    def __init__(self, c_in, c_out, use_bn=True, activation="relu"):
        super().__init__()
        # BN varken conv bias gereksiz (BN zaten kaydirma yapiyor)
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(c_out) if use_bn else nn.Identity()
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SCSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation (Roy et al., 2018).
    Yaratici bilesen: hangi kanallar ve hangi uzamsal bolgeler onemli, model ogreniyor."""

    def __init__(self, c, r=8):
        super().__init__()
        hidden = max(c // r, 2)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out, use_bn=True, dropout=0.0,
                 attention=True, activation="relu"):
        super().__init__()
        self.conv1 = ConvBnAct(c_in + c_skip, c_out, use_bn, activation)
        self.conv2 = ConvBnAct(c_out, c_out, use_bn, activation)
        # Dropout2d: tek piksel yerine tum kanali dusurur -> konvolusyonel katmanlarda
        # standart dropout'tan daha etkili bir duzenlilestirme.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.att = SCSE(c_out) if attention else nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return self.att(self.drop(x))


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, use_bn=True,
                 dropout=0.2, attention=True, activation="relu",
                 cls_head=True):
        super().__init__()
        from torchvision.models import resnet34
        try:
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            enc = resnet34(weights=weights)
        except Exception as e:  # eski torchvision veya internet kapali
            print(f"[uyari] on egitimli agirlik yuklenemedi ({e}); sifirdan basliyoruz.")
            enc = resnet34(weights=None) if not pretrained else resnet34()

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64,  /2
        self.pool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2         # 64 /4, 128 /8
        self.layer3, self.layer4 = enc.layer3, enc.layer4         # 256 /16, 512 /32

        kw = dict(use_bn=use_bn, dropout=dropout, attention=attention, activation=activation)
        self.dec4 = DecoderBlock(512, 256, 256, **kw)
        self.dec3 = DecoderBlock(256, 128, 128, **kw)
        self.dec2 = DecoderBlock(128, 64, 64, **kw)
        self.dec1 = DecoderBlock(64, 64, 32, **kw)
        self.dec0 = DecoderBlock(32, 0, 16, **kw)

        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Yardimci siniflandirma basligi (coklu gorev ogrenme).
        # Goruntulerin ~%50'sinde hic kusur yok; bu baslik modele "once kusur var mi?"
        # sorusunu ogretip yanlis pozitifleri azaltiyor.
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(512, num_classes)) if cls_head else None

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d = self.dec4(x4, x3)
        d = self.dec3(d, x2)
        d = self.dec2(d, x1)
        d = self.dec1(d, x0)
        d = self.dec0(d)

        seg = self.seg_head(d)
        if seg.shape[-2:] != x.shape[-2:]:
            seg = F.interpolate(seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cls = self.cls_head(x4) if self.cls_head is not None else None
        return seg, cls   # ikisi de LOGIT (sigmoid uygulanmadi)


def build_model(cfg):
    return UNetResNet34(
        num_classes=cfg.num_classes,
        pretrained=bool(int(cfg.pretrained)),
        use_bn=bool(int(cfg.use_bn)),
        dropout=float(cfg.decoder_dropout),
        attention=bool(int(cfg.use_attention)),
        activation=cfg.activation,
        cls_head=float(cfg.cls_loss_weight) > 0,
    )


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Hizli sekil kontrolu:  python src/model.py
    from config import Config
    cfg = Config()
    m = build_model(cfg)
    x = torch.randn(2, 3, 256, 512)
    seg, cls = m(x)
    print("seg:", seg.shape, "cls:", None if cls is None else cls.shape)
    print("parametre sayisi: %.2fM / egitilebilir %.2fM" % tuple(v / 1e6 for v in count_parameters(m)))
