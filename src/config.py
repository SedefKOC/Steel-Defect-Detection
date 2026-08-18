"""Merkezi konfigurasyon. Butun hiperparametreler burada tanimli.

train.py komut satirindan bu alanlarin hepsini ezebilir:
    python src/train.py --lr 1e-4 --decoder_dropout 0.3
"""
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # ---------------- Yollar ----------------
    # Kaggle notebook'ta bu yol hazir gelir. Lokalde calisiyorsan degistir.
    data_dir: str = "/kaggle/input/severstal-steel-defect-detection"
    out_dir: str = "outputs"
    run_name: str = "final"

    # ---------------- Veri ----------------
    img_h: int = 256          # orijinal yukseklik
    img_w: int = 1600         # orijinal genislik
    crop_w: int = 512         # egitimde rastgele kirpma genisligi (hiz icin)
    num_classes: int = 4
    augment: int = 1          # 0/1 -> veri artirma acik/kapali (ablasyon icin)
    limit: int = 0            # >0 ise sadece N goruntu kullan (hizli test icin)

    # ---------------- Bolme ----------------
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # ---------------- Model ----------------
    pretrained: int = 1        # ImageNet on egitimli ResNet34 encoder
    use_bn: int = 1            # BatchNorm acik/kapali (ablasyon)
    decoder_dropout: float = 0.2
    use_attention: int = 1     # SCSE attention bloklari (yaratici kisim)
    activation: str = "relu"   # relu | leakyrelu | gelu | silu | elu

    # ---------------- Kayip ----------------
    dice_weight: float = 0.5       # toplam = (1-w)*BCE + w*Dice
    cls_loss_weight: float = 0.3   # yardimci siniflandirma basligi agirligi (0 = kapali)
    pos_weight: float = 2.0        # BCE'de pozitif piksellere ekstra agirlik

    # ---------------- Optimizasyon ----------------
    optimizer: str = "adamw"   # adamw | adam | sgd
    lr: float = 3e-4
    weight_decay: float = 1e-4  # L2 regularization
    momentum: float = 0.9       # sadece SGD icin
    scheduler: str = "cosine"   # cosine | plateau | step | none
    epochs: int = 30
    batch_size: int = 16
    val_batch_size: int = 4
    early_stop_patience: int = 6
    amp: int = 1                # mixed precision (GPU'da 2x hiz)
    num_workers: int = 2
    grad_clip: float = 1.0

    # ---------------- Cikarim ----------------
    threshold: float = 0.5
    min_size: int = 600        # bu piksel sayisindan kucuk tahminleri sil

    def to_dict(self):
        return asdict(self)


def add_config_args(parser):
    """Config alanlarini otomatik olarak argparse argumanina cevirir."""
    for f in fields(Config):
        parser.add_argument(f"--{f.name}", type=f.type, default=None)
    return parser


def config_from_args(args):
    cfg = Config()
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg
