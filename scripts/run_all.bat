@echo off
REM ===================================================================
REM  CENG 476 - Severstal - GECELIK TAM CALISMA (Windows)
REM  Kullanim: bu dosyaya cift tikla. Butun cikti log.txt'ye yazilir.
REM ===================================================================
cd /d "%~dp0.."
set DATA=data

echo [1/6] Hizli kontrol (smoke test)...
python src\train.py --data_dir %DATA% --limit 300 --epochs 2 --run_name smoke_test > log.txt 2>&1
if errorlevel 1 (
  echo HATA! Smoke test basarisiz. log.txt dosyasini ac ve hatayi oku.
  pause
  exit /b 1
)
echo     OK.

echo [2/6] Baseline model egitiliyor...
python src\train.py --data_dir %DATA% --run_name baseline --pretrained 0 --use_bn 0 ^
  --decoder_dropout 0.0 --use_attention 0 --weight_decay 0.0 --augment 0 ^
  --cls_loss_weight 0.0 --scheduler none --epochs 15 >> log.txt 2>&1

echo [3/6] Final model egitiliyor (en uzun adim)...
python src\train.py --data_dir %DATA% --run_name final --epochs 30 >> log.txt 2>&1

echo [4/6] Final model degerlendiriliyor...
python src\evaluate.py --data_dir %DATA% --run_name final >> log.txt 2>&1

echo [5/6] Baseline degerlendiriliyor...
python src\evaluate.py --data_dir %DATA% --run_name baseline >> log.txt 2>&1

echo [6/6] Karsilastirma tablosu olusturuluyor...
python src\compare_runs.py --out_dir outputs >> log.txt 2>&1

echo.
echo ================== BITTI ==================
echo Sonuclar: outputs\final\  ve  outputs\baseline\
echo Tum log: log.txt
pause
