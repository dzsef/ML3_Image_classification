# ML3_Image_classification

This repo contains a simple SRCNN pipeline for DIV2K. If you only want evaluation, do step 1 and then run the evaluation. You do not need steps 2 and 3 if you already have model weights.

# 1) Download DIV2K HR train and val into data/div2k
Linux or macOS
```
chmod +x scripts/get_div2k.sh
./scripts/get_div2k.sh
```

Windows PowerShell
```
powershell -ExecutionPolicy Bypass -File scripts/get_div2k.ps1
```

MSYS2 or Git Bash
```
bash scripts/get_div2k.sh
```

# 2) Build HDF5 files (needed only for training)
```
chmod +x scripts/prepare_div2k_h5.sh
./scripts/prepare_div2k_h5.sh
```

# 3) Train (direct script)
```
python third_party/srcnn_pytorch/train.py --train-file artifacts/h5/div2k_train_x2.h5 --eval-file artifacts/h5/div2k_val_x2.h5 --outputs-dir outputs/srcnn_div2k_x2 --scale 2 --lr 1e-4 --batch-size 16 --num-epochs 50 --num-workers 8 --seed 123
```

# Single image evaluation
```
python third_party/srcnn_pytorch/test.py --weights-file outputs/srcnn_div2k_x2/x2/best.pth --image-file data/div2k/HR/val/0801.png --scale 2
```

# Evaluation on DIV2K val (quantitative and qualitative)
```
python scripts/eval_div2k.py --weights outputs/srcnn_div2k_x2/x2/best.pth --hr-dir data/div2k/HR/val --scale 2 --out-dir outputs/eval_x2 --ssim
```

# One-command runner for prepare, train, eval
```
python scripts/run_experiment.py --scale 2 --patch-size 33 --stride 64 --batch-size 16 --epochs 50 --lr 1e-4 --num-workers 8 --seed 123 --eval-ssim
```

You can also run only specific steps with flags
```
--skip-prepare
--skip-train
--skip-eval
```

# Config file option
```
python scripts/run_experiment.py --config config.yml
```
CLI flags override config values.
