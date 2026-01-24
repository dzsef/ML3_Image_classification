# ML3_Image_classification


If you don't intend to train, just experiment with evaluation please DO NOT execute step 2) and 3) -> jump to the Evaluation title, or single image evaluation depending on your goals.
only download the images, then use the evaluation script, or your own eval pipeline.


# 1) Download DIV2K HR train/valid and unpacks into data/div2k/: 

chmod +x scripts/get_div2k.sh
./scripts/get_div2k.sh

# 2) Then Build HDF5 training/eval files:

chmod +x scripts/prepare_div2k_h5.sh
./scripts/prepare_div2k_h5.sh

# 3) Train

python third_party/srcnn_pytorch/train.py \
  --train-file artifacts/h5/div2k_train_x2.h5 \
  --eval-file artifacts/h5/div2k_val_x2.h5 \
  --outputs-dir outputs/srcnn_div2k_x2 \
  --scale 2 \
  --lr 1e-4 \
  --batch-size 16 \
  --num-epochs 50 \
  --num-workers 8 \
  --seed 123



## _FOR a single image evaluation:_


python third_party\srcnn_pytorch\test.py `
   --weights-file outputs\srcnn_div2k_x2\x2\best.pth ` 
   --image-file data\div2k\HR\val\0801.png `
   --scale 

or use any other png.

# EVALUATION
Evaluation on DIV2K val (quantitative + qualitative):

python scripts\eval_div2k.py `
  --weights outputs\srcnn_div2k_x2\x2\best.pth `
  --hr-dir data\div2k\HR\val `
  --scale 2 `
  --out-dir outputs\eval_x2 `
  --ssim


# WRAPPER for prepare, train, eval
One-command runner ONLY AFTER POPULATING THE TRAIN & VAL FOLDERS WITH pngs (prepare -> train -> eval):

python scripts\run_experiment.py `
  --scale 2 `
  --patch-size 33 `
  --stride 64 `
  --batch-size 16 `
  --epochs 50 `
  --lr 1e-4 `
  --num-workers 8 `
  --seed 123 `
  --eval-ssim

Config file option (CLI flags still override config values):

python scripts\run_experiment.py `
  --config config.yml
