# ML3_Image_classification


1) Download DIV2K HR train/valid and unpacks into data/div2k/: 

chmod +x scripts/get_div2k.sh
./scripts/get_div2k.sh

2) Then Build HDF5 training/eval files:

chmod +x scripts/prepare_div2k_h5.sh
./scripts/prepare_div2k_h5.sh

3) Train

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
