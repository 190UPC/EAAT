
# EAAT-pytorch
## Dataset

## Testing

We provide the pretrained models in checkpoint directory.
To test EAAT on benchmark dataset:
`python sample.py \
    --test_data_dir dataset/<dataset> \
    --scale 4 \
    --ckpt_path ./checkpoints/<path>.pth \
    --sample_dir <sample_dir>`


Example:

`python sample.py --test_data_dir dataset/DeepRock2D/test --scale 4 --ckpt_path ./checkpoints/EAAT_x4.pth --sample_dir ./results`

## Training

Here are our settings to train EAAT:

`python train.py \
    --patch_size 64 \
    --batch_size 64 \
    --max_steps 400000 \
    --lr 0.001 \
    --decay 200000 \
    --scale 4`

