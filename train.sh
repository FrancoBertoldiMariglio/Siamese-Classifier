#!/bin/bash

python train.py \
    --train_path train_siamese_dataset \
    --val_path val_siamese_dataset \
    -o output_siamese \
    -b resnet18 \
    -lr 0.0001 \
    -e 1000 \
    -s 25
