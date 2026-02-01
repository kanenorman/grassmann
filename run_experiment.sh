#!/bin/bash

python train.py \
  --name attention-6-layer-128-seq \
  --num-layers 6 \
  --d-model 256 \
  --max-context-len 128 \
  --batch-size 32
