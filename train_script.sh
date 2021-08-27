#!/bin/bash
python train.py \
	--dataset MaskSplitByProfileDataset\
	--epochs 5\
	--batch_size 128\
	--valid_batch_size 128\
	--name proc\
