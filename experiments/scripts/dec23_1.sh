#!/bin/sh

# Base pinpad
python hieros/train.py --configs pinpad \
--batch_size=16 --batch_length=64 \
--wandb_prefix=base --wandb_name=pinpad_three

# Pyramidal pinpad
python hieros/train.py --configs pinpad hierarchy_decrease \
--batch_size=16 --batch_length=64 \
--wandb_prefix=pyramidal --wandb_name=pinpad_three 