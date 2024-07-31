#! /bin/bash

export CONFIG_PATH='./src/config.yaml'

python ./src/train.py --config $CONFIG_PATH

python ./src/evaluate.py --config $CONFIG_PATH