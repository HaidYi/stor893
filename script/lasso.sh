#!/bin/bash

BIN_PROGRAM=../lasso.py
lambda_list=("0.01" "0.1" "0.2" "0.5" "1.0")

for lambda in ${lambda_list[@]}
  do
    python -W ignore ${BIN_PROGRAM} \
    --data_path ./QuizeData.mat \
    --Lambda ${lambda} \
    --max_iter 1000 \
    --tol 1e-8 \
    --silent
done