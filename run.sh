#!/bin/bash

# parameters setup
gamma_list=("0.1" "0.3" "0.6" "1.0")
solver_list=("ISTA" "FISTA" "ADMM")
A=data/A.mat
b=data/b.mat
tolerance=1e-8

# run the program with different parameters
for gamma in ${gamma_list[@]}
do
  for solver in ${solver_list[@]}
    do
      python lasso.py \
      --A ${A} \
      --b ${b} \
      --gamma ${gamma} \
      --opt ${solver} \
      --tol ${tolerance}
  done
done

# run the plot program
python plot.py \
  --output_dir ./output \
  --gamma 0.1,0.3,0.6,1.0

# quit
exit