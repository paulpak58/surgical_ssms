#!/bin/bash

home="/home/ubuntu/surgical_ncp/surgical-liquid-S5/s5/scripts"
scripts=(
    "${home}/run_liquidS5_cholec80_0.sh"
    "${home}/run_liquidS5_cholec80_0_bilinear.sh"
    "${home}/run_liquidS5_cholec80_1.sh"
    "${home}/run_liquidS5_cholec80_2.sh"
    "${home}/run_liquidS5_cholec80_3.sh"
    "${home}/run_liquidS5_cholec80_4.sh"
    "${home}/run_liquidS5_cholec80_5.sh"
    "${home}/run_liquidS5_cholec80_6.sh"
    "${home}/run_liquidS5_cholec80_7.sh"
)

# Loop through each script and execute them sequentially
for script in "${scripts[@]}"; do
    echo "Running $script"
    sh "$script"
    echo "Finished running $script"
done
