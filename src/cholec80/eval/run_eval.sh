#!/bin/bash

dir=./best_preds
data_path=../../../Dataset/cholec80

output_file="best_preds.txt"

echo “$dir”

for pkl_file in "$dir"/*.pkl; do
    echo "=========================="
    echo "** Processing $pkl_file **"
    echo "=========================="

    python3 eval_model.py --data_path "$data_path" --pred_path "$pkl_file"

    echo "$pkl_file has been processed"
    echo "$pkl_file has been processed" >> "$output_file"

    cd ./matlab-eval

    echo "==========================\n" >> "../output_files/$output_file"
    echo "Final evaluation for $pkl_file\n" >> "../output_files/$output_file"
    echo "==========================\n" >> "../output_files/$output_file"

    matlab -nodisplay -nosplash -nodesktop -r "run('Main.m');exit;" | tail -n +11 >> "../output_files/$output_file"

    echo "Matlab evaluation complete!"
    
    cd ../
done
    echo "Done"
