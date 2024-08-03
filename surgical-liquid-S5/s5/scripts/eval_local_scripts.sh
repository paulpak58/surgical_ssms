#!/bin/bash

dir=./pred_labels
data_path=../Dataset/cholec80

output_file=outputs0.txt

for pkl_file in "$dir"/*.pkl; do

	echo "Processing $pkl_file:"

	python3 eval_model.py --data_path "$data_path" --pred_path "$dir"/pkl_file

	echo "$pkl_file has been processed"

	cd ./matlab-eval

	echo "Final evaluation for $pkl_file"

	matlab -nodisplay -nosplash -nodesktop -r "run('Main.m');exit;" | tail -n +11 >> "output_file"

	echo "Matlab evaluation complete!"
done
	echo "Done"
