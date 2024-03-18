# Surgical_NCP repository

## 1. Get data
Final dataset should be in this format. Follow Steps 1A and 1B.
```
├── Dataset 
|  ├── cholec80 
|  |  ├── cutMargin 
|  |  |  ├── 1.jpg 
|  |  |  ├── 2.jpg 
|  |  |  ├── 3.jpg 
|  |  |  ├── ......  
|  |  |  ├── 80.jpg   
|  |  ├── phase_annotations 
|  |  |  ├── video01-phase.txt 
|  |  |  ├── ......  
|  |  |  ├── video80-phase.txt
```



## 1A. Data Preprocessing
### Download Cholec80 dataset
Cholec80 can be downloaded [here](http://camma.u-strasbg.fr/datasets).
SAIIL members also have access to this dataset.
```
conda create -n surgical_ncp python=3.9
pip install -r requirements.txt
```

### 1B. Extract Cholec80 videos into frames
--source = Path of Existing Cholec80 dataset containing .mp4 video files [downloaded from above].
           (e.g. "./Data/cholec80/Video/")

--save = Path of New Cholec80 dataset with individual frames for each video [default: current directory].

```
python video2frame.py --source $SOURCE_PATH --save ./
```
The above python script will generate a directory "Dataset" containing a folder for each video with each frame extracted. You also need to explicitly copy over the labels from the original dataset. You can complete this by running the command below. The "./Dataset/cholec80/phaseAnnotations" path is created from the command above.



## 2. Labels
### Download pkl file from Google Drive
The labels can be separately extracted from the default Cholec80 dataset, but we make use of this pkl file for efficient data transfer.

This pkl file splits the 80 videos into
* Training: Videos 1-40
* Validation: Videos 41-48
* Test: Videos 49-80

```
wget "https://drive.google.com/u/0/uc?id=1YCT0HEqdTFrJkcZpMPM8uG2P52AmOPjd&export=download" -O "train_val_paths_labels1.pkl"
```


## 3. Pretrained Weights
### Download pretrained Resnet & TCN Features from Google Drive
The model follows the Trans-SVNet pipeline, which takes in TCN features, which in turn takes in Resnet features. Model weights do not need
to be retrieved if Resnet and TCN are not used.

```
wget "https://drive.google.com/u/0/uc?id=15nuDP_fSoehHhH-sWQ9hB9ybO_FdSZzk&export=download" -O "resnet_cholec80_weights.pkl"
```

```
wget "https://drive.google.com/u/0/uc?id=1YCkFZGc0_u_WjbUjUESaQsALEgS9TQW0&export=download" -O "tcn_cholec80_weights.pkl"
```


## 4. Extract ResNet Spatial Features and TCN Temporal Features
```
# Extract Resnet Features
python train_resnet.py -L ./train_val_paths_labels1.pkl --weights ./resnet_cholec80_weights.pkl 
```
```
# Extract Temporal Features
python train_tcn.py 
```

	
## Training the model
```
python train_S4.py
```

## Evaluation
### Matlab evaluation according to Cholec80 competition
```
cd ./eval/

python eval_model.py --data_path ./../Dataset/cholec80 --pred_path ./S4_weights.pkl
```

```
cd ./matlab-eval
matlab -nodisplay -nosplash -nodesktop -r "run('Main.m');exit;" | tail -n +11
```


### Knowledge Base about Files
To be updated for further clarity and efficiency
