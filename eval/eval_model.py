########################################################
# Base code adopted from Jin et. al.
# https://github.com/YuemingJin/TMRNet/blob/main/code/eval/python/get_paths_labels.py
# https://github.com/YuemingJin/TMRNet/blob/main/code/eval/python/export_phase_copy.py
# 
# Revised by: Paul Pak
########################################################

import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', required=True)
parser.add_argument('--pred_path', type=str, default='', required=True)
parser.add_argument('--dropped_last', type=bool, default=False, required=False)     # if true, then label and pred size different below
parser.add_argument('--numTest', type=int, default=40, required=False)
parser.add_argument('--use_pkl', type=bool, default=True, required=False)
args = parser.parse_args()

data_path = args.data_path
pred_path = args.pred_path

root_dir2 = data_path
img_dir2 = os.path.join(root_dir2, 'cutMargin')
phase_dir2 = os.path.join(root_dir2, 'phase_annotations')
# root_dir2 = '/home/ppak/surgical_adventure/Dataset/cholec80/'


######################################
# Retrieve Phase Classification Labels
######################################  
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x:int(x))
    file_paths.sort(key=lambda x:int(os.path.basename(x)))
    return file_names, file_paths

def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths



######################################
# Retrieve Phase Classification Labels
######################################  
img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)
# tool_file_names2, tool_file_paths2 = get_files2(tool_dir2)
phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)

phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)


######################################
# Retrieve Phase Classification Labels
######################################  
all_info_all2 = []
for j in range(len(phase_file_names2)):
    downsample_rate = 25
    phase_file = open(phase_file_paths2[j])
    # tool_file = open(tool_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))

    print("video_num_file:", video_num_file,"video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[1]])
            info_all.append(info_each)            

    # print(len(info_all))
    all_info_all2.append(info_all)


######################################
# cholec80.pkl file gives more freedom
# in allocating what is train, val, test
######################################  
with open('./cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)


############################################
# Given the predictions of the model,
# we generate folders classifying each frame
############################################
with open('./cholec80.pkl', 'rb') as f:
    test_paths_labels = pickle.load(f)

sequence_length = 1
model_name = pred_path.split('.pkl')[0]


############################################
# Extract out either from a pkl or a 
# jax npy serial file
############################################
with open(pred_path , 'rb') as f:
    if args.use_pkl:
        ori_preds = pickle.load(f)
    else:
        ori_preds = np.load(f, allow_pickle=True)



#  Test videos are Video indices 40-79

start_video_idx = 40
end_video_idx = 80
num_video = 40
num_labels = 0
for i in range(start_video_idx, end_video_idx):
    print(f'Video {i} len {len(test_paths_labels[i])}')
    num_labels += len(test_paths_labels[i])
num_preds = len(ori_preds)


print('num of labels  : {:6d}'.format(num_labels))
print("num ori preds  : {:6d}".format(num_preds))
print("labels example : ", test_paths_labels[0][0][1])
# print("labels example : ", test_paths_labels[0])
print("preds example  : ", ori_preds[0])
print("sequence length : ", sequence_length)

if args.dropped_last:
    raise Exception('fix size check below to match batch size that model was trained on')

if num_labels == (num_preds + (sequence_length - 1) * num_video):

    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
    preds_all = []
    label_all = []
    count = 0
    for i in range(start_video_idx, end_video_idx):
        # filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/phase/video' + str(1 + i) + '-phase.txt'
        # gt_filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/gt-phase/video' + str(1 + i) + '-phase.txt'
        if not os.path.exists('./phase'):
            os.makedirs('./phase')
        if not os.path.exists('./gt-phase'):
            os.makedirs('./gt-phase')
        filename = './phase/video' + str(i+1) + '-phase.txt'
        gt_filename = './gt-phase/video' + str(i+1) + '-phase.txt'
    
        f = open(filename, 'w')
        f2 = open(gt_filename, 'w')
        preds_each = []
        for j in range(count, count + len(test_paths_labels[i]) - (sequence_length - 1)):
            if j == count:
                for k in range(sequence_length - 1):
                    preds_each.append(0)
                    preds_all.append(0)
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(25 * k))
            f.write('\t')
            f.write(str(int(preds_each[k])))
            f.write('\n')
            
            f2.write(str(25 * k))
            f2.write('\t')
            f2.write(str(int(test_paths_labels[i][k][1])))
            label_all.append(test_paths_labels[i][k][1])
            f2.write('\n')
        print(f"Video {i} complete")

        f.close()
        f2.close()
        count += len(test_paths_labels[i]) - (sequence_length - 1)
    test_corrects = 0

    print('num of labels       : {:6d}'.format(len(label_all)))
    print('rsult of all preds  : {:6d}'.format(len(preds_all)))
    for i in range(num_labels):
        if int(label_all[i]) == int(preds_all[i]):
            test_corrects += 1
    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')

print('Test predictions extraction complete')