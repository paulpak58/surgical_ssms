#################################
# Base code from Jin et al.
# Copyright (c) CUHK 2021. 
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
#
# Revised by Paul Pak 
#################################


import cv2
import os
import numpy as np
import PIL
from PIL import Image
import argparse
import sys
import shutil

class ArgParserHelper(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'ERROR: {message}\n')
        sys.stderr.write(f'Source path should point to existing cholec80 dataset containing .mp4 videos.\n')
        sys.stderr.write(f'Save path should point to directory to save extracted frames.\n')
        self.print_help
        sys.exit(2)

parser = ArgParserHelper()
parser.add_argument('--source', type=str, required=True, help='Path to cholec80 videos')
parser.add_argument('--save', type=str, required=True, help='Path to save directories of frames')


def change_size(image):
 
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    
    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left  
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    # Check boundary conditions
    # Videos [7,40,54,56,70] have no change in size in at least one edge after cropping
    if not width and not height:
        pre1_picture = image[:, :]
    elif not width:
        pre1_picture = image[:, bottom:bottom + height]
    elif not height:
        pre1_picture = image[left:left + width, :]
    else:
        pre1_picture = image[left:left + width, bottom:bottom + height]

    #print(pre1_picture.shape) 
    
    return pre1_picture  

def create_frames(source_path, save_path):

    Video_nums = np.arange(1,81,1)

    for Video_num in Video_nums:

        frame_num = 0
        if not os.path.exists(save_path+str(Video_num)):
            os.mkdir(save_path+str(Video_num)) 

        # cap = cv2.VideoCapture(source_path+"Chole"+str(Video_num)+".mp4")
        if Video_num<10:
            cap = cv2.VideoCapture(source_path+"video0"+str(Video_num)+".mp4")
        else:
            cap = cv2.VideoCapture(source_path+"video"+str(Video_num)+".mp4")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            img_save_path = save_path+str(Video_num)+'/'+ str(frame_num)+".jpg"
            
            dim = (int(frame.shape[1]/frame.shape[0]*300), 300)
            
            frame = cv2.resize(frame,dim)
            frame = change_size(frame)
            img_result = cv2.resize(frame,(250,250))
            # print(img_result.shape)
            # print(img_result.dtype)

            img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            img_result = PIL.Image.fromarray(img_result)
            img_result = np.array(img_result)
            # print(img_result.mode)

            cv2.imwrite(img_save_path, img_result)
            print(img_save_path) 
            frame_num = frame_num+1
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print("Cut Done")

def create_phase_annotations(source_path, phase_annotation_path):
    Video_nums = np.arange(1,81,1)
    for Video_num in Video_nums:
        # Copy phase annotations from source path
        if Video_num<10:
            src_annotation_path = source_path+"video0"+str(Video_num)+"-phase.txt"
            shutil.copy(src_annotation_path, phase_annotation_path+"video0"+str(Video_num)+"-phase.txt")
        else:
            src_annotation_path = source_path+"video"+str(Video_num)+"-phase.txt"
            shutil.copy(src_annotation_path, phase_annotation_path+"video"+str(Video_num)+"-phase.txt")
        print(f'Video {Video_num} Complete')

    print("Phase Annotation Copy Complete")


if __name__=='__main__':
    # File path to cholec80 videos
    # source_path = "/saiil2/paulpak/Dataset/cholec80/Video/"
    # File path to save directories of frames
    # save_path = "/saiil2/paulpak/Dataset/cholec80/frames/"

    args = parser.parse_args()
    source_path = args.source
    save_path = args.save
    if not os.path.exists(save_path + 'Dataset'):
        save_path = save_path + 'Dataset/'
        os.mkdir(save_path)
    if not os.path.exists(save_path + 'cholec80'):
        save_path = save_path + 'cholec80/'
        os.mkdir(save_path)
    if not os.path.exists(save_path + 'cutMargin'):
        frame_path = save_path + 'cutMargin/'
        os.mkdir(frame_path)
    if not os.path.exists(save_path + 'phaseAnnotations'):
        phase_annotation_path = save_path + 'phaseAnnotations/'
        os.mkdir(phase_annotation_path)

    create_phase_annotations(source_path, phase_annotation_path)
    create_frames(source_path, frame_path)