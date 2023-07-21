import numpy as np
import torch
import matplotlib.pyplot as plt
#import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os
import time


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def seg(predictor,image,x,y):
     

    predictor.set_image(image)
    input_point = np.array([[x, y]]) 
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    #predictor.set_image(image)

    return masks, scores, logits


def save_masked_image(image, mask, save_path):
    # 마스크 이진화
    binary_mask = (mask > 0).astype(np.uint8)

    # 마스크가 있는 부분 추출
    masked_area = image * binary_mask[:, :, np.newaxis]

    #bbox 추출
    seg_value=1
    bbox = 0,0,0,0
    np_seg = np.array(mask)
    segmentation = np.where(np_seg == seg_value)
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, x_max, y_min, y_max

    #crop image
    cropped_image = masked_area[y_min:y_max, x_min:x_max]
    # print(f"crop :{cropped_image.shape}")
    # image = cropped_image.convert("RGB") 
    # print(f"rgb : {image.shape}")
    # 이미지 저장
    masked_image = Image.fromarray(cropped_image)
    masked_image.save(save_path)

