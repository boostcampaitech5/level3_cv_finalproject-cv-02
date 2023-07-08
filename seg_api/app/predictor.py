import numpy as np
import torch
import matplotlib.pyplot as plt
#import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os

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

def seg(image):
    sam_checkpoint = './weights/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    input_point = np.array([[300, 500]]) ##임의로 설정 나중에 좌표값 포인트로 바꿔주기
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    return masks, scores, logits


def save_masked_image(image, mask, save_path):
    # 마스크 이진화
    binary_mask = (mask > 0).astype(np.uint8)

    # 마스크가 있는 부분 추출
    masked_area = image * binary_mask[:, :, np.newaxis]

    # 이미지 저장
    masked_image = Image.fromarray(masked_area)
    masked_image.save(save_path)
    print("aaaaaaaaa")
