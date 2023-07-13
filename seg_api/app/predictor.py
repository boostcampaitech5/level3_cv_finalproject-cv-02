import numpy as np
import torch
import matplotlib.pyplot as plt
#import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os
import time
import warnings
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything.utils.onnx import SamOnnxModel


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

def seg(predictor,ort_session,image,x,y):
    
    predictor.set_image(image)
    input_point = np.array([[x, y]]) 
    input_label = np.array([1])
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }
    
    masks, scores, _ = ort_session.run(None, ort_inputs)
    
    masks = masks > predictor.model.mask_threshold
    masks=np.squeeze(masks, axis=0)
    scores=np.squeeze(scores, axis=0)

    ms = []
    for mask, score in zip(masks, scores) :
        ms.append([mask, score])

    ms.sort(key = lambda x : x[1], reverse = True)
    

    return list(x[0] for x in ms)




def save_masked_image(image, mask, save_path):
    # 마스크 이진화
    binary_mask = (mask > 0).astype(np.uint8)

    # 마스크가 있는 부분 추출
    masked_area = image * binary_mask[:,:, np.newaxis]

    # 이미지 저장
    masked_image = Image.fromarray(masked_area)
    masked_image.save(save_path)

