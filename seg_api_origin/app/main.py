from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
#from predictor import SamPredictor
import numpy as np
import torch
#import matplotlib.pyplot as plt
import os
from datetime import datetime
from app.predictor import seg, save_masked_image
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import time
import warnings
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything.utils.onnx import SamOnnxModel
app = FastAPI()

class SegRequest(BaseModel):
    x: int
    y: int
    file_name: str

def model_define():
    sam_checkpoint = './weights/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    onnx_model_path = "/opt/ml/level3_cv_finalproject-cv-02-1/seg_api/weights/sam_onnx_quantized_example.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)


    return predictor, ort_session 
predictor, ort_session = model_define()

class SegRequest(BaseModel):
    x: int
    y: int
    file_name: str

@app.post("/seg/")
async def upload_file(request: Request, data: SegRequest):
    file_name = data.file_name
    x = data.x
    y = data.y
    input_root = "./app/input" 
    save_root = "./app/seg_db"
    image_path = os.path.join(input_root, file_name)
    image = Image.open(image_path)
    image = np.array(image)
    start = time.time()
    masks,_,_  = seg(predictor, image,x,y) #mask, score, logit
    end = time.time()
    print(f"seg time : {end-start} sec")

    save_paths = []  # 저장된 파일 경로 리스트

    for i in range(3):
        # 파일 이름과 확장자 분리
        name, extension = os.path.splitext(file_name)
        data_file_name = f"{name}_{i+1}{extension}"
        save_path = os.path.join(save_root, data_file_name) 
        save_masked_image(image, masks[i], save_path)
        save_paths.append(save_path)  # 저장된 파일 경로 추가

    return save_paths
    


@app.get("/")
def main():
    return  0
