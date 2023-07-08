from fastapi import FastAPI, UploadFile, File, Form
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
from PIL import Image
app = FastAPI()


@app.post("/seg/")
async def upload_file(data: str = Form(...)):
    input_root = "/opt/ml/seg_app/app/input" 
    save_root = "/opt/ml/seg_app/app/seg_db"
    image_path = os.path.join(input_root, data)
    image = Image.open(image_path)
    image = np.array(image)
    masks, scores, logits = seg(image)

    save_paths = []  # 저장된 파일 경로 리스트
    
    for i in range(3):
        # 파일 이름과 확장자 분리
        file_name, extension = os.path.splitext(data)
        data_file_name = f"{file_name}_{i}{extension}"
        save_path = os.path.join(save_root, data_file_name) 
        save_masked_image(image, masks[i], save_path)
        save_paths.append(save_path)  # 저장된 파일 경로 추가

    return save_paths
    


@app.get("/")
def main():
    return  0
