# torch
import torch

# cloth-mask library
from carvekit.ml.files.models_loc import download_all
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

# external-library
from PIL import Image, ImageOps
import numpy as np
import cv2

# built-in library
import os.path as osp
import warnings


warnings.filterwarnings('ignore') # 짜잘한 에러 무시


class TracerB7:
    def __init__(self) -> None:
        # download_all()

        SHOW_FULLSIZE = False #param {type:"boolean"}
        PREPROCESSING_METHOD = "none" #param ["stub", "none"]
        SEGMENTATION_NETWORK = "tracer_b7" #param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
        POSTPROCESSING_METHOD = "fba" #param ["fba", "none"]
        SEGMENTATION_MASK_SIZE = 640 #param ["640", "320"] {type:"raw", allow-input: true}
        TRIMAP_DILATION = 30 #param {type:"integer"}
        TRIMAP_EROSION = 5 #param {type:"integer"}
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                    preprocessing_method=PREPROCESSING_METHOD,
                    postprocessing_method=POSTPROCESSING_METHOD,
                    seg_mask_size=SEGMENTATION_MASK_SIZE,
                    trimap_dilation=TRIMAP_DILATION,
                    trimap_erosion=TRIMAP_EROSION,
                    device=DEVICE)

        self.interface = init_interface(config)
    
    
    def inference(self, storage_root: str, img_name: str, mode: str = 'cloth'):
        if mode == 'person':
            img_path = osp.join(storage_root, 'raw_data/person', img_name)
        elif mode == 'cloth':
            img_path = osp.join(storage_root, 'raw_data/cloth', img_name)
        else:
            print("Please check your mode arguments. Supported mode is 'cloth' or 'person'.")
            raise
        
        # inference
        images = self.interface([img_path])
        for _, im in enumerate(images):
            img = np.array(im)
            img = img[...,:3] # no transparency
            idx = (img[...,0]==130)&(img[...,1]==130)&(img[...,2]==130) # background 0 or 130, just try it
            img = np.ones(idx.shape) * 255
            img[idx] = 0
            im = Image.fromarray(np.uint8(img), 'L')
        
        # 마스킹 결과 저장
        mask_path = osp.join(storage_root, 'preprocess/mask', mode, img_name)
        im.save(mask_path, 'JPEG')
        
        save_state = False

        # 파일이 올바르게 저장(경로 및 데이터가 존재)되었다면 save_state=True
        if osp.exists(mask_path) and osp.getsize(mask_path):
            save_state = True

        return save_state
