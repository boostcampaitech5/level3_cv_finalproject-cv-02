from fashion_clip.fashion_clip import FashionCLIP
from transformers import CLIPProcessor
import numpy as np
import os
import yaml
from PIL import Image

import onnxruntime as ort

class EmbeddingModel:
    
    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.IMG_DIR = os.path.join(self.conf["storage"], "queries")
        # self.flip = FashionCLIP("fashion-clip")
        self.image_model_path = self.conf['embedding']['image_model']
        self.text_model_path = self.conf['embedding']['text_model']
        
        name = "patrickjohncyh/fashion-clip"
        auth_token = None
        self.preprocess = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)
        
        self.vision_session = ort.InferenceSession(self.image_model_path, providers=['CPUExecutionProvider'])
        self.text_session = ort.InferenceSession(self.text_model_path, providers=['CPUExecutionProvider'])
        
        
    def image_preprocessing(self, images: list[Image.Image], onnx=False):
        result = self.preprocess(images=images, return_tensors='pt')['pixel_values']
        if onnx:
            return {'input.1': result.detach().cpu().numpy().astype(np.float32)}
        else: 
            return result


    def text_preprocessing(self, texts: list[str], onnx=False):
        result = self.preprocess(text=texts, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        ids, masks = result['input_ids'], result['attention_mask']
        if onnx:
            return {
            'onnx::Reshape_0': ids.detach().cpu().numpy().astype(np.int64),
            'attention_mask': masks.detach().cpu().numpy().astype(np.int64),
            }
        else:
            return ids, masks


    def get_image_embedding(self, rid: str) -> list[float]:
        filename = f"{rid}.png"
        
        img_path = os.path.join(self.IMG_DIR, filename)
        
        # onnx run 참고하기
        # ort_outs = ort_session.run(None, ort_inputs)
        # img_out_y = ort_outs[0]

        onnx_image = self.image_preprocessing([Image.open(img_path)], onnx=True)
        embedding = self.vision_session.run(None, onnx_image)[0].flatten() # (1, 512) -> flatten -> (512,)
        embedding /= np.linalg.norm(embedding) # 정규화
        return embedding.tolist()
    
    
    def get_text_embedding(self, text: str) -> list[float]:
        onnx_text = self.text_preprocessing([text], onnx=True)
        embedding = self.text_session.run(None, onnx_text)[0].flatten() # (1, 512) -> flatten -> (512,)
        embedding /= np.linalg.norm(embedding) # 정규화
        return embedding.tolist()