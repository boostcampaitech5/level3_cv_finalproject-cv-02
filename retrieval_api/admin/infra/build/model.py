from fashion_clip.fashion_clip import FashionCLIP
import os
import pandas as pd
import pickle
import yaml

import numpy as np


class BuildModel:

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.SAVE_DIR = os.path.join(self.conf["storage"], "pickle")

        self.flip = FashionCLIP("fashion-clip")


    def MakePickle(self, keys: list[int], paths: list[str]) -> list[str]:
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)
        
        embedding = self.flip.encode_images(paths, batch_size=32) # (1, 512) -> flatten -> (512,) -> flatten이 문젠가?
        embedding /= np.linalg.norm(embedding) # 정규화
        
        pickles = []

        for key, output_emb in zip(keys, embedding):
            with open(os.path.join(self.SAVE_DIR, f"{key}.pkl"), "wb") as f:
                pickles.append(os.path.join(self.SAVE_DIR, f"{key}.pkl"))
                pickle.dump(output_emb, f)

        return pickles
