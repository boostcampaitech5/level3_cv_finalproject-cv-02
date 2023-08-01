import faiss
import os
import numpy as np
import pickle
import yaml

from .utils import cos_sim

'''
faiss 사용법 참고(https://lsjsj92.tistory.com/605)
주의) 2단계에 걸쳐서 진행, 변수는 웬만해서는 np.array로 들어간다. 
1) output_embeddings에 id 부여 (현재는 output_embeddigs를 pkl로 저장함)
index.add_with_ids(output_embs, ids) -> 각 임베딩과 id를 매칭시켜주는 작업
2) 입력으로 들어온 input_embedding과 유사도 검색
result = index.search(input_emb.reshape(1, -1), 2) <- input_emb: np.array
(array([[94.22209, 89.08495]], dtype=float32), array([[32, 34]]))
'''

class SearchModel:
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.DATA_DIR = os.path.join(self.conf["storage"], "pickle") # embedding -> pickle로 수정할 것!
        
        index = faiss.IndexFlatIP(512) #--embedding의 차원 수,
        self.index = faiss.IndexIDMap2(index) # --embedding에 id를 부여하기위해서
    
        ids = []
        embs = []
        for filename in os.listdir(self.DATA_DIR):
            id = filename.split(".")[0] # 000.pkl
            ids.append(id)
            with open(os.path.join(self.DATA_DIR, filename), "rb") as f:
                emb = pickle.load(f)
                embs.append(emb)
        print("여기는 id..개수",len(ids))
        print("여기는 embs..개수",len(embs))
        self.index.add_with_ids(np.array(embs), np.array(ids))
        
        
    def search(self, embedding: list[float], thresh: float) -> tuple[list[float], list[int]]:
        dists, ids = self.index.search(np.array(embedding).reshape(1, -1), 50)
        print("잘 되지?", "여기는 search!")
        
        indices = np.where(dists >= thresh)[0]
        k = len(indices)
        dists = dists[:k]
        ids = ids[:k]
            
        return dists.flatten().tolist(), ids.flatten().tolist()
        
        
    def search_order_by_filter(self, embedding: list[float], filter_embedding: list[float], thresh: float) -> tuple[list[float], list[int]]:
        dists, ids = self.index.search(np.array(embedding).reshape(1, -1), 50)
        print("잘 되지?", "여기는 search_order_by_filter야!")
        
        indices = np.where(dists >= thresh)[0]
        k = len(indices)
        dists = dists[:k]
        ids = ids[:k]
        
        dists = dists.flatten().tolist()
        ids = ids.flatten().tolist()
        
        results = [(cos_sim(np.array(filter_embedding), self.index.reconstruct(id)), id) for id in ids]
        results.sort(reverse=True)
        
        return tuple(list(x) for x in zip(*results))