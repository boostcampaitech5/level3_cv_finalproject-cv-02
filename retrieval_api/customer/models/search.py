from pydantic import BaseModel


class SearchParm(BaseModel):
    
    embedding: list[float]
    thresh: float
    

class SearchWithFilterParam(BaseModel):
    
    embedding: list[float]
    filter_embedding: list[float]
    thresh: float