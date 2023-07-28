from fastapi import APIRouter

from infra.search.model import SearchModel
from models.search import SearchParm, SearchWithFilterParam

search_router = APIRouter(
    tags=["Search"],
)

model = SearchModel(config_path="../config.yaml")

# -- get을 써야해, post를 써야해? => "요청바디 사용" => post
# (text or image의 단일한 embedding)을 처리하는 라우터로 변경해야함.
@search_router.post("")
async def search(parm: SearchParm) -> dict:
    
    dists, ids = model.search(parm.embedding, parm.thresh) 
    
    # k = 50
    # if len(dists) > k:
    #     dists = dists[:k]
    #     ids = ids[:k]

    return {
        "msg": "Search OK!",
        "dists": dists,
        "ids": ids,
    }
    

@search_router.post("/with-filter")
async def search_with_filter(parm: SearchWithFilterParam) -> dict:
    
    dists, ids = model.search_order_by_filter(parm.embedding, parm.filter_embedding, parm.thresh)
    
    # k = 10
    # if len(dists) > k:
    #     dists = dists[:k]
    #     ids = ids[:k]

    return {
        "msg": "Search OK!",
        "dists": dists,
        "ids": ids,
    }