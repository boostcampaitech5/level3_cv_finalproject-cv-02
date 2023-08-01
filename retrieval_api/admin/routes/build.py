from fastapi import APIRouter, File, UploadFile, Response, Form
from typing import Annotated

from infra.build.model import BuildModel
from models.build import BuildParam

build_router = APIRouter(
    tags=["Build"]
)

build = BuildModel(config_path="../config.yaml")

@build_router.post("/make-pickle")
async def make_pickle(parm: BuildParam) -> dict:
    pickles = build.MakePickle(parm.keys, parm.paths)

    return {
        "msg": "pickle 저장 성공!",
        "pickles": len(pickles)
    }
