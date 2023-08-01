# fastapi
from fastapi import Form
from fastapi import APIRouter

# built-in library
import os
import os.path as osp
import subprocess


router = APIRouter(
    prefix="/densepose",
    tags=["densepose"]
)


@router.post("")
def densepose_estimation(storage_root: Form(...), img_name: Form(...)):
    # det2 패키지로 동일한 결과를 출력하기 위해 일종의 치트를 사용
    os.chdir("/opt/ml/VIT-ON-Demo/backend/detectron2/projects/DensePose")

    config_file = "./configs/densepose_rcnn_R_50_FPN_s1x.yaml"
    model_file = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    image_path = osp.join(storage_root, "raw_data/person", img_name)

    # command를 실행
    command = ["python3",
               "apply_net.py",
               "show",
               config_file,
               model_file,
               image_path,
               "dp_segm"]
    
    subprocess.run(command)

    save_path = osp.join(storage_root, "preprocess/densepose", img_name)
    save_state = False
    if osp.exists(save_path) and osp.getsize(save_path):
        save_state = True
    
    return save_state