# fast-api
from fastapi import File, UploadFile, Form
from fastapi import APIRouter

# external-library
from PIL import Image

# built-in library
import os.path as osp
import warnings
import uuid
import io

# custom-library
from infra.preprocess.masking.client import MaskingClient
from infra.preprocess.pose_estimation.client import PoseEstimationClient
from infra.preprocess.human_parser.schp.client import HumanParsingClient
from infra.preprocess.densepose.client import DensePoseClient
from infra.viton.client import VitOnClient

warnings.filterwarnings('ignore')

# 기본 설정값
configs = {
    "storage": "/opt/ml/storage",
    "available_model": {"viton": ["hr_viton", "ladi_viton"]}
}


# 라우터 설정
router = APIRouter(
    prefix="/proxy",
    tags=["proxy"])


# 클라이언트 선언
mask_client = MaskingClient()
pose_estimation_client = PoseEstimationClient()
human_parsing_client = HumanParsingClient()
dense_pose_client = DensePoseClient()
viton_client = VitOnClient()


# TODO: API 요청이 올바르게 이루어졌는지 판단하는 state를 보다 논리적인 방법으로 바꾸기
@router.post("/save")
async def save_img(img: UploadFile = File(...), mode: str = Form(...)) -> dict:
    """서버의 로컬 스토리지에 png 형식으로 이미지를 저장할 때 사용하는 API입니다.

    Args:
        img (UploadFile, optional): 이미지 데이터입니다. Defaults to File(...).
        mode (str, optional): mode에 따라 데이터를 다르게 읽습니다. Defaults to Form(...).

    Returns:
        dict: 저장 상태를 나타내는 state와, 저장 시 사용했던 이미지의 이름입니다.
    """
    if mode == 'cloth':
        file_content = await img.read()
        im = Image.open(io.BytesIO(file_content))
    elif mode == 'person':
        im = Image.open(img.file)
    else:
        print('Available mode: cloth, person')
        raise
    
    im_name = str(uuid.uuid4()) + '.png'
    im_path = osp.join(configs['storage'], f'raw_data/{mode}', im_name)
    im.save(im_path, 'PNG')

    save_state = False
    if osp.exists(im_path) and osp.getsize(im_path):
        save_state = True

    return {"save_state": save_state, "im_name": im_name}


@router.post("/preprocess/cloth")
async def pp_cloth(img_name: str = Form(...)) -> str:
    """서버의 로컬 스토리지에 저장되어있는 상품 이미지의 이름을 바탕으로
    마스킹 클라이언트에게 마스크 예측을 요청하는 API입니다.

    Args:
        img_name (str, optional): 상품 이미지의 이름입니다. Defaults to Form(...).

    Returns:
        str : 저장 상태를 나타내는 state를 반환합니다.
    """
    save_state = await mask_client.predict_mask(configs['storage'], img_name, mode='cloth')

    return save_state


# TODO: hr-viton과 ladi-vton이 사용하는 human parser가 다른 환경, 다른 모델임 -> 조건에 따라 요청하도록 수정
@router.post("/preprocess/person")
async def pp_person(img_name: str = Form(...), model_name: str = Form(...)) -> dict:
    """서버의 로컬 스토리지에 저장되어있는 사람 이미지의 이름을 바탕으로
    생성에 필요한 모든 전처리 과정을 수행하는 API입니다.

    Args:
        img_name (str, optional): 사람 이미지의 이름입니다. Defaults to Form(...).
        model (str, optional): 사용할 생성 모델의 이름입니다. 
        230722 기준 hr-viton, LaDi-vton 모델을 지원합니다. Defaults to Form(...).

    Returns:
        dict : 모든 전처리 과정에 대한 상태가 담긴 dictionary입니다.
    """
    parse_map_save_state = await human_parsing_client.human_parsing(configs['storage'], img_name)

    if model_name == "hr_viton":
        dense_pose_save_state = await dense_pose_client.predict_dense_pose_map(configs['storage'], img_name)
    else:
        dense_pose_save_state = False

    mask_save_state = await mask_client.predict_mask(configs['storage'], img_name, mode='person')

    if mask_save_state:
        pose_save_state = await pose_estimation_client.predict_pose_kpts(configs['storage'], img_name, model_name)

    return {"parse map": parse_map_save_state, "dense pose map": dense_pose_save_state,
            "mask": mask_save_state, "pose img & kpts": pose_save_state}


# TODO: category에 따라 사용하는 weights가 다른 ladi 모델을 backend에서 어떻게 사용할지 고민해보기
@router.post("/generate")
async def gen_viton_img(p_img_name: str = Form(...), c_img_name: str = Form(...),
                        model_name: str = Form(...), category: str = Form(...)):
    
    if model_name not in configs['available_model']['viton']:
        print(f"Available model: {configs['available_model']['viton']}")
        raise
    
    tryon_img_base64 = await getattr(viton_client, model_name)(configs['storage'], p_img_name, c_img_name, category)

    return tryon_img_base64