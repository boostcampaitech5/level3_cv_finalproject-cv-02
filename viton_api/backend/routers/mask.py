# fastapi
from fastapi import Form
from fastapi import APIRouter

# custom-library
from infra.preprocess.masking.model import TracerB7


router = APIRouter(
    prefix="/mask",
    tags=["mask"]
)


model = TracerB7()


@router.post("/inference")
def predict_mask(storage_root: str = Form(...), img_name: str = Form(...), mode: str = Form(...)) -> bool:
    """마스크를 예측할 때 사용하는 API입니다.

    Args:
        storage_root (str, optional): 서버의 로컬 스토리지 root 주소입니다. Defaults to Form(...).
        img_name (str, optional): 마스킹할 대상 이미지의 이름입니다. Defaults to Form(...).
        mode (str, optional): 마스킹 결과를 저장할 때 사용하는 변수입니다. Defaults to Form(...).

    Returns:
        bool: 마스킹 결과의 상태를 반환합니다.
    """
    if mode not in ['cloth', 'person']:
        print("Available mode: cloth, person")    
        raise

    save_state = model.inference(storage_root, img_name, mode)

    return save_state