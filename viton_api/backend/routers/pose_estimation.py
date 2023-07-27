# fastapi
from fastapi import Form
from fastapi import APIRouter

# custom-library
from infra.preprocess.pose_estimation.body25 import OpenPoseBody25


router = APIRouter(
    prefix="/pose_estimation",
    tags=["pose_estimation"]
)


body25 = OpenPoseBody25()


@router.post("")
def get_pose_keypoints(storage_root: str = Form(...),
                       img_name: str = Form(...),
                       model_name: str = Form(...)) -> bool:
    """사람 신체의 pose kpts를 예측할 때 사용하는 API입니다.

    Args:
        storage_root (str, optional): 서버의 로컬 스토리지 root 주소입니다. Defaults to Form(...).
        img_name (str, optional): pose estimation 대상 이미지의 이름입니다. Defaults to Form(...).
        model_name (str, optional): 모델의 이름에 따라 저장하는 데이터가 달라집니다. Defaults to Form(...).

    Returns:
        bool: pose estimation 결과의 상태를 반환합니다.
    """
    save_state = body25.inference(storage_root, img_name, model_name)

    return save_state