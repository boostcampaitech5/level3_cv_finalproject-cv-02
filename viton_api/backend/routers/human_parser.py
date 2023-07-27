# fastapi
from fastapi import Form
from fastapi import APIRouter

# custom-library
# from infra.preprocess.human_parser.cihp_pgn.model import CIHP_PGN
from infra.preprocess.human_parser.schp.model import SCHP
from routers.utils import get_config

router = APIRouter(
    prefix="/human_parser",
    tags=["human_parser"]
)


schp = SCHP(get_config())


# TODO: CIHP_PGN을 조건에 따라 적용할 수 있는 코드 생각해보기(tf라 가상환경이 달라져서 까다로움)
@router.post("")
def human_parsing(storage_root: str = Form(...), img_name: str = Form(...)) -> bool:
    """사람의 사진으로부터 각 신체 부위에 대한 seg map을 예측할 때 사용하는 API 입니다.

    Args:
        storage_root (str, optional): 서버의 로컬 스토리지 root 주소입니다. Defaults to Form(...).
        img_name (str, optional): parsing할 대상 이미지의 이름입니다. Defaults to Form(...).

    Returns:
        bool: human parsing 결과의 상태를 반환합니다.
    """
    save_state = schp.inference(storage_root, img_name)

    return save_state