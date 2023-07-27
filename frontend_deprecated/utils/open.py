# streamlit
import streamlit as st

# built-in library
import yaml
from yaml.loader import SafeLoader


@st.cache_data
def open_yaml(yaml_path: str, mode: str = 'r') -> dict:
    """yaml_path의 yaml 파일을 불러와 mode에 따른 처리를 수행하고,
    그 결과를 dictionary에 담아 반환합니다.

    Args:
        yaml_path (str): 불러올 yaml 파일의 경로
        mode (str, optional): yaml 파일을 처리할 때 사용할 모드이며, open()에 입력될 인자입니다.
        Defaults to 'r'.

    Returns:
        dict: yaml 파일 속 내용이 담긴 dictionary
    """
    result = None
    with open(yaml_path, mode) as file:
        if mode == 'r': # yaml 읽기
            result = yaml.load(file, Loader=SafeLoader)
        elif mode == 'w': # yaml 저장하기
            yaml.dump(st.session_state.accounts, file, default_flow_style=False)
        else:
            print("아직은 지원하지 않습니다.")

    return result
