# streamlit
import streamlit as st
from streamlit.components.v1 import html

# external-library
from PIL import Image

# built-in library
import yaml
from yaml.loader import SafeLoader
import requests
import io


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


# https://discuss.streamlit.io/t/streamlit-button-with-link/41171
def open_page(url):
    open_script = """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)


def open_img_from_url(img_url: str):
    try:
        # 이미지 데이터 가져오기
        response = requests.get(img_url)
        response.raise_for_status()  # 에러가 발생했는지 확인
        
        img_bytes = io.BytesIO(response.content)

        return img_bytes
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 오류가 발생했습니다: {http_err}")
    except Exception as err:
        print(f"에러가 발생했습니다: {err}")


# Test Code
if __name__ == "__main__":
    img_url = "https://image.msscdn.net/images/goods_img/20210520/1959696/1959696_16819694709230_500.jpg"
    open_img_from_url(img_url)