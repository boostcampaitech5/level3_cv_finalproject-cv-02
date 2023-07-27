# streamlit
import streamlit as st
from streamlit_image_select import image_select
import requests

# built-in library
import os.path as osp

# custom library

STORAGE = "/opt/ml/storage/raw_data/"


def init_setting():    
    if 'button' not in st.session_state:
        st.session_state.button = False

    if 'person_img_name' not in st.session_state:
        st.session_state['person_img_name'] = None

    if 'cloth_img_name' not in st.session_state:
        st.session_state['cloth_img_name'] = None


def click_button():
    st.session_state.button = not st.session_state.button


if __name__ == "__main__":
    user = init_setting()
    st.title("가상 피팅 서비스")

    if st.button("server test"):
        res = requests.get(url="http://127.0.0.1:8000/")
        st.write(res.text)

    # 1. 사람의 정면 사진 업로드
    st.header(":man_dancing: 정면에서 찍은 전신 사진을 올려주세요!")
    st.button("예시", on_click=click_button)

    if st.session_state.button:
        st.image(osp.join("/opt/ml/data/test", 'image', '00006_00.jpg'))
    
    user_img = st.file_uploader(label="upload an image")

    if user_img is not None:
        st.image(user_img)

        if st.button("이 사진을 사용하시겠어요?"):
            files = {"img": user_img}
            res = requests.post(url=f"http://127.0.0.1:8000/proxy/save", files=files)

            # person_img_save_state = res.json()['save_state']
            # person_img_name = res.json()['im_name']
            # st.session_state['person_img_name'] = person_img_name
            
            # if person_img_save_state: # 올바르게 저장되었다면
            #     res = requests.post(url=f"http://127.0.0.1:8000/proxy/preprocess/person?img_name={person_img_name}")
            #     # st.write(res.json())
    
    
    # 2. 아카빙한 상품 이미지 중 택 1
    st.header(":shirt: 어떤 옷을 입어보고 싶으신가요?")

    # 파이프라인 통합 과정에서 상품이 저장되어 있는 경로가 달라질 수 있음
    cloth_img_path = image_select(
        label='Select a cloth',
        images=[
            osp.join(STORAGE, 'cloth', 'musinsa_01.jpg'),
            osp.join(STORAGE, 'cloth', 'musinsa_02.jpg'),
            osp.join(STORAGE, 'cloth', 'musinsa_03.jpg'),
            osp.join(STORAGE, 'cloth', 'musinsa_04.jpg'),
            osp.join(STORAGE, 'cloth', 'musinsa_05.jpg'),
            osp.join(STORAGE, 'cloth', 'musinsa_06.jpg')
        ]
    )

    st.image(cloth_img_path)

    if st.button("이 상품을 입어보시겠어요?"):
        cloth_img_name = cloth_img_path.split('/')[-1]
        st.session_state['cloth_img_name'] = cloth_img_name
        res = requests.post(url=f"http://127.0.0.1:8000/proxy/preprocess/cloth?img_name={cloth_img_name}")
        pp_cloth_state = res.text

    
    # 3. 사람 이미지 및 상품 이미지 파일 이름을 가지고 백엔드에 생성 요청
    st.header(":art: 입어보기 버튼을 눌러주세요!")

    if st.button('입어보기!'):
        res = requests.post(url=f"http://127.0.0.1:8000/proxy/generate?p_img_name={st.session_state['person_img_name']}&c_img_name={st.session_state['cloth_img_name']}")
        img_name = res.text.replace('"', '').replace("\\", "")
        st.image(osp.join('/opt/ml/storage/viton', img_name))
        