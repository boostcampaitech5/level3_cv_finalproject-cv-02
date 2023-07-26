# streamlit
import streamlit as st

# external-library
from PIL import Image

# built-in library
import io

# custom-modules
from utils.open import open_img_from_url
from meotandard_apis import MeotandardViton


# Viton 관련 메소드들이 정의되어있는 클래스 객체 생성
meotandard_viton = MeotandardViton()

   
if __name__ == "__main__":
    st.title(":art: 가상 피팅 서비스")
    # 상품 이미지 url을 데이터로 만들어서 저장 API 요청
    c_img_bytes = open_img_from_url(st.session_state.tryon_cloth)
    c_state, c_img_name = meotandard_viton.querying_saveimg_api(c_img_bytes, mode='cloth')

    # 상품 사진 마스킹 API 호출
    cm_state = meotandard_viton.querying_ppcloth_api(c_img_name)

    # TODO: 예시 이미지나, 가이드라인 보여주기
    # 사용자의 전신 사진 업로드
    st.header(":man_dancing: 정면에서 찍은 전신 사진을 올려주세요!")
    user_img = st.file_uploader("tmp", label_visibility="collapsed")

    if user_img is not None:
        with st.columns([0.2, 0.6, 0.2])[1]:
            st.image(user_img, use_column_width=True)

        if st.button("이 사진을 사용하시겠어요?", use_container_width=True):
            # 사용자의 사진을 서버 스토리지에 저장
            p_state, p_img_name = meotandard_viton.querying_saveimg_api(user_img, mode='person')

            # 사람 사진 전처리 요청
            state_dict = meotandard_viton.querying_ppperson_api(p_img_name, model_name='ladi-viton')
            # st.write(state_dict)

            # TODO: 생성 API 요청
            # st.header(":art: 입어보기 버튼을 눌러주세요!")        

            # if st.button("입어보기!", use_container_width=True):
            tryon_img = meotandard_viton.querying_genvitonimg_api(p_img_name, c_img_name, model_name='ladi_viton', category='upper_body')

            col1, col2 = st.columns([0.35, 0.65])

            with col1:
                st.image(st.session_state.tryon_cloth, use_column_width=True)
                st.image(user_img, use_column_width=True)

            with col2:
                st.image(tryon_img, use_column_width=True)