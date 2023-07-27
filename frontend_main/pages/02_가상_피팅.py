# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# custom-modules
from utils.open import open_img_from_url
from utils.open import open_page
from utils.management import ManageSessionState as MSS
from meotandard_apis import MeotandardViton


MSS.init_session_state([("user_img", None),
                        ("tryon_img", None)])

# Viton 관련 메소드들이 정의되어있는 클래스 객체 생성
meotandard_viton = MeotandardViton()


def main():
    # 상품 이미지 url을 데이터로 만들어서 저장 API 요청
    c_img_bytes = open_img_from_url(st.session_state.tryon_cloth_data["image_link"])
    c_state, c_img_name = meotandard_viton.querying_saveimg_api(c_img_bytes, mode='cloth')

    # 상품 사진 마스킹 API 호출
    cm_state = meotandard_viton.querying_ppcloth_api(c_img_name)

    # 사용자의 전신 사진 업로드
    st.header(":man_dancing: 정면에서 찍은 사진을 올려주세요!")
    st.image("./figure/viton_guideline.png", use_column_width=True)
    user_img = st.file_uploader("tmp", label_visibility="collapsed")

    # 새로운 사진이 올라오면
    if user_img != st.session_state.user_img:
        MSS.change_session_state([("user_img", user_img),
                                  ("tryon_img", None)])

    if st.session_state.user_img is not None:
        with st.columns([0.2, 0.6, 0.2])[1]:
            st.image(user_img, use_column_width=True)

        if st.button("이 사진을 사용하시겠어요?", use_container_width=True):
            # 사용자의 사진을 서버 스토리지에 저장
            p_state, p_img_name = meotandard_viton.querying_saveimg_api(user_img, mode='person')

            # 사람 사진 전처리 요청
            state_dict = meotandard_viton.querying_ppperson_api(p_img_name, model_name='ladi-viton')
            # st.write(state_dict)

            st.session_state.tryon_img = meotandard_viton.querying_genvitonimg_api(p_img_name, c_img_name, model_name='ladi_viton', category='upper_body')
            

        col1, col2 = st.columns([0.35, 0.65])
        with col1:
            st.subheader("상품 정보")
            # 상품 관련 데이터
            product_img_url = st.session_state.tryon_cloth_data["image_link"]
            product_name = st.session_state.tryon_cloth_data["name"]
            product_price = st.session_state.tryon_cloth_data["price"]
            product_url = st.session_state.tryon_cloth_data["link"]

            # 상품 사진
            st.image(product_img_url, use_column_width=True)
            
            # 상품명
            st.markdown(f"**상품명**  \n{product_name}")

            # 상품 가격
            st.markdown(f"**가격**  \n{product_price}")

            # 구매 링크
            st.button("상품 사이트", on_click=open_page,
                        args=(product_url,), use_container_width=True)
            
            if st.button("다른 옷도 입어볼래요", use_container_width=True):
                MSS.change_session_state([(st.session_state.tryon_cloth_data, None)])
                switch_page("상품 찾기")

        with col2:
            if st.session_state.tryon_img is not None:
                st.image(st.session_state.tryon_img, use_column_width=True)

   
if __name__ == "__main__":
    if ('tryon_cloth_data' not in st.session_state) or (st.session_state.tryon_cloth_data == None):
        st.warning("상품 검색 기능에서 가상 피팅 서비스를 이용해주세요.")
    else:
        st.title(":art: 가상 피팅 서비스")
        main()
    