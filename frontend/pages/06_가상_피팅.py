# streamlit
import streamlit as st

# custom-modules
from utils.open import open_img_from_url
from meotandard_apis import MeotandardViton


# Viton 관련 메소드들이 정의되어있는 클래스 객체 생성
meotandard_viton = MeotandardViton()


# 테스트용 이미지 주소(서비스 시에는 검색된 데이터가 session state에 등록될 것이므로 그걸 이용해서)
img_url = "https://image.msscdn.net/images/goods_img/20210520/1959696/1959696_16819694709230_500.jpg"

   
if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        # 상품 이미지 url을 데이터로 만들어서 저장 API 요청
        c_img_bytes = open_img_from_url(img_url)
        c_state, c_img_name = meotandard_viton.querying_saveimg_api(c_img_bytes, mode='cloth')

        # 상품 사진 마스킹 API 호출
        cm_state = meotandard_viton.querying_ppcloth_api(c_img_name)

        # TODO: 예시 이미지나, 가이드라인 보여주기
        # 사용자의 전신 사진 업로드
        st.header(":man_dancing: 정면에서 찍은 전신 사진을 올려주세요!")
        user_img = st.file_uploader("tmp", label_visibility=None)

        if user_img is not None:
            st.image(user_img)

            if st.button("이 사진을 사용하시겠어요?", use_container_width=True):
                # 사용자의 사진을 서버 스토리지에 저장
                p_state, p_img_name = meotandard_viton.querying_saveimg_api(user_img, mode='person')

                # 사람 사진 전처리 요청
                state_dict = meotandard_viton.querying_ppcloth_api(p_img_name, model='ladi-vton')
                st.write(state_dict)

        # TODO: 생성 API 요청
        st.header(":art: 입어보기 버튼을 눌러주세요!")        

        if st.button("입어보기!", use_container_width=True):
            pass

    else:
        st.subheader("가상 피팅 서비스는 오픈 준비중입니다! :person_doing_cartwheel:")
    