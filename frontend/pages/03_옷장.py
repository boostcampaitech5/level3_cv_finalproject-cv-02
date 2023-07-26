# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# custom-modules
from utils.management import ManageSessionState as MSS


def main():
    st.session_state.authenticator.logout('로그아웃', 'main')
    # TODO: 옷장 관련 기능들을 구현 (아마도 DB에서 불러올 것 같음)

    st.title(f'Welcome *{st.session_state.name}*!')
    st.subheader(f"*{st.session_state.name}*의 옷장")       

    st.write('Some content')
    st.markdown("![상품 이름](https://image.msscdn.net/images/goods_img/20230307/3129731/3129731_16817899185077_500.jpg)")


if __name__ == "__main__":
    # 새로고침으로 인한 에러 방지
    MSS.get_authenticator()
    MSS.call_login()

    # 로그인 여부에 따른 기능 제공
    if st.session_state.authentication_status == True:
        main()
    