# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# custom-modules
from utils.management import ManageSessionState as MSS


def main():
    if st.session_state.authentication_status:
        switch_page("옷장")

    if st.session_state.authentication_status == False:
        st.error('아이디/비밀번호를 확인해주세요.')

    if st.session_state.authentication_status == None:
        st.warning('아이디와 비밀번호를 입력해주세요.')

    # 계정이 없는 경우 회원가입
    st.subheader("계정이 없으시다면? 가입하기 버튼을 클릭해주세요!")

    if st.button("가입하기"):
        switch_page("회원가입")


if __name__ == "__main__":
    # 새로고침 대비 코드
    MSS.get_authenticator()
    MSS.call_login()
    
    # 로그인 페이지 동작
    main()