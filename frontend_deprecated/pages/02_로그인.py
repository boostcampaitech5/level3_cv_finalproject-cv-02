# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page


if __name__ == "__main__":
    # TODO: login 시 예외처리
    # name, authentication_status, username, logout은 authenticator 객체를 선언할 때 자동으로 session state에 등록됨.
    name, authentication_status, username = st.session_state.authenticator.login('로그인', 'main')
    
    if st.session_state.authentication_status:
        switch_page("옷장")
    elif st.session_state.authentication_status == False:
        st.error('아이디/비밀번호를 확인해주세요.')
    elif st.session_state.authentication_status == None:
        st.warning('아이디와 비밀번호를 입력해주세요.')

    # 계정이 없는 경우 회원가입
    st.subheader("계정이 없으시다면? 가입하기 버튼을 클릭해주세요!")
    if st.button("가입하기"):
        switch_page("회원가입")