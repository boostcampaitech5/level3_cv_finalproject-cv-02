# streamlit
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_extras.switch_page_button import switch_page

# built-in library
import yaml
from yaml.loader import SafeLoader

# custom-modules
from utils import ManageSessionState


login_session_state = ManageSessionState()


if __name__ == "__main__":
    # 회원 정보 불러오기
    with open("./accounts.yaml") as file:
        accounts = yaml.load(file, Loader=SafeLoader)

    # 사용자 인증을 위해 선언
    authenticator = stauth.Authenticate(
        accounts['credentials'],
        accounts['cookie']['name'],
        accounts['cookie']['key'],
        accounts['cookie']['expiry_days'],
        accounts['preauthorized']
    )

    # 필요한 sesion state 초기화
    login_session_state.init_session_state([('accounts', accounts),
                                            ('authenticator', authenticator)])
    
    # name, authentication_status, username, logout은 authenticator 객체를 선언할 때 자동으로 session state에 등록됨.
    name, authentication_status, username = st.session_state.authenticator.login('로그인', 'main')
    
    if authentication_status:
        switch_page("옷장")
    elif authentication_status == False:
        st.error('아이디/비밀번호를 확인해주세요.')
    elif authentication_status == None:
        st.warning('아이디와 비밀번호를 입력해주세요.')

    # 계정이 없는 경우 회원가입
    st.subheader("계정이 없으시다면? 가입하기 버튼을 클릭해주세요!")
    if st.button("가입하기"):
        switch_page("회원가입")