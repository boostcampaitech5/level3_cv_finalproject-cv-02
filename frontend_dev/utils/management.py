# streamlit
import streamlit as st
import streamlit_authenticator as stauth

# built-in library
from typing import *

# custom-modules
from utils.open import open_yaml


class ManageSessionState:
    def get_api_adress(self, api_address_path: str = "./api_address.yaml"):
        # api별 ip 주소가 담겨있는 yaml 파일을 열어서, sesion state에 등록
        api_address = open_yaml(api_address_path)
        self.init_session_state(list(api_address.items()))
    
    @staticmethod
    def get_authenticator(accounts_path: str = "./accounts.yaml"):
        # 고객 계정 정보가 담겨있는 yaml 파일을 열고,
        accounts = open_yaml(accounts_path)
        
        # login 기능을 위해 authenticator를 선언한 뒤,
        authenticator = stauth.Authenticate(
            accounts['credentials'],
            accounts['cookie']['name'],
            accounts['cookie']['key'],
            accounts['cookie']['expiry_days'],
            accounts['preauthorized']
        )

        # session state에 등록
        st.session_state.authenticator = authenticator
        
    @staticmethod
    def call_login(pos: str = 'main'):
        """새로고침으로 인해 모든 session state가 사라지는 경우를 대비하여,
        쿠키 기능이 존재하는 stauth.Authenticate.login() 기능을 사용,
        잠깐 동안 login 기능을 불러온 뒤 원래 서비스를 사용할 수 있도록 돕는 함수입니다.
        """
        if st.session_state.authentication_status == None:
            _ = st.session_state.authenticator.login('로그인', pos)

    @staticmethod
    def init_session_state(states: List[Tuple[str, Any]]) -> None:
        """streamlit으로 프로그램을 실행했을 때,
        모든 페이지에서 공유할 수 있는 session state 값을 초기화하기 위해
        사용하는 함수입니다(Python의 전역변수 느낌)

        Args:
            states (List[Tuple[str, Any]]): [(str, Any)] 형태이며
            앞쪽엔 session state의 이름을, 뒤쪽엔 초기화할 값을 아무거나 입력해주세요.
        """
        for name_state, val_state in states:
            if name_state not in st.session_state:
                st.session_state[name_state] = val_state

    @staticmethod
    def change_session_state(states: List[Tuple[str, Any]]) -> None:
        """session state의 값이 변경된 경우, 이를 반영하기 위해 사용하는 함수입니다.

        Args:
            states (List[Tuple[str, Any]]): 앞쪽엔 session state의 이름을, 뒤쪽엔 초기화할 값을 아무거나 입력해주세요.
        """
        for name_state, val_state in states:
            st.session_state[name_state] = val_state