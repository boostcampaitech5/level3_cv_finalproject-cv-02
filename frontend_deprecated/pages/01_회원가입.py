# streamlit
import streamlit as st
import streamlit_authenticator as stauth

# custom-modules
from utils.open import open_yaml


if __name__ == "__main__":
    try:
        # authenticator.register_user -> bool(True or False) : 계정이 제대로 만들어졌다면 True, 아니면 False
        if st.session_state.authenticator.register_user("회원가입", preauthorization=False):
            # 가입한 정보 저장
            st.session_state.accounts = open_yaml("./accounts.yaml", 'w')
            st.success("회원 가입이 완료되었습니다!")

    except Exception as e:
        st.error(e)
    