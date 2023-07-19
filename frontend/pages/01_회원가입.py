# streamlit
import streamlit as st
import streamlit_authenticator as stauth

# built-in library
import yaml


if __name__ == "__main__":
    # 회원가입 시 받을 값들을 변경하려면 stauth 라이브러리의 authenticate 내부를 변경하면 됨
    try:
        # authenticator.register -> bool(True or False) : 계정이 제대로 만들어졌다면 True, 아니면 False
        if st.session_state.authenticator.register_user("회원가입", preauthorization=False):
            # 가입한 정보 저장
            # stauth github을 보면, authenticator의 특정 메소드들을 사용했을 때 무조건 yaml 파일을 새로 저장하라고 얘기함.
            with open("./accounts.yaml", 'w') as file:
                yaml.dump(st.session_state.accounts, file, default_flow_style=False)

            st.success("회원 가입이 완료되었습니다!")

    except Exception as e:
        st.error(e)
    