# streamlit
import streamlit as st


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        pass
    else:
        st.subheader("가상 피팅 서비스는 오픈 준비중입니다! :person_doing_cartwheel:")
    