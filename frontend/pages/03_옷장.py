# streamlit
import streamlit as st


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.session_state.authenticator.logout('로그아웃', 'main')
        # TODO: 옷장 관련 기능들을 구현

        st.title(f'Welcome *{st.session_state.name}*!')
        st.subheader(f"*{st.session_state.name}*의 옷장")       

        st.write('Some content')
        st.markdown("![상품 이름](https://image.msscdn.net/images/goods_img/20230307/3129731/3129731_16817899185077_500.jpg)")
    else:
        st.header("로그인 후 이용해주세요.")
    