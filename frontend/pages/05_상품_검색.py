# streamlit
import streamlit as st

# built-in library
import requests


color = {''}


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title("상품 검색 서비스")

        # TODO: 상품 이미지를 중간 정렬하기, session state에 저장된 이미지 초기화가 안되는 문제 해결
        # 선택한 상품 이미지를 보여주기
        st.image(st.session_state.segmented_img)

        # TODO: text 사용하고 싶지 않은 경우 search-by-image api를 호출하도록 코드 작성
        # 상품 검색에 활용할 text 입력
        # TODO: 색상을 어떻게 입력받을지 얘기해보고 정하기
        text = st.text_input("상품 검색에 사용할 옷의 색상 정보를 영어로 입력해주세요!")

        if text:
            threshold = 0.0
            files = {'file': st.session_state.segmented_byte_img}
            form_data = {'thresh': threshold, 'text': text}

            response = requests.post(url=f'{st.session_state.retrieval}/proxy/search-by-filter', files=files, data=form_data)
            st.write(response.json())
        
        st.stop()
    else:
        st.header("로그인 후 이용해주세요.")