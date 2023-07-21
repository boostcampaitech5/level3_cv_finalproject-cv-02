# streamlit
import streamlit as st

# external-library
import pandas as pd

# built-in library
import requests

# custom-modules
from meotandard_apis import MeotandardRetrieval

# Retrieval API 관련 메소드들이 정의되어있는 클래스 객체 생성
meotandard_retrieval = MeotandardRetrieval()


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title(":tshirt: 상품 검색 서비스")
        st.markdown('---')

        # 선택한 상품 이미지를 보여주기
        with st.columns(3)[1]:
            st.image(st.session_state.segmented_img, use_column_width=True)

        # TODO: text 사용하고 싶지 않은 경우 search-by-image api를 호출하도록 코드 작성
        # TODO: 색상을 어떻게 입력받을지 얘기해보고 정하기
        # 상품 검색에 활용할 text 입력
        text = st.text_input("상품 검색에 사용할 옷의 색상 정보를 영어로 입력해주세요!")

        if text:
            # threshold는 default == 0.0
            clothes_metadata = meotandard_retrieval.querying_searchbyfilter_api(st.session_state.segmented_byte_img, text)

            st.header("상품 검색 결과입니다!")
            st.subheader("(상품 이미지를 클릭하면 구매 링크로 이동합니다 :airplane_departure:)")
            st.markdown('---')

            # 2 x 5 그리드 형식으로 이미지 보여주기
            for i in range(2):
                cols = st.columns(5)

                for j in range(5):
                    idx = i * 5 + j

                    cloth_metadata = clothes_metadata[idx]
                    img_url = cloth_metadata['image_link']
                    link_url = cloth_metadata['link']
                    link = f'<a href="{link_url}" target="_blank"><img src="{img_url}" alt="Image" width="150" height="200"></a>'

                    cols[j].markdown(link, unsafe_allow_html=True)
                    
    else:
        st.header("로그인 후 이용해주세요.")