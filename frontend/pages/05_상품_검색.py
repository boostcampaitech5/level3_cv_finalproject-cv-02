# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# external-library
import pandas as pd

# built-in library
import requests

# custom-modules
from meotandard_apis import MeotandardRetrieval
from utils.management import ManageSessionState as MSS
from utils.move import prev, next, open_page
from datas.cloth import SearchedCloth

# 필요한 session state 초기화
MSS.init_session_state([('counter', 0),
                        ('searched_cloth', list())])


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

            # TODO: clothes_metadata가 제대로 return되지 않았을 때 예외처리가 필요
            # TODO: 이미지 크기가 달라지면 button 위치가 자꾸 변하는 문제 해결
            # TODO: next, prev를 통해 이미지를 이동하면 처음부터 코드가 실행됨 -> 캐싱 혹은 별도로 페이지 만드는 것을 고려
            if clothes_metadata:
                searched_cloth = [SearchedCloth(cloth_metadata) for cloth_metadata in clothes_metadata]
                MSS.change_session_state([('searched_cloth', searched_cloth)])

                # 검색된 상품 개수 반환
                n_clothes = len(searched_cloth)

                # counter를 기반으로 하나의 상품 선택
                cloth = searched_cloth[st.session_state.counter % n_clothes]

                # 구매링크 이동 버튼
                container = st.empty()
                st.button("구매 링크로 이동", on_click=open_page, 
                          args=(cloth.purchase_link,), use_container_width=True)

                # 좌, 우 이동 버튼
                col1, col2 = st.columns(2)
                with col1: st.button("⬅️ 이전", on_click=prev, use_container_width=True)
                with col2: st.button("다음 ➡️", on_click=next, use_container_width=True)
                
                # 저장, 입어보기 버튼
                col1, col2 = st.columns(2)
                with col1: st.button("저장", use_container_width=True)
                with col2:
                    if st.button("입어보기", use_container_width=True):
                        switch_page("가상 피팅")

                with container.container():
                    # 검색된 상품 개수 설명
                    st.subheader(f"총 {n_clothes} 개의 상품이 검색되었습니다!")

                    # 이미지 시각화
                    with st.columns([0.1, 0.8, 0.1])[1]:
                        st.image(cloth.img_link, use_column_width=True)

            # st.header("상품 검색 결과입니다!")
            # st.subheader("(상품 이미지를 클릭하면 구매 링크로 이동합니다 :airplane_departure:)")
            # st.markdown('---')

            # # 2 x 5 그리드 형식으로 이미지 보여주기
            # for i in range(2):
            #     cols = st.columns(5)

            #     for j in range(5):
            #         idx = i * 5 + j

            #         cloth_metadata = clothes_metadata[idx]
            #         img_url = cloth_metadata['image_link']
            #         link_url = cloth_metadata['link']
            #         link = f'<a href="{link_url}" target="_blank"><img src="{img_url}" alt="Image" width="150" height="200"></a>'

            #         cols[j].markdown(link, unsafe_allow_html=True)
                    
    else:
        st.header("로그인 후 이용해주세요.")