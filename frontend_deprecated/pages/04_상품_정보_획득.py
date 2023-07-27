# streamlit
import streamlit as st
from streamlit_image_select import image_select
from streamlit_extras.switch_page_button import switch_page
from streamlit_image_coordinates import streamlit_image_coordinates

# external-library
from PIL import Image

# built-in library
import requests
import os.path as osp
import io
import base64

# custom-modules
from utils.management import ManageSessionState as MSS
from meotandard_apis import MeotandardSeg


# 필요한 state 초기화
MSS.init_session_state([("result", None),
                        ("segmented_img", None),
                        ("segmented_byte_img", None)])

# Segmentation API 관련 메소드들이 정의되어있는 클래스 객체 생성
meotandard_seg = MeotandardSeg()


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title(":tshirt: 상품 검색 서비스")
        st.markdown('---')
        
        # 사진 업로드
        st.subheader("검색을 원하는 상품 사진을 업로드해주세요!")
        uploaded_img = st.file_uploader("tmp", type=["png", "jpg", "jpeg"], label_visibility='collapsed')

        # 사용자가 파일을 업로드했다면
        if uploaded_img is not None:
            # TODO: 마크다운 꾸미기
            st.markdown("""
                        업로드한 이미지에서 검색하고 싶은 상품의 위치를 마우스:mouse:로 클릭해주세요!\n
                        :sparkles: **보다 정확한 검색 결과를 얻고 싶으시면 프린팅을 제외한 곳을 클릭해주세요 :-)** :sparkles:
                        """)
            
            # 시각화
            img = Image.open(uploaded_img)
            img_w, img_h = img.size
            
            # 리사이즈
            # TODO: 하드코딩 되어있는 부분을 변경하기 -> 정해진 값보다는, 비율로 줄이는 것이 더 안전해보임
            # TODO: w나 h가 1000이 넘으면 (700, 1000)이라서 center로 보이긴 하는데, 그거보다 작으면 왼쪽에 align되는 문제를 해결하기
            if img_w > 1000 or img_h > 1000:
                img = img.resize((700, 1000))
            
            # TODO: 좌표 찍을 때 위치를 이미지 위에 그릴 수 있는 것으로 보임 (https://github.com/blackary/streamlit-image-coordinates/blob/main/pages/dynamic_update.py)
            # 좌표 설정
            coordinates = streamlit_image_coordinates(
                img,
                key = "lacal2")

            # 좌표를 입력받으면
            if coordinates is not None:
                # PIL.Image 객체를 BytesIO 객체로 변환
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")

                # segmentation api 호출
                segmented_imgs, seg_and_pad_imgs, segmented_byte_imgs = meotandard_seg.querying_seg_api(img_bytes, coordinates)
                
                st.markdown("""
                            ---
                            :thinking_face: 다음 결과 중 **검색하고 싶은 상품과 가장 유사한 이미지를 선택**해주세요.
                            """)
                
                # image select로 보여줄 때는 padding된 이미지로
                img_index = image_select("선택해주세요!", seg_and_pad_imgs, return_value='index')

                # TODO: 버튼을 클릭하지 않았을 때 페이지 전체가 다시 렌더링되는 것을 막기(기존 정보가 유지되도록)
                # 선택을 확정지으면
                if st.button("이걸로 검색하시겠어요?"):
                    # 상품 검색 페이지에서 사용할 수 있도록 선택한 이미지를 session state에 등록 
                    # 검색할 때는 딱 맞게 잘려진 이미지를 사용함
                    MSS.change_session_state([('segmented_img', segmented_imgs[img_index]),
                                              ('segmented_byte_img', segmented_byte_imgs[img_index])])

                    # 상품 검색 페이지로 이동
                    switch_page("상품 검색")

    else:
        st.header("로그인 후 이용해주세요.")

    