# streamlit
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_select import image_select
from streamlit_extras.switch_page_button import switch_page

# external-library
from PIL import Image
# import cv2
# import numpy as np

# built-in library
import requests
import os.path as osp
import io
import base64

# custom-library
from utils import ManageSessionState


seg_session_state = ManageSessionState()


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title(":tshirt: 상품 검색 서비스")

        # 필요한 state 초기화
        seg_session_state.init_session_state([("result", None),
                                              ("segmented_img", None),
                                              ("segmented_byte_img", None)])
        
        # 사진 업로드
        st.subheader("검색을 원하는 상품 사진을 업로드해주세요!")
        uploaded_img = st.file_uploader("tmp", type=["png", "jpg", "jpeg"], label_visibility='collapsed')

        # 사용자가 파일을 업로드했다면
        if uploaded_img is not None:
            img_bytes = io.BytesIO(uploaded_img.read())

            # TODO: 이미지 크기 조정을 생각해보기 (단, PIL.Image.open을 사용 시 backend로 송신할 때 에러남)
            # 프론트에 보이는 이미지 크기만 조정하는 거면, cv2가 아닌 pillow를 써서 조정해도 된다.
            # img_array = np.frombuffer(img_bytes.getvalue(), np.uint8)
            # img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            # img_cv2 = cv2.resize(img_cv2, (700, 800))
            
            # 시각화
            # TODO: 마크다운 꾸미기
            st.markdown("""
                        업로드한 이미지에서 검색하고 싶은 상품의 위치를 마우스:mouse:로 클릭해주세요!\n
                        :sparkles: **보다 정확한 검색 결과를 얻고 싶으시면 프린팅을 제외한 곳을 클릭해주세요 :-)** :sparkles:
                        """)
            # st.markdown("<div style='display: flex; justify-content: center; align-items: center;'> \
            #             <span style='color:#FF0000; text-align: center;'>무늬가 아닌, 옷 전체적인 부분을 누른다면 정확도가 더욱 높아집니다 :-)</span> \
            #             </div>", unsafe_allow_html=True)
            img = Image.open(uploaded_img)

            # 좌표 설정 (이미지가 들어오면 함수 내에서 출력하도록 만들어져 있음)
            # with st.form("form test"):
            coordinates = streamlit_image_coordinates(
                img,
                key = "lacal2")

            if coordinates is not None:
                # segmentation api 호출
                files = {"img": img_bytes}
                response = requests.post(f"{st.session_state.segmentation}/seg?x={coordinates['x']}&y={coordinates['y']}", files=files)
                
                bytes_list = response.json()
                segmented_imgs, segmented_byte_imgs = [], []
                for bytes_data in bytes_list:
                    image_data = base64.b64decode(bytes_data)
                    segmented_byte_imgs.append(image_data)

                    image = Image.open(io.BytesIO(image_data))
                    segmented_imgs.append(image)

                st.markdown("""
                            ---
                            :thinking_face: 다음 결과 중 **검색하고 싶은 상품과 가장 유사한 이미지를 선택**해주세요.
                            """)
                img_index = image_select("선택해주세요!", segmented_imgs, return_value='index')
                st.image(segmented_imgs[img_index])

                # TODO: 속도 이슈가 있음 -> st.stop() 혹은 st.experimental_rerun() 사용, 혹은 switch_page()를 이용해서 구현하는 걸로 (조금 더 생각해보기)
                if st.button("이걸로 검색하시겠어요?"):
                    # 상품 검색 페이지에서 사용할 수 있도록 선택한 이미지를 session state에 등록 
                    seg_session_state.change_session_state([('segmented_img', segmented_imgs[img_index]),
                                                            ('segmented_byte_img', segmented_byte_imgs[img_index])])

                    # 상품 검색 페이지로 이동
                    switch_page("상품 검색")
                else:
                    st.stop()

    else:
        st.header("로그인 후 이용해주세요.")

    