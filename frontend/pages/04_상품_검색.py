# streamlit
import streamlit as st
# from streamlit_extras.streamlit_image_coordinates import streamlit_image_coordinates
# from streamlit_image_select import image_select
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_image_select import image_select

# external-library
from PIL import Image

# built-in library
import requests
import os.path as osp
import io
import base64

# custom-library
from utils import init_session_state


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title("상품 검색 서비스")

        # 필요한 state 초기화
        init_session_state([("result", None)])
        
        # 사진 업로드
        uploaded_img = st.file_uploader("검색하고 싶은 상품 사진을 업로드해주세요!", 
                                        type=["png", "jpg", "jpeg"])

        # 사용자가 파일을 업로드했다면
        if uploaded_img is not None:
            img_bytes = io.BytesIO(uploaded_img.read())

            # TODO: 이미지 크기 조정을 생각해보기(단, PIL.Image.open을 사용 시 backend로 송신할 때 에러남)
            # 시각화
            st.markdown("<div style='display: flex; justify-content: center; align-items: center;'> \
                        <span style='color:#FF0000; text-align: center;'>무늬가 아닌, 옷 전체적인 부분을 누른다면 정확도가 더욱 높아집니다 :-)</span> \
                        </div>", unsafe_allow_html=True)
            img = Image.open(uploaded_img)

            # 좌표 설정
            coordinates = streamlit_image_coordinates(
                img,
                key = "lacal2"
            )

            if coordinates is not None:
                x, y = coordinates['x'], coordinates['y']
                
                # segmentation api 호출
                files = {"img": img_bytes}
                # response = requests.post(f"{st.sessesion_state.segmentation}/seg?x={x}&y={y}", files=files)
                response = requests.post(f"http://49.50.164.40:30006/seg?x={coordinates['x']}&y={coordinates['y']}", files=files)
                
                segmented_imgs = []
                bytes_list = response.json()
                for bytes_data in bytes_list:
                    image_data = base64.b64decode(bytes_data)                
                    image = Image.open(io.BytesIO(image_data))
                    if image.size[0] > image.size[1] : 
                        sz = image.size[0]
                        new_img = Image.new(image.mode, (sz, sz), (0,0,0))
                        new_img.paste(image,(0,int((image.size[0]-image.size[1])/2)))
                    else :
                        sz = image.size[1]
                        new_img = Image.new(image.mode, (sz, sz), (0,0,0))
                        new_img.paste(image, (int((image.size[1]-image.size[0])/2),0))
                    segmented_imgs.append(new_img)
                

    

                st.success('서버 전송 완료', icon="✅")
                
                img = image_select("선택해주세요", segmented_imgs)

                # TODO: 속도 이슈가 있음 -> st.stop() 혹은 st.experimental_rerun() 사용, 혹은 switch_page()를 이용해서 구현하는 걸로 (조금 더 생각해보기)
                if st.button("이걸로 검색하시겠어요?"):
                    st.write('something')
                    st.success("success")

                    # TODO: 검색 API 요청할 때 작성할 부분
                    # response2 = 

    else:
        st.header("로그인 후 이용해주세요.")
    