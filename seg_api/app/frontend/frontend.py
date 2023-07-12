import requests
import streamlit as st
import os
from PIL import Image

import streamlit as st
import requests
import time

import streamlit as st
import requests
import os
from streamlit_image_coordinates import streamlit_image_coordinates


def main():
    st.title("Segment clothes")
    st.markdown("---")
    st.header("Upload Image")
    
    if "result" not in st.session_state:
        st.session_state["result"] = None
        
    if "x" not in st.session_state:
        st.session_state["x"],st.session_state["y"] = None,None
        
    if "save_path" not in st.session_state:
        st.session_state["save_path"] = None

    uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and st.session_state["x"] is None:
        file_name = uploaded_file.name
        save_path = os.path.join("./app/input", file_name)
        # 이미지 열기
        image = Image.open(uploaded_file)
        width, height = image.size
        if width > 1000 or height > 1000: # 이미지 크기 조정
            print("resize 이전",image.size)
            image = image.resize((600, 600))
            print("resize 이후",image.size)
        image.save(save_path)
        image = Image.open(save_path)

        st.session_state["save_path"] = save_path
        st.markdown("<div style='display: flex; justify-content: center; align-items: center;'> \
        <span style='color:#FF0000; text-align: center;'>무늬가 아닌, 옷 전체적인 부분을 누른다면 정확도가 더욱 높아집니다 :-)</span> \
        </div>", unsafe_allow_html=True)

        value = streamlit_image_coordinates(
                image,

                key = "lacal2"
            )

        if value is not None:
            x = value['x']
            y = value['y']
            data = {
                "x": x,
                "y": y,
                "file_name": file_name
            }
            st.session_state["x"],st.session_state["y"] = x,y
            response = requests.post("http://localhost:8555/seg/",  json=data)
            result = response.json()
            st.session_state["result"] = result
            st.session_state["save_path"] = None
            st.success("서버 전송 완료")

    if st.session_state["result"] is not None:
        uploaded_file = None
        result = st.session_state["result"] 
        columns = st.columns(3)  # 3개의 컬럼 생성
        for i, path in enumerate(result):
            image = Image.open(path)
            with columns[i % 3]:
                file_name = os.path.basename(path)
                st.image(image, caption=f"{i+1} : {file_name}")
        st.header("Choose Inference Image")
        text_input = st.text_input("원하는 이미지를 입력하세요. [ex.1]")
        submit_pressed = False
        if st.button("Submit", key="final_bt"):
            submit_pressed = True
        if text_input or submit_pressed:
            try:
                print(f"{text_input}번째 이미지 선택함")
                if int(text_input) <= 0:
                    st.error("상품 번호를 잘못 입력하셨습니다.")
                else:
                    final_seg_path = result[int(text_input) - 1]
                    response2 = requests.post("http://localhost:8555/re/", data={"data": final_seg_path})
                    st.success("success")
                    st.write(f"확인용 : output - {final_seg_path} ")
                    
                    print(final_seg_path)
            except IndexError:
                st.error("상품 번호를 잘못 입력하셨습니다.")





if __name__ == "__main__":
    #st.set_page_config(layout="wide")
    main()
