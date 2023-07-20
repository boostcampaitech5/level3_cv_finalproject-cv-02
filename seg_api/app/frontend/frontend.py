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
from streamlit_image_select import image_select

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
    if uploaded_file is not None :#and st.session_state["x"] is None:
        file_name = uploaded_file.name
        save_path = os.path.join("./app/input", file_name)
        # 이미지 열기
        image = Image.open(uploaded_file)
        width, height = image.size
        if width > 1000 or height > 1000: # 이미지 크기 조정
            print("resize 이전",image.size)
            image = image.resize((700, 800)) ## 상의 후 나중에 바꾸기
            print("resize 이후",image.size)
        image.save(save_path)
        
        image = Image.open(save_path)
        print(image.size, "front")
        st.session_state["save_path"] = save_path
        st.markdown("<div style='display: flex; justify-content: center; align-items: center;'> \
        <span style='color:#FF0000; text-align: center;'>무늬가 아닌, 옷 전체적인 부분을 누른다면 정확도가 더욱 높아집니다 :-)</span> \
        </div>", unsafe_allow_html=True)

        value = streamlit_image_coordinates(
                image,
                key = "lacal2"
            )

        if value is not None:
            print("aaaaaaaaaaaaaa")
            x = value['x']
            y = value['y']
            print(x,y)
            data = {
                "x": x,
                "y": y,
                "file_name": file_name
            }
            st.session_state["x"],st.session_state["y"] = x,y
            response = requests.post("http://localhost:30007/seg/",  json=data)
            result = response.json()
            print(result[0])
            st.session_state["result"] = result
            st.success("서버 전송 완료")
        
            st.header("Choose Inference Image")
            img = image_select("원하는 이미지를 선택해주세요", result)#st.session_state["result"])#[0], st.session_state["result"][1], st.session_state["result"][2]])
            print(f"{img} 선택")
            st.success("success")
            #response2 = requests.post("http://localhost:8555/re/", data={"data": img})
        





if __name__ == "__main__":
    #st.set_page_config(layout="wide")
    main()