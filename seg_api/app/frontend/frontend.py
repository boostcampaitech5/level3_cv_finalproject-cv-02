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

def main():
    st.title("Segment clothes")
    st.markdown("---")
    st.header("Upload Image")
    seg_state = False
    if "result" not in st.session_state:
        st.session_state["result"] = None
        seg_state = True
    uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None and seg_state is None:
        print("출력xxxxx")
        st.write("업로드된 파일:", uploaded_file.name)
        file_name = uploaded_file.name
        if st.button('Seg start!', key="start_bt"):
            save_path = os.path.join("./app/input", file_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            response = requests.post("http://localhost:8555/seg/", data={"data": file_name})
            result = response.json()
            st.session_state["result"] = result
            st.success("서버 전송 완료")

    if st.session_state["result"] is not None:
        result = st.session_state["result"] 
        columns = st.columns(3)  # 3개의 컬럼 생성
        for i, path in enumerate(result):
            image = Image.open(path)
            with columns[i % 3]:
                file_name = os.path.basename(path)
                st.image(image, caption=f"{i+1} : {file_name}")
        st.header("Choose Inference Image")
        text_input = st.text_input("텍스트를 입력하세요. [ex.1]")
        submit_pressed = False
        if st.button("Submit", key="final_bt"):
            submit_pressed = True
        if text_input and submit_pressed :
            print(f"{text_input}번째 이미지 선택함")
            final_seg_path =result[int(text_input)-1]
            response2 = requests.post("http://localhost:8555/re/", data={"data": final_seg_path})
            st.write(f"output : {final_seg_path} ")
            print(final_seg_path)





if __name__ == "__main__":
    
    main()
