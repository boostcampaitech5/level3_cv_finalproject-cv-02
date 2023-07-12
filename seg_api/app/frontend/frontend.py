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
    start_time = time.time()
    print("시작1",time.time() )
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
        print("업로드 완료2", time.time())
        #print("출력xxxx, 업로드:", uploaded_file,"세그 상태 ")
        file_name = uploaded_file.name
        save_path = os.path.join("./app/input", file_name)
        st.session_state["save_path"] = save_path
        st.markdown("<div style='display: flex; justify-content: center; align-items: center;'> \
        <span style='color:#FF0000; text-align: center;'>무늬가 아닌, 옷 전체적인 부분을 누른다면 정확도가 더욱 높아집니다 :-)</span> \
        </div>", unsafe_allow_html=True)
        with open(st.session_state["save_path"], "wb") as f:
            f.write(uploaded_file.getbuffer())
        print("이미지 오픈3",time.time())
        value = streamlit_image_coordinates(
                save_path,

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
            print("좌표찍기4, 서버한테 보내기 전",time.time())
            response = requests.post("http://localhost:8555/seg/",  json=data)
            result = response.json()
            st.session_state["result"] = result
            st.session_state["save_path"] = None
            st.success("서버 전송 완료")
            print("서버한테 받음5",time.time())

    if st.session_state["result"] is not None:
        uploaded_file = None
        result = st.session_state["result"] 
        columns = st.columns(3)  # 3개의 컬럼 생성
        print("이제 이미지 불러오기 전6",time.time())
        for i, path in enumerate(result):
            image = Image.open(path)
            with columns[i % 3]:
                file_name = os.path.basename(path)
                st.image(image, caption=f"{i+1} : {file_name}")
                
        print("이미지 3장 다 불러옴7",time.time())
        st.header("Choose Inference Image")
        text_input = st.text_input("텍스트를 입력하세요. [ex.1]")
        submit_pressed = False
        if st.button("Submit", key="final_bt"):
            submit_pressed = True
            print("true")
        if text_input or submit_pressed:
            try:
                print(f"{text_input}번째 이미지 선택함")
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