import requests
import streamlit as st

def send_csv_file(file_path):
    url = 'http://118.67.135.82:30007/proxy/products-upload'  
    files = {'file': open(file_path, 'rb')}  

    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            st.success('파일 업로드 성공!')
            st.write(response.json())
        else:
            st.error('파일 업로드 실패')
    except requests.exceptions.RequestException as e:
        st.error(f'에러 코드: {e}')

if __name__ == "__main__":
    st.title(":tshirt: 상품 등록 서비스")
    st.markdown('---')
            
    # 파일 업로드
    st.subheader("상품 정보 csv 파일 업로드")
    uploaded_csv = st.file_uploader("tmp", type=["csv"], label_visibility='collapsed')

    if uploaded_csv is not None:
        with open("tmp.csv", "wb") as f:
            f.write(uploaded_csv.getvalue())
        send_csv_file("tmp.csv")
        
    st.write("완료")
