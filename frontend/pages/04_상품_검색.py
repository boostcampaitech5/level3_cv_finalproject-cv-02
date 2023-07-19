# streamlit
import streamlit as st

# built-in library

# custom-library
from utils import init_session_state


if __name__ == "__main__":
    if st.session_state.authentication_status == True:
        st.title("상품 검색 서비스")

        # 필요한 state 초기화
        init_session_state([("result", None), 
                            ("x", None), 
                            ("y", None), 
                            ("save_path", None)])
        

        upload_file = st.file_uploader("검색을 원하는 상품 사진을 업로드해주세요!", type=["png", "jpg", "jpeg"])

        # 사용자가 파일을 업로드했다면
        if upload_file is not None:
            file_name = upload_file.name

            # TODO: seg 쪽 api가 upload file로 바뀐다고 하지 않았었나..? 체크해보기
    else:
        st.header("로그인 후 이용해주세요.")
    