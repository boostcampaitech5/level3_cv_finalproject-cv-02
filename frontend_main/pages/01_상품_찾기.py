# streamlit
import streamlit as st
from streamlit_image_select import image_select
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_extras.switch_page_button import switch_page

# external-library
from PIL import Image

# built-in library
import io

# custom-modules
from utils.move import next, prev
from utils.management import ManageSessionState as MSS
from meotandard_apis import MeotandardSeg, MeotandardRetrieval
from utils.open import open_page

# seg 관련 session state 초기화
MSS.init_session_state([("segmented_img", None),
                        ("segmented_byte_img", None),
                        ("seg_select_state", False)])

# 검색 관련 session state 초기화
MSS.init_session_state([('selected', None), ('counter', 0),
                        ('text_before', ''), ('tryon_cloth_data', None),
                        ('choose_idx', None), ('clothes_metadata', None)])

# Seg, Retrieval API 클래스 정의
meotandard_seg = MeotandardSeg()
meotandard_retrieval = MeotandardRetrieval()

# css
bg_container = """
<style>
[data-testid="stVerticalBlock"]
[class="css-1n76uvr e1f1d6gn0"]{
background-color: whitesmoke;
}
</style>
"""


def seg():
    uploaded_img = st.file_uploader("tmp", type=["png", "jpg", "jpeg"], label_visibility='collapsed')

    if uploaded_img is not None:
        st.markdown("""
                    업로드한 이미지에서 검색하고 싶은 상품의 위치를 마우스:mouse:로 클릭해주세요!\n
                    :sparkles: **보다 정확한 검색 결과를 얻고 싶으시면 프린팅을 제외한 곳을 클릭해주세요 :-)** :sparkles:
                    """)

        # 시각화
        img = Image.open(uploaded_img)        
        img_w, img_h = img.size

        if img_w > 1000 or img_h > 1000:
            img = img.resize((700, 1000))

        coordinates = streamlit_image_coordinates(
            img,
            key='lacal2'
        )

        if coordinates is not None:
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

            # 선택을 확정지으면
            if st.button("이걸로 검색하시겠어요?", use_container_width=True):
                # 상품 검색 페이지에서 사용할 수 있도록 선택한 이미지를 session state에 등록 
                # 검색할 때는 딱 맞게 잘려진 이미지를 사용함
                MSS.change_session_state([('segmented_img', segmented_imgs[img_index]),
                                        ('segmented_byte_img', segmented_byte_imgs[img_index]), 
                                        ('seg_select_state', True)])
                st.experimental_rerun()               


def retrieval(mode: str = 'image'):
    if mode == 'image':
        with st.columns(3)[1]:
            st.image(st.session_state.segmented_img, use_column_width=True)

        # threshold는 default == 0.0
        clothes_metadata = meotandard_retrieval.querying_searchbyimage_api(st.session_state.segmented_byte_img)
        MSS.change_session_state([('clothes_metadata', clothes_metadata)])
    
    elif mode == 'text':
        text = st.text_input("텍스트를 입력해주세요.")

        if text:
            # 새로운 text가 입력되었다면
            if text != st.session_state.text_before:
                st.session_state.text_before = text
                MSS.change_session_state([('counter', 0)])

            translation_result, clothes_metadata = meotandard_retrieval.querying_searchbytext_api(text)
            MSS.change_session_state([('clothes_metadata', clothes_metadata)])

            if translation_result:
                st.write(f"번역 결과입니다: {translation_result}")

    elif mode == 'image_and_text':
        # 선택한 상품 이미지를 보여주기
        with st.columns(3)[1]:
            st.image(st.session_state.segmented_img, use_column_width=True)

        text = st.text_input("텍스트를 입력해주세요.")

        if text:
            translation_result, clothes_metadata = meotandard_retrieval.querying_searchbyfilter_api(st.session_state.segmented_byte_img, text)
            
            st.write(f"번역 결과입니다: {translation_result}")
            MSS.change_session_state([('clothes_metadata', clothes_metadata)])


    if st.session_state.clothes_metadata:
        st.header("상품 검색 결과입니다! :airplane_departure:")
        st.markdown('---')
        
        n_clothes = len(st.session_state.clothes_metadata)
        
        for i in range(2):
            cols = st.columns(5)
            st.session_state.counter %= (n_clothes // 10)

            for j in range(5):
                if st.session_state.counter < 0:
                    idx = 10 * (5 + st.session_state.counter) + (i * 5) + j
                else:
                    idx = (10 * st.session_state.counter) + (i * 5) + j

                meta_data = st.session_state.clothes_metadata[idx]
                
                image_url = meta_data['image_link']
                link_url = meta_data['link']

                # with cols[j]: st.image(image_url, use_column_width=True)
                cols[j].write(f'<img src="{image_url}" alt="Image" width="126" height="168">', unsafe_allow_html=True)

                if cols[j].button(f"{idx + 1}번 상품", key=idx, use_container_width=True):
                    st.session_state['choose_idx'] = idx

        # 이미지를 선택했다면
        if  st.session_state['choose_idx'] is not None:
            choose_data = st.session_state.clothes_metadata[st.session_state['choose_idx']]

            container = st.container()
            st.markdown(bg_container, unsafe_allow_html=True)
            with container.container():
                col1, col2 = st.columns(2)
                with col1: st.markdown(f"**{st.session_state['choose_idx'] + 1}번 상품명**  \n{choose_data['name']}")
                with col2: st.markdown(f"**가격**  \n{choose_data['price']}")
                
                if choose_data['meta'] == "0":
                    st.write("")
                else:
                    # "[]", ",", "''" 제거
                    st.write(choose_data["meta"][1:-1].replace("'", "").replace(",", ""))
                
            col1, col2 = st.columns(2)
            col1.button("상품 사이트", on_click=open_page, args=(choose_data['link'],), use_container_width=True)
            if col2.button("입어보기!", use_container_width=True):
                if choose_data['check'] == 0:
                    st.warning("상품 사진에 사람이 있거나 여러 장의 옷이 있는 경우 입어볼 수 없어요 :cry:")
                else:                
                    MSS.change_session_state([('tryon_cloth_data', choose_data)])
                    switch_page("가상 피팅")

            col1.button("⬅️ 이전", on_click=prev, use_container_width=True)
            col2.button("다음 ➡️", on_click=next, use_container_width=True)

    # 다시 seg로 돌아가기
    if mode != "text":
        if st.button("다른 사진 업로드하기", use_container_width=True):
            # 상품 upload 페이지로 이동
            MSS.change_session_state([('seg_select_state', False)])
            st.experimental_rerun()    


def image_retrieval():
    if st.session_state.seg_select_state == False:
        seg()
    else:
        retrieval(mode='image')


def image_and_text_retrieval():
    if st.session_state.seg_select_state == False:
        seg()
    else:
        retrieval(mode='image_and_text')


def reset_session_state():
    # 필요한 session state 초기화
    MSS.change_session_state([("segmented_img", None),
                            ("segmented_byte_img", None),
                            ("seg_select_state", False)])

    # 필요한 session state 초기화
    MSS.change_session_state([('selected', None),
                            ('clothes_metadata', None)])

    # 필요한 session state 초기화
    MSS.change_session_state([('counter', 0),
                            ('searched_cloth', list()),
                            ('tryon_cloth', None),
                            ('choose_idx', 0)])


if __name__ == "__main__":
    st.title("상품 찾기")
    options = ["이미지로 상품 찾기", "텍스트로 상품 찾기", "이미지 + 텍스트로 상품 찾기"]

    selected = st.selectbox("기능을 선택해주세요", options)

    if selected != st.session_state.selected:
        reset_session_state()

    st.session_state.selected = selected

    if selected == options[0]:
        image_retrieval()
        
    elif selected == options[1]:
        retrieval(mode='text')
    
    elif selected == options[2]:
        image_and_text_retrieval()
