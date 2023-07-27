# streamlit
import streamlit as st
from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page

# custom-modules
from utils.management import ManageSessionState as MSS


def main():
    # 멋탠다드 로고
    st.image("./figure/meotandard_logo.png", use_column_width=True)

    st.subheader("프로젝트 개요가 궁금하지 않으시다면?")
    if st.button("바로 시작하기!", use_container_width=True):
        switch_page("상품 찾기")

    tab1, tab2, tab3 = st.tabs([":newspaper: 프로젝트 소개", ":memo: 사용 방법", ":superhero: 멋쟁이 팀원들"])

    # 서비스 소개
    with tab1:
        st.subheader("프로젝트 멋탠다드")
        
        st.markdown(
            """
            혹시 이런 경험 있으신가요?\n
            **다양한 온라인 플랫폼(인스타그램, 유튜브 등)을 이용하다가 마음에 드는 옷을 발견**했는데,\n
            **옷에 대한 정보를 찾을 수 없었던 경험** 말이죠.

            옷에 대한 정보를 찾을 수 없는 경우, 우리는 다음 행동으로 무엇을 선택할까요?
            
            ```
            1. 옷 정보를 물어본다.
            2. 직접 찾아본다.
            3. 마음에는 들지만 정보가 없으니 포기한다.
            ```
            
            옷에 대한 정보를 얻기 위해 번거로운 과정이 필요하고,\n
            그럼에도 정보를 얻을 수 있는지는 불확실하답니다.

            이번엔 옷에 대한 정보를 획득한 상황이라고 가정해봅시다.\n
            다음 행동으로 우리는 무엇을 할 수 있을까요?

            이 옷이 나에게 어울리겠다는 생각이 들면 구매를 하겠지만,\n
            이를 판단하기 어려운 사람도 존재할 수 있습니다.\n
            **온라인 환경이기에 입어볼 수가 없기 때문이죠** :cry:

            이러한 불편함을 해결해보고자 프로젝트 멋탠다드가 시작되었습니다.

            사용자가 찾고 싶은 상품을 업로드하면
            해당 상품과 동일하거나 유사한 상품들을 검색해주며,\n
            가상 피팅 서비스를 통해 온라인 환경에서도 원하는 상품을 입어볼 수 있습니다.

            저희는 멋탠다드 프로젝트가 이러한 효과를 보이길 기대합니다.

            ```
            1. 접근성 향상 : 원하는 상품에 대한 정보를 쉽게 찾을 수 있도록 도와줘요.
            2. 구매율 향상 : 의류 공급업체가 잠재적인 고객층을 확보할 수 있도록 도와줘요.
            ```
            """
            )
        

    # 사용 방법
    with tab2:
        st.subheader("상품 찾기 서비스")

        st.markdown("""
                    상품은 총 세 가지 방법으로 찾을 수 있어요!\n
                    아래의 네모 박스를 클릭하여 기능별 사용방법을 확인해주세요 :smile:
                    """)
        
        # TODO: 성한이형한테 각 기능별 설명 물어보기
        with st.expander("이미지로 상품 찾기"):
            st.subheader(":exclamation: 이럴 때 사용하세요 - 이미지로 상품 찾기")
            st.markdown("""
                        <p style=color:dodgerblue;><b>이 사람이 입은 옷이 마음에 드는데,,, 옷 정보가 없다면?</b></p>\n
                        사진을 캡처하고 이미지로 상품 찾기 기능을 활용해보세요.
                        """, unsafe_allow_html=True)

            st.markdown("---")
            
            st.markdown("""
                        1. 찾고 싶은 **상품 사진을 업로드**해주세요!\n
                        2. 업로드한 상품 사진 속에서, **찾고 싶은 상품을 마우스로 클릭**해주세요.\n
                        **상의를 찾고 싶다면 상의를 클릭, 하의를 찾고 싶다면 하의를 클릭**해주세요.\n
                        3. 보다 정확한 상품 검색을 위해, 인공지능 알파가 세 가지 후보를 추천해줘요.\n
                        그 중 **가장 깔끔한 후보를 선택**해주세요!\n
                        4. 선택한 후보를 가지고, 인공지능 베타가 상품 정보를 찾아줘요.\n
                        검색이 완료되면 깨끗한 상품 이미지와 함께 구매처도 제공한답니다.
                        """)
            
        with st.expander("텍스트로 상품 찾기"):
            st.subheader(":exclamation: 이럴 때 사용하세요 - 텍스트로 상품 찾기")
            st.markdown("""
                        <p style=color:dodgerblue;><b>찾아보고 싶은 옷은 있지만, 직접 찾아보기 귀찮을 때!</b></p>\n
                        옷을 표현하는 적절한 문장을 구성하여, 텍스트로 상품 찾기 기능을 활용해보세요.
                        """, unsafe_allow_html=True)

            st.markdown("---")
            
            st.markdown("""
                        1. **찾고 싶은 옷에 대한 정보를 텍스트로 작성**해주세요!\n
                        2. 인공지능 베타는 고객님이 올려주신 텍스트를 기반으로 비슷한 상품 정보를 찾아줘요.\n
                        검색이 완료되면 깨끗한 상품 이미지와 함께 구매처도 제공한답니다.
                        """)

        with st.expander("이미지 + 텍스트로 상품 찾기"):
            st.subheader(":exclamation: 이럴 때 사용하세요 - 이미지 + 텍스트로 상품 찾기")
            st.markdown("""
                        <p style=color:dodgerblue;><b>옷이 마음에 들기는 한데, 뭔가 2% 부족하다고 느낀다면?</b></p>\n
                        부족한 2%에 대한 정보를 텍스트로 작성해 이미지 + 텍스트로 상품 찾기 기능을 활용해보세요!
                        """, unsafe_allow_html=True)

            st.markdown("---")
            
            st.markdown("""
                        1. **찾고 싶은 상품 사진을 업로드**해주세요!\n
                        2. 업로드한 상품 사진 속에서, **찾고 싶은 상품을 마우스로 클릭**해주세요.\n
                        **상의를 찾고 싶다면 상의를 클릭, 하의를 찾고 싶다면 하의를 클릭**해주세요.\n
                        3. 보다 정확한 상품 검색을 위해, 인공지능 알파가 세 가지 후보를 추천해줘요.\n
                        그 중 **가장 깔끔한 후보를 선택**해주세요!\n
                        4. **2% 부족한 정보를 텍스트로 작성**해주세요. **색상이나 로고, 모양에 대한 정보도 좋아요!**
                        5. 선택한 후보와 입력한 텍스트를 활용해서, 인공지능 베타가 상품 정보를 찾아줘요.\n
                        검색이 완료되면 깨끗한 상품 이미지와 함께 구매처도 제공한답니다.
                        """)
        
        # TODO: 생성 서비스 이용 방법 설명
        st.subheader("가상 피팅 서비스")
        st.markdown("""
                    상품 검색 서비스는 잘 사용해 보셨나요? 그럼 이제 가상 피팅 서비스를 이용해봅시다  \n
                    아래의 네모 박스를 클릭하여 기본적인 사용 방법을 알아봅시다 :smile:
                    """)
        
        with st.expander("가상 피팅 서비스 설명서"):
            st.subheader(":exclamation: 이럴 때 사용하세요 - 가상 피팅 서비스")
            st.markdown("""
                        <p style=color:dodgerblue;><b>원하는 옷은 찾은 것 같은데, 이게 나랑 어울릴까..? 고민될 때!</b></p>\n
                        가상 피팅 서비스를 이용하여 내 사진에 입혀보세요.
                        """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("""
                        <p style=color:tomato;><b>[주의] 상품 검색 결과가 있어야만 사용할 수 있어요.</b></p>\n
                        1. 옷을 입혀볼 사람이 필요해요! **정면에서 찍은 인물 사진**을 올려주세요.\n
                        2. 인공지능 감마가 피팅을 도와드려요. **피팅에 시간이 걸리니 조금만 기다려주세요!**\n
                        3. 완성입니다! 피팅 결과가 마음에 드신다면, 별점을 남겨주세요 :star: x 5
                        """, unsafe_allow_html=True)

    # 팀원 소개 및 프로젝트 내에서 수행한 역할 기재
    with tab3:
        # TODO: 팀원 소개글 및 각자 프로젝트에서 맡은 역할 기재 (자유 양식?)
        st.subheader("함께한 멋쟁이들")
        st.markdown("🐦 **멋쟁이_김성한님**  : 전체 서비스 아키텍쳐 설계 / Retrieval BE & DB 설계 / CLIP 모델 경량화")
        st.markdown("🕊️ **멋쟁이_박수영님**  : Retrieval, Try-On FE 구현 / Try-On BE 구현")
        st.markdown("🦉 **멋쟁이_정호찬님**  : 데이터 크롤링 / Try-On 모델링")
        st.markdown("🦅 **멋쟁이_이채원님**  : Segmentation BE/FE 구현 / SAM 모델 경량화")
        st.markdown("🐥 **멋쟁이_이다현님**  : Segmentation BE/FE 구현 / Image 유사도 측정 모델 비교")

if __name__ == "__main__":
    # 에러 방지
    MSS.get_authenticator()

    # home page 동작
    main()