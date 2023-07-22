# 상품 검색 탭에서 검색 결과가 나왔을 때, 하나를 선택하여 입어보기를 누르면 생성 페이지로 이동
# 생성 페이지에서는 선택한 상품 이미지가 보여야 하며, 사용자 사진을 업로드해야 함.
# 상품 이미지와 사용자 사진이 업로드되면 생성 시작.

# streamlit
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# custom-modules
from utils.management import ManageSessionState as MSS

# api 주소 및 accounts 정보를 session state에 등록 (Home에서 한 번만)
base_session_state = MSS(api_address_path="./api_address.yaml",
                         accounts_path="./accounts.yaml")
MSS.init_session_state([("seg_select_state", False)])


if __name__ == "__main__":
    # 프로젝트 제목
    st.title(":shirt: 쇼핑의 기본, \"멋탠다드\"")
    st.markdown("---")

    st.subheader("프로젝트 개요가 궁금하지 않으시다면?")
    if st.button("바로 시작하기!"):
        switch_page("로그인")

    tab1, tab2, tab3 = st.tabs([":newspaper: 프로젝트 소개", ":memo: 사용 방법", ":superhero: 멋쟁이 팀원들"])

    # 서비스 소개
    with tab1:
        # TODO: 적절한 이미지와 글씨 색깔, bold체 등을 활용하여 핵심 정보가 눈에 들어오도록 수정하기
        st.subheader("프로젝트 멋탠다드")
        
        st.markdown(
            """
            혹시 이런 경험 있으신가요?\n
            다양한 온라인 플랫폼(인스타그램, 유튜브 등)을 이용하다가 마음에 드는 옷을 발견했는데,\n
            옷에 대한 정보를 찾을 수 없었던 경험 말이죠.

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
            온라인 환경이기에 입어볼 수가 없기 때문이죠 :cry:

            이러한 불편함을 해결해보고자 프로젝트 멋탠다드가 시작되었습니다.

            사용자가 찾고 싶은 상품을 업로드하면
            해당 상품과 동일하거나 유사한 상품들을 검색해주며,\n
            가상 피팅 서비스를 통해 온라인 환경에서도 원하는 상품을 입어볼 수 있습니다.

            저희는 멋탠다드 프로젝트가 이러한 효과를 보이길 기대합니다.

            1. 접근성 향상 : 원하는 상품에 대한 정보를 쉽게 찾을 수 있도록 도와줘요.
            2. 구매율 향상 : 의류 공급업체가 잠재적인 고객층을 확보할 수 있도록 도와줘요.
            """
            )
        

    # 사용 방법
    with tab2:
        # TODO: 상품 검색 서비스 이용 방법 설명
        # TODO: 업로드 하는 방법을 이미지 혹은 영상으로 첨부하기.
        st.subheader("상품 검색 서비스")
        st.markdown("1. 찾고 싶은 상품 사진을 업로드해주세요!")

        st.markdown("2. 업로드한 상품 사진 속에서, 찾고 싶은 상품을 마우스로 클릭해주세요.  \n\
                    상의를 찾고 싶다면 상의를 클릭, 하의를 찾고 싶다면 하의를 클릭해주세요.")

        st.markdown("3. 보다 정확한 상품 검색을 위해, 인공지능 알파가 세 가지 후보를 추천해줘요.  \n\
                    그 중 가장 깔끔한 후보를 선택해주세요!")

        st.markdown("4. 선택한 후보를 가지고, 인공지능 베타가 상품 정보를 찾아줘요.  \n\
                    검색이 완료되면 깨끗한 상품 이미지와 함께 구매처도 제공한답니다.")
        
        # TODO: 생성 서비스 이용 방법 설명
        st.subheader("가상 피팅 서비스")
        st.markdown("상품 검색 서비스는 잘 사용해 보셨나요? 그럼 이제 가상 피팅 서비스를 이용해봅시다 :smile:")
        st.markdown("1. 옷을 입혀볼 사람이 필요해요! 정면에서 찍은 전신 사진을 올려주세요.")

        st.markdown("2. 입어보고 싶은 옷을 골라주세요!")

        st.markdown("3. 입어보기! 버튼을 클릭해주시면, 인공지능 감마가 가상 피팅을 도와드려요.")

        st.markdown("4. 완성입니다! 피팅 결과가 마음에 드신다면, 별점과 함께 사진을 자랑해주세요.")

    # 팀원 소개 및 프로젝트 내에서 수행한 역할 기재
    with tab3:
        # TODO: 팀원 소개글 및 각자 프로젝트에서 맡은 역할 기재 (자유 양식?)
        st.subheader("함께한 팀원들")
        st.markdown("김성한 님")
        st.markdown("박수영 님")
        st.markdown("정호찬 님")
        st.markdown("이채원 님")
        st.markdown("이다현 님")

