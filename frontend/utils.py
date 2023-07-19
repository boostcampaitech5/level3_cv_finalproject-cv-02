# streamlit
import streamlit as st

# built-in library
from typing import *


def init_session_state(states: List[Tuple[str, Any]]) -> None:
    """streamlit으로 프로그램을 실행했을 때,
    모든 페이지에서 공유할 수 있는 session state 값을 초기화하기 위해
    사용하는 함수입니다(Python의 전역변수 느낌)

    Args:
        states (List[Tuple[str, Any]]): [(str, Any)] 형태이며
        앞쪽엔 session state의 이름을, 뒤쪽엔 초기화할 값을 아무거나 입력해주세요.
    """
    for name_state, val_state in states:
        if name_state not in st.session_state:
            st.session_state[name_state] = val_state
