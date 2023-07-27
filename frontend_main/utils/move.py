# streamlit
import streamlit as st


def prev(): st.session_state.counter -= 1
def next(): st.session_state.counter += 1