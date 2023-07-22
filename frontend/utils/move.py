# streamlit
import streamlit as st
from streamlit.components.v1 import html


def prev(): st.session_state.counter -= 1
def next(): st.session_state.counter += 1


# https://discuss.streamlit.io/t/streamlit-button-with-link/41171
def open_page(url):
    open_script = """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    html(open_script)
