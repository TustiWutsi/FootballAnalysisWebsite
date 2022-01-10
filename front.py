import streamlit as st

def header(text):
    st.markdown(f'<p style="background-color:#5D6D7E;color:#FDFEFE;">{text}</p>', unsafe_allow_html=True)