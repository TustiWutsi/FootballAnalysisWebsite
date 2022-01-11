import streamlit as st

def header(text):
    st.markdown(f'<p style="background-color:#5D6D7E;color:#FDFEFE;">{text}</p>', unsafe_allow_html=True)

def in_progress():
    st.markdown(f'<p style="background-color:#FAE5D3;color:#E67E22;"><center>Construction in progress</center></p>', unsafe_allow_html=True)