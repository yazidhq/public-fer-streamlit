import streamlit as st

st.title("Facial Emotion Recognition")

# Embed the Flask video stream in Streamlit
st.markdown("""<iframe src="http://localhost:5000" width="700" height="auto" frameborder="0" allowfullscreen></iframe>""",unsafe_allow_html=True)
