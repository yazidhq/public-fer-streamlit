import streamlit as st

st.title("Real-Time Facial Emotion Recognition")

# Display video feed from the Flask server
st.markdown(
    """
    <div style="text-align: center;">
        <img src="http://127.0.0.1:5000/video_feed" style="width: 80%; border: 1px solid black;">
    </div>
    """,
    unsafe_allow_html=True
)
