import streamlit as st
from streamlit import success
from streamlit.source_util import page_icon_and_name

from recommend import df, recommend_songs


#Set custom streamlit page config
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎶",
    layout="centered"

)

st.title("Music Recommender System")

song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎶Select a Song:", song_list)

if st.button("Recommend Similar Songs"):
    with st.spinner("Finding Similar Songs.."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("No Similar Song Found")
        else:
            st.success("Top Similar Song:")
            st.table(recommendations)