import cv2
import streamlit as st
import pandas as pd
import random
from fer import FER

# Load songs.csv
songs_df = pd.read_csv("songs.csv")

# Initialize FER detector
detector = FER(mtcnn=True)

st.title("Real-time Emotion Detection & Music Recommendation")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect emotions
        emotions = detector.detect_emotions(rgb_frame)

        if emotions:
            dominant_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            st.subheader(f"Detected Emotion: **{dominant_emotion}**")

            # Recommend 3 random songs
            filtered_songs = songs_df[songs_df['emotion'].str.lower() == dominant_emotion.lower()]
            if not filtered_songs.empty:
                st.write("### Recommended Songs:")
                recommendations = filtered_songs.sample(min(3, len(filtered_songs)))  # pick up to 3
                for idx, row in recommendations.iterrows():
                    song_name = row['song']
                    song_link = row['link'] if 'link' in row else None

                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.write(f"- {song_name}")
                    with col2:
                        if st.button("Play", key=f"play_{idx}"):
                            if song_link:
                                st.audio(song_link)
                            else:
                                st.warning("No link available for this song.")

        FRAME_WINDOW.image(rgb_frame, channels="RGB")

cap.release()
