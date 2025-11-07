import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from fer import FER

st.set_page_config(page_title="Emotion-based Music Recommender", page_icon="🎵")

# --- Load songs CSV ---
@st.cache_data
def load_songs():
    df = pd.read_csv("songs.csv")

    # Clean whitespace and normalize text
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["emotion", "language", "title", "artist", "link"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Clean and normalize links
    df['link'] = df['link'].apply(lambda x: x if x.startswith("http") else "https://" + x)

    return df


df = load_songs()


# --- Map FER labels to your categories ---
def map_emotion(fer_emotion):
    e = fer_emotion.lower()
    if e in ["happy", "happiness"]:
        return "Happy"
    if e in ["sad", "sadness", "fear", "fearful"]:
        return "Sad"
    if e in ["angry", "anger", "disgust"]:
        return "Angry"
    return "Relaxed"

# --- Detect emotion from image ---
def detect_emotion(img):
    arr = np.array(img.convert("RGB"))
    detector = FER(mtcnn=False)
    results = detector.detect_emotions(arr)
    if not results:
        return None
    emotions = results[0]["emotions"]
    fer_label = max(emotions, key=emotions.get)
    confidence = emotions[fer_label]
    return map_emotion(fer_label), confidence


# --- Recommend songs based on emotion + language ---
def recommend(df, emotion, language, k=3):
    # normalize comparison
    subset = df[
        (df["emotion"].astype(str).str.strip().str.lower() == emotion.lower().strip())
        & (df["language"].astype(str).str.strip().str.lower() == language.lower().strip())
    ]

    if subset.empty:
        st.warning(f"No {language} songs found for emotion '{emotion}'. Showing random {language} songs.")
        subset = df[df["language"].astype(str).str.strip().str.lower() == language.lower().strip()]

    return subset.sample(min(k, len(subset)))


# --- App title ---
st.title("Emotion-based Music Recommender 🎶")
st.write("Detect your emotion and get songs that match your mood.")

# 🔹 NEW: Language selector (in sidebar)
st.sidebar.header("Preferences")
selected_language = st.sidebar.selectbox("Choose song language:", ["English", "Hindi", "Tamil"])

# --- Input method ---
mode = st.radio("Choose input method:", ["Upload photo", "Webcam", "Pick emotion manually"])
emotion = None

# --- Upload photo ---
if mode == "Upload photo":
    uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=200)

        # Analyze with spinner
        with st.spinner("Analyzing your emotion..."):
            emotion_result = detect_emotion(img)

        if emotion_result:
            emotion, conf = emotion_result
            st.success(f"Detected Emotion: **{emotion}** ({conf*100:.1f}% confidence)")
        else:
            st.warning("No face detected 😕 — please try another image or use manual mode.")




# --- Webcam (use Streamlit camera_input) ---
if mode == "Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", width=200)
        emotion_result = detect_emotion(img)
        if emotion_result:
           emotion, conf = emotion_result
           st.success(f"Detected Emotion: **{emotion}** ({conf*100:.1f}% confidence)")
        else:
           st.warning("No face detected 😕 — please try another image or use manual mode.")


# --- Manual selection ---
if mode == "Pick emotion manually":
    emotion = st.selectbox("Select your emotion:", ["Happy", "Sad", "Angry", "Relaxed"])

# --- Recommend and play songs ---
if emotion:
    st.subheader(f"Recommended {selected_language} Songs 🎶")

    if (
        "recs" not in st.session_state
        or st.session_state.get("last_emotion") != emotion
        or st.session_state.get("last_language") != selected_language
    ):
        st.session_state["recs"] = recommend(df, emotion, selected_language)
        st.session_state["last_emotion"] = emotion
        st.session_state["last_language"] = selected_language

    recs = st.session_state["recs"]

    for i, row in recs.iterrows():
        st.markdown(f"- **{row['title']}** — {row['artist']}")
        button_key = f"play_{i}"

        if st.button("Play", key=button_key):
            st.session_state["song_to_play"] = row['link']

    if "song_to_play" in st.session_state:
        st.video(st.session_state["song_to_play"])

