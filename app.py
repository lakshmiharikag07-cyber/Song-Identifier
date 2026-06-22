import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import Counter
import os



def generate_fingerprints(song_file):
    y, sr = librosa.load(song_file, sr=None)
    D = librosa.stft(y)
    S = np.abs(D)
    local_max = maximum_filter(S, size=20)
    peaks = (S == local_max)
    threshold = np.percentile(S, 99)
    peaks = peaks & (S >= threshold)
    peak_freqs, peak_times = np.where(peaks)
    peak_points = list(zip(peak_times, peak_freqs))
    fingerprints = []
    fan_out = 5
    for i in range(len(peak_points)):
        t1, f1 = peak_points[i]
        for j in range(1, fan_out + 1):
            if i + j < len(peak_points):
                t2, f2 = peak_points[i + j]
                dt = t2 - t1
                if dt > 0:
                    fingerprints.append((t1, f1, f2, dt))
    return fingerprints, peak_times, peak_freqs


@st.cache_resource
def build_database():
    fingerprint_db = {}
    songs = [f for f in os.listdir(".") if f.endswith(".mp3")]
    for song in songs:
        fps, _, _ = generate_fingerprints(song)
        for t1, f1, f2, dt in fps:
            key = (f1, f2, dt)
            if key not in fingerprint_db:
                fingerprint_db[key] = []
            fingerprint_db[key].append((song, t1))
    return fingerprint_db


fingerprint_db = build_database()


def identify_song(query_file):
    query_fingerprints, _, _ = generate_fingerprints(query_file)
    offsets = {}
    for query_t1, f1, f2, dt in query_fingerprints:
        key = (f1, f2, dt)
        if key in fingerprint_db:
            for db_song, db_t1 in fingerprint_db[key]:
                offset = db_t1 - query_t1
                if db_song not in offsets:
                    offsets[db_song] = []
                offsets[db_song].append(offset)
    song_scores = {}
    for song, offset_list in offsets.items():
        counts = Counter(offset_list)
        song_scores[song] = max(counts.values())
    if not song_scores:
        return None, None, None
    best_match = max(song_scores, key=song_scores.get)
    return best_match, song_scores[best_match], offsets



st.title("Song Identifier")
mode = st.sidebar.selectbox("Choose Mode", ["Single Clip", "Batch Mode"])



if mode == "Single Clip":
    st.write("Upload a query clip to identify it.")
    uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if uploaded_file is not None:
        with open("query_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        st.audio("query_audio.wav")

    
        y, sr = librosa.load("query_audio.wav", sr=None)
        D = librosa.stft(y)
        S = np.abs(D)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.specshow(
            librosa.amplitude_to_db(S, ref=np.max),
            sr=sr, x_axis="time", y_axis="hz", ax=ax
        )
        ax.set_title("Spectrogram")
        st.pyplot(fig)
        plt.close(fig)

        
        fps, peak_times, peak_freqs = generate_fingerprints("query_audio.wav")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.scatter(peak_times, peak_freqs, s=5)
        ax2.set_title("Constellation Map")
        ax2.set_xlabel("Time Bin")
        ax2.set_ylabel("Frequency Bin")
        st.pyplot(fig2)
        plt.close(fig2)

       
        best_match, score, offsets = identify_song("query_audio.wav")

        if best_match is None:
            st.error("No Match Found")
        else:
            st.success(f"Identified Song: {best_match}")
            st.write(f"Histogram Peak Score: {score}")

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.hist(offsets[best_match], bins=50)
            ax3.set_title("Offset Histogram")
            ax3.set_xlabel("Offset")
            ax3.set_ylabel("Count")
            st.pyplot(fig3)
            plt.close(fig3)



elif mode == "Batch Mode":
    import pandas as pd

    uploaded_files = st.file_uploader(
        "Upload Multiple Audio Files",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        for file in uploaded_files:
            temp_name = file.name
            with open(temp_name, "wb") as f:
                f.write(file.read())

            best_match, score, _ = identify_song(temp_name)

            if best_match is None:
                prediction = "No Match"
            else:
                prediction = os.path.splitext(best_match)[0]

            results.append([file.name, prediction])

        df = pd.DataFrame(results, columns=["filename", "prediction"])
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
