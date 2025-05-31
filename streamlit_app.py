import os

import streamlit as st
import requests
from io import BytesIO
import zipfile

API_BASE = "http://localhost:8000"  # replace with your Render URL

st.write("Listing localhost ports...")
st.code(os.popen("lsof -i -P -n | grep LISTEN").read())


# ───── Health Check ─────
try:
    res = requests.get(f"{API_BASE}/ping", timeout=5)
    if res.status_code == 200:
        st.success("✅ FastAPI backend is running.")
    else:
        st.warning("⚠️ FastAPI backend responded, but not healthy.")
except Exception as e:
    st.error(f"❌ Cannot reach FastAPI backend at {API_BASE}")
    st.stop()

st.title("🔍 Face Matching App")



# Session-level state management
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False

# Upload photos
st.header("Step 1: Upload Photos (max 100)")
uploaded_files = st.file_uploader("Upload .jpg images", type=["jpg", "jpeg"], accept_multiple_files=True)


if uploaded_files:
    with st.spinner("Uploading..."):
        files = [("files", (f.name, f, "image/jpeg")) for f in uploaded_files]
        res = requests.post(f"{API_BASE}/upload/", files=files)
        if res.ok:
            st.session_state.session_id = res.json()["session_id"]
            st.success(f"✅ Uploaded {res.json()['uploaded']} images")
        else:
            st.error("❌ Upload failed")

# Reference photo
if st.session_state.session_id:
    st.header("Step 2: Upload Reference")
    ref_image = st.file_uploader("Reference photo", type=["jpg", "jpeg"], key="ref")
    if ref_image and st.button("🔍 Match Faces"):
        with st.spinner("Matching..."):
            res = requests.post(
                f"{API_BASE}/reference/?session_id={st.session_state.session_id}",
                files={"reference": (ref_image.name, ref_image, "image/jpeg")}
            )
            if res.status_code == 200:
                st.success("Matched faces found!")
                st.download_button(
                    "📦 Download Target Photos",
                    res.content,
                    "target_photos.zip",
                    mime="application/zip"
                )
            elif res.status_code == 404:
                st.error("❌ No faces matched.")
            else:
                st.error(f"Matching failed: {res.text}")

    if st.session_state.zip_ready:
        st.download_button(
            "📦 Download Target Photos (.zip)",
            data=st.session_state.zip_data,
            file_name="target_photos.zip",
            mime="application/zip"
        )


