import streamlit as st
import requests
from io import BytesIO
import zipfile

API_BASE = "https://face-retrieval.onrender.com"  # replace with your Render URL

st.title("🔍 Face Matching App")

# ─────────────── Upload photo batch ───────────────
st.header("Step 1: Upload up to 100 photos")

uploaded_files = st.file_uploader(
    "Upload multiple .jpg images",
    type=["jpg", "jpeg"],
    accept_multiple_files=True,
    key="photo_batch"
)

if uploaded_files:
    with st.spinner("Uploading..."):
        files = [("files", (f.name, f, "image/jpeg")) for f in uploaded_files]
        response = requests.post(f"{API_BASE}/upload/", files=files)
        if response.status_code == 200:
            session_id = response.json()["session_id"]
            st.success(f"✅ Uploaded {response.json()['uploaded']} files")
        else:
            st.error(f"Upload failed: {response.text}")
            session_id = None
else:
    session_id = None

# ─────────────── Reference photo ───────────────
if session_id:
    st.header("Step 2: Upload reference image")

    ref_image = st.file_uploader(
        "Upload a reference face image",
        type=["jpg", "jpeg"],
        key="ref_photo"
    )

    zip_file_bytes = None

    if ref_image:
        if st.button("🔍 Match Faces"):
            with st.spinner("Processing and matching..."):
                files = {"reference": (ref_image.name, ref_image, "image/jpeg")}
                response = requests.post(
                    f"{API_BASE}/reference/?session_id={session_id}",
                    files=files
                )
                if response.status_code == 200:
                    zip_file_bytes = BytesIO(response.content)
                    st.success("🎯 Match complete!")
                    st.download_button(
                        label="📦 Download Target Photos (.zip)",
                        data=zip_file_bytes,
                        file_name="target_photos.zip",
                        mime="application/zip"
                    )
                else:
                    st.error(f"Matching failed: {response.text}")
