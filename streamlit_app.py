# import os
#
# import streamlit as st
# import requests
# from io import BytesIO
# import zipfile
#
# API_BASE = "http://localhost:8000"  # replace with your Render URL
#
# st.write("Listing localhost ports...")
# st.code(os.popen("lsof -i -P -n | grep LISTEN").read())
#
#
# # ───── Health Check ─────
# try:
#     res = requests.get(f"{API_BASE}/ping", timeout=5)
#     if res.status_code == 200:
#         st.success("✅ FastAPI backend is running.")
#     else:
#         st.warning("⚠️ FastAPI backend responded, but not healthy.")
# except Exception as e:
#     st.error(f"❌ Cannot reach FastAPI backend at {API_BASE}")
#     st.stop()
#
# st.title("🔍 Face Matching App")
#
#
#
# # Session-level state management
# if "session_id" not in st.session_state:
#     st.session_state.session_id = None
# if "zip_ready" not in st.session_state:
#     st.session_state.zip_ready = False
#
# # Upload photos
# st.header("Step 1: Upload Photos (max 100)")
# uploaded_files = st.file_uploader("Upload .jpg images", type=["jpg", "jpeg"], accept_multiple_files=True)
#
#
# if uploaded_files:
#     with st.spinner("Uploading..."):
#         files = [("files", (f.name, f, "image/jpeg")) for f in uploaded_files]
#         res = requests.post(f"{API_BASE}/upload/", files=files)
#         if res.ok:
#             st.session_state.session_id = res.json()["session_id"]
#             st.success(f"✅ Uploaded {res.json()['uploaded']} images")
#         else:
#             st.error("❌ Upload failed")
#
# # Reference photo
# if st.session_state.session_id:
#     st.header("Step 2: Upload Reference")
#     ref_image = st.file_uploader("Reference photo", type=["jpg", "jpeg"], key="ref")
#     if ref_image and st.button("🔍 Match Faces"):
#         with st.spinner("Matching..."):
#             res = requests.post(
#                 f"{API_BASE}/reference/?session_id={st.session_state.session_id}",
#                 files={"reference": (ref_image.name, ref_image, "image/jpeg")}
#             )
#             if res.status_code == 200:
#                 st.success("Matched faces found!")
#                 st.download_button(
#                     "📦 Download Target Photos",
#                     res.content,
#                     "target_photos.zip",
#                     mime="application/zip"
#                 )
#             elif res.status_code == 404:
#                 st.error("❌ No faces matched.")
#             else:
#                 st.error(f"Matching failed: {res.text}")
#
#     if st.session_state.zip_ready:
#         st.download_button(
#             "📦 Download Target Photos (.zip)",
#             data=st.session_state.zip_data,
#             file_name="target_photos.zip",
#             mime="application/zip"
#         )
#
#



import os
import shutil
import zipfile
from insightface.app import FaceAnalysis
import cv2
import numpy as np

UPLOAD_DIR = "uploaded_images"
TARGET_DIR = "target_photos"
DIST_THR = 0.7

face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)

def save_uploaded_files(uploaded_files):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

def extract_face_embedding(img_bytes):
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = face_model.get(img_np)
    return faces[0].embedding if faces else None

def match_faces(reference_embedding):
    matched = []
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)

    for filename in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, filename)
        img = cv2.imread(path)
        faces = face_model.get(img)
        if not faces:
            continue
        face_embedding = faces[0].embedding
        dist = np.linalg.norm(reference_embedding - face_embedding)
        if dist < DIST_THR:
            matched.append(filename)
            shutil.copy(path, os.path.join(TARGET_DIR, filename))
    return matched


import streamlit as st

st.title("Face Matching App")

uploaded_files = st.file_uploader("Upload up to 100 images", accept_multiple_files=True, type=["jpg", "png"])

if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.success("✅ Uploaded and saved images.")

reference = st.file_uploader("Upload a reference image", type=["jpg", "png"])

if reference and st.button("Match Faces"):
    ref_embedding = extract_face_embedding(reference.read())
    if ref_embedding is None:
        st.error("❌ No face found in reference image.")
    else:
        matched_files = match_faces(ref_embedding)
        if not matched_files:
            st.error("❌ No faces matched.")
        else:
            zip_path = "matched_faces.zip"
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for f in matched_files:
                    zipf.write(os.path.join(TARGET_DIR, f), arcname=f)
            with open(zip_path, "rb") as f:
                st.download_button("Download Matched Images", f, file_name=zip_path)
