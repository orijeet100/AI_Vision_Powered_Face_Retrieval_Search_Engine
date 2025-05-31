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





# streamlit_app.py

import os
import cv2
import faiss
import zipfile
import shutil
import pickle
import streamlit as st
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from insightface.app import FaceAnalysis
import logging

# ───── Setup ─────────────────────────────────────────────────────────────
DATABASE_DIR = Path("database")
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "user_photos"
TARGET_DIR = Path("target_photos")
INDEX_FILE = DATA_DIR / "faces.faiss"
META_FILE = DATA_DIR / "faces.pkl"
DIST_THR = 0.7

# ───── Ensure dirs ───────────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)
TARGET_DIR.mkdir(exist_ok=True)

# ───── Model Load ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face model (~600MB)...")
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = load_model()

import shutil

def check_disk_space():
    """Check available disk space"""
    total, used, free = shutil.disk_usage("/")
    st.sidebar.write(f"💾 Disk Space: {free // (1024**3)} GB free")
    if free < 1024**3:  # Less than 1GB
        st.warning("⚠️ Low disk space available")

# Add this call in your main UI
check_disk_space()

# ───── Index helpers ─────────────────────────────────────────────────────
def _load_index():
    return faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else faiss.IndexFlatIP(512)

def _load_meta():
    return pickle.load(open(META_FILE, "rb")) if META_FILE.exists() else []

def _save_index(index, meta):
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

# ───── Add user images ───────────────────────────────────────────────────
def add_images(files):
    index = _load_index()
    meta = _load_meta()
    new_vecs, new_paths = [], []

    for file in files:
        file_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        faces = face_app.get(img)
        if not faces:
            continue

        dst_path = IMG_DIR / file.name
        cv2.imwrite(str(dst_path), img)

        for face in faces:
            new_vecs.append(face["embedding"])
            new_paths.append(str(dst_path))

    if new_vecs:
        vecs = np.vstack(new_vecs).astype("float32")
        faiss.normalize_L2(vecs)
        index.add(vecs)
        meta.extend(new_paths)
        _save_index(index, meta)
        return len(new_vecs)
    return 0

# ───── Search ────────────────────────────────────────────────────────────
def match_faces(reference_file) -> Tuple[BytesIO, list]:
    shutil.rmtree(TARGET_DIR, ignore_errors=True)
    TARGET_DIR.mkdir(exist_ok=True)

    index = _load_index()
    meta = _load_meta()

    # Reset file pointer to beginning
    reference_file.seek(0)
    file_bytes = reference_file.read()

    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = face_app.get(img)
    if not faces:
        return None, "No face detected."

    q = faces[0]["embedding"].astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, min(50, index.ntotal))

    results = []
    matched_files = []  # Store actual file data instead of paths

    for d, i in zip(D[0], I[0]):
        similarity = d
        distance = 1 - similarity / 2
        if distance < DIST_THR:
            source_path = meta[i]
            img_match = cv2.imread(source_path)
            if img_match is not None:
                fname = Path(source_path).name
                target_path = TARGET_DIR / fname
                cv2.imwrite(str(target_path), img_match)

                # Read the file data immediately
                with open(target_path, 'rb') as f:
                    file_data = f.read()
                matched_files.append((fname, file_data))
                results.append((str(target_path), similarity))

    if not results:
        return None, "No faces matched."

    # Create ZIP with actual file data
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, file_data in matched_files:
            zipf.writestr(fname, file_data)

    # Important: Reset buffer position
    zip_buffer.seek(0)

    logging.basicConfig(level=logging.INFO)

    # In match_faces function:
    logging.info(f"Found {len(results)} matches")
    logging.info(f"ZIP buffer size: {len(zip_buffer.getvalue())} bytes")

    return zip_buffer, results


# ───── UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Retrieval App", layout="centered")
st.title("🔍 Face Retrieval System")

# Upload images
st.header("Step 1: Upload Images to Index")
img_files = st.file_uploader("Upload JPG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if img_files and st.button("📁 Add to Face Index"):
    with st.spinner("Indexing..."):
        count = add_images(img_files)
        if count > 0:
            st.success(f"✅ {count} faces indexed!")
        else:
            st.warning("⚠️ No faces found.")

# Upload reference
st.header("Step 2: Upload Reference Image")
ref_file = st.file_uploader("Reference photo", type=["jpg", "jpeg"])

# After face matching - FIXED VERSION
if ref_file and st.button("🔍 Match Faces"):
    with st.spinner("Matching..."):
        zip_buffer, result = match_faces(ref_file)

    if zip_buffer is None:
        st.error(f"❌ {result}")
    else:
        st.success(f"✅ Found {len(result)} matched faces!")

        # Get ZIP data properly
        zip_data = zip_buffer.getvalue()

        # Verify ZIP data is valid
        if len(zip_data) > 0:
            st.download_button(
                label="📦 Download Matched Images (.zip)",
                data=zip_data,
                file_name="matched_faces.zip",
                mime="application/zip",
                key="download_zip"
            )

            # Show file size for debugging
            st.info(f"ZIP file size: {len(zip_data)} bytes")

            # Show preview
            st.subheader("Matched Faces Preview")
            for path, score in result[:5]:  # Show only first 5 for performance
                if Path(path).exists():
                    st.image(path, caption=f"{Path(path).name} (score: {score:.3f})", width=200)
        else:
            st.error("❌ Generated ZIP file is empty")