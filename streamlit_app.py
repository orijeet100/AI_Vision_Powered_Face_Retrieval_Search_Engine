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



# streamlit_app.py

import os
import cv2
import faiss
import pickle
import shutil
import zipfile
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List
from insightface.app import FaceAnalysis

# ─── Config ─────────────────────────────────────────────────────────────
DATABASE_DIR = Path("database")          # user-uploaded folder of jpg images
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "user_photos"       # internal vault
INDEX_FILE = DATA_DIR / "faces.faiss"
META_FILE = DATA_DIR / "faces.pkl"
TARGET_DIR = Path("target_photos")       # result folder
DIST_THR = 0.7                           # cosine distance threshold

# ─── Init Directories ────────────────────────────────────────────────────
for path in [DATA_DIR, IMG_DIR, TARGET_DIR]:
    path.mkdir(exist_ok=True)

# ─── Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

face_app = load_model()

# ─── Helper Functions ────────────────────────────────────────────────────
def _load_or_create_index(dim: int = 512):
    if INDEX_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    return faiss.IndexFlatIP(dim)

def _load_meta() -> List[str]:
    if META_FILE.exists():
        return pickle.load(open(META_FILE, "rb"))
    return []

def _save_index(index, meta):
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

def add_images(files: List):
    index = _load_or_create_index()
    meta = _load_meta()

    new_vecs, new_paths = [], []
    for uploaded_file in files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            continue
        faces = face_app.get(img)
        if not faces:
            continue

        dst_path = IMG_DIR / uploaded_file.name
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

def search(reference_img_bytes, k: int = 50):
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise RuntimeError("Index not found. Add images first.")

    index = _load_or_create_index()
    meta = _load_meta()

    TARGET_DIR.mkdir(exist_ok=True)
    for f in TARGET_DIR.glob("*"):
        f.unlink()

    img = cv2.imdecode(np.frombuffer(reference_img_bytes, np.uint8), 1)
    if img is None:
        raise FileNotFoundError("Invalid image uploaded.")

    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face detected in reference image.")

    q = faces[0]["embedding"].astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, min(k, index.ntotal))

    results = []
    for d, i in zip(D[0], I[0]):
        similarity = d
        distance = 1 - similarity / 2
        if distance < DIST_THR:
            src_path = meta[i]
            tgt_path = TARGET_DIR / Path(src_path).name
            img_match = cv2.imread(src_path)
            if img_match is not None:
                cv2.imwrite(str(tgt_path), img_match)
                results.append((src_path, similarity))

    return sorted(results, key=lambda x: -x[1])

def zip_target_photos():
    zip_path = Path("matched_photos.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in TARGET_DIR.glob("*.jpg"):
            zipf.write(file, arcname=file.name)
    return zip_path

# ─── Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Face Search", layout="centered")
st.title("🧠 Face Match App")

# Step 1: Upload face database
st.subheader("Step 1: Upload Photos to Build Index")
uploaded_photos = st.file_uploader("Upload JPG images", type=["jpg"], accept_multiple_files=True)
if st.button("➕ Add to Index"):
    if uploaded_photos:
        with st.spinner("Indexing faces..."):
            added_count = add_images(uploaded_photos)
            st.success(f"✅ Indexed {added_count} faces.")
    else:
        st.warning("Please upload at least one image.")

# Step 2: Reference image
st.subheader("Step 2: Upload Reference Photo")
reference_img = st.file_uploader("Upload a reference image (jpg)", type=["jpg"], key="ref")

if st.button("🔍 Match Faces"):
    if reference_img:
        with st.spinner("Matching faces..."):
            try:
                matches = search(reference_img.read())
                if matches:
                    st.success(f"✅ Found {len(matches)} matches!")
                    for path, sim in matches:
                        st.image(path, caption=f"{Path(path).name} — Similarity: {sim:.2f}", width=200)
                    zip_path = zip_target_photos()
                    with open(zip_path, "rb") as f:
                        st.download_button("📦 Download Matched Faces", f, file_name="matched_faces.zip")
                else:
                    st.error("❌ No matching faces found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please upload a reference image.")
