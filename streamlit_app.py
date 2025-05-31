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
# streamlit_app.py

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTANT: st.set_page_config() MUST be the first Streamlit command
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st

st.set_page_config(page_title="Face Retrieval App", layout="centered")

# Now import everything else
import os
import cv2
import faiss
import zipfile
import shutil
import pickle
import numpy as np
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis

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
        file.seek(0)  # Reset file pointer
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


# ───── Search with robust ZIP creation ──────────────────────────────────
def match_faces(reference_file) -> Tuple[Optional[bytes], str]:
    """
    Returns (zip_bytes, message) where zip_bytes is None on error
    """
    try:
        # Clean up target directory
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        TARGET_DIR.mkdir(exist_ok=True)

        index = _load_index()
        meta = _load_meta()

        if index.ntotal == 0:
            return None, "No faces in database. Please add images first."

        # Process reference image
        reference_file.seek(0)
        file_bytes = reference_file.read()

        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "Could not decode reference image."

        faces = face_app.get(img)
        if not faces:
            return None, "No face detected in reference image."

        # Search for matches
        q = faces[0]["embedding"].astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = index.search(q, min(50, index.ntotal))

        # Collect matches
        matched_images = []
        results = []

        for d, i in zip(D[0], I[0]):
            if i == -1:  # Invalid index
                continue

            similarity = float(d)
            distance = 1 - similarity / 2

            if distance < DIST_THR and i < len(meta):
                source_path = meta[i]
                if not os.path.exists(source_path):
                    continue

                # Read source image
                try:
                    with open(source_path, 'rb') as f:
                        img_data = f.read()

                    fname = Path(source_path).name
                    matched_images.append((fname, img_data))
                    results.append((source_path, similarity))

                except Exception as e:
                    st.warning(f"Could not read {source_path}: {e}")
                    continue

        if not matched_images:
            return None, f"No faces matched with threshold {DIST_THR}"

        # Create ZIP file using temporary file for better reliability
        with tempfile.NamedTemporaryFile() as temp_file:
            with zipfile.ZipFile(temp_file, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for fname, img_data in matched_images:
                    zipf.writestr(fname, img_data)

            # Read the complete ZIP file
            temp_file.seek(0)
            zip_bytes = temp_file.read()

        if len(zip_bytes) == 0:
            return None, "Generated ZIP file is empty"

        return zip_bytes, f"Found {len(matched_images)} matching faces"

    except Exception as e:
        return None, f"Error during face matching: {str(e)}"


# ───── Utility function for download link ───────────────────────────────
def create_download_link(zip_bytes: bytes, filename: str = "target_images.zip") -> str:
    """Create a download link using base64 encoding"""
    b64 = base64.b64encode(zip_bytes).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="{filename}">📦 Click here to download {filename}</a>'


# ───── UI ────────────────────────────────────────────────────────────────
st.title("🔍 Face Retrieval System")

# Show current database status
index = _load_index()
meta = _load_meta()
st.sidebar.info(f"Database: {index.ntotal} faces indexed")

# Upload images
st.header("Step 1: Upload Images to Index")
img_files = st.file_uploader("Upload JPG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if img_files and st.button("📁 Add to Face Index"):
    with st.spinner("Indexing faces..."):
        count = add_images(img_files)
        if count > 0:
            st.success(f"✅ {count} faces indexed!")
            st.rerun()  # Refresh to update sidebar
        else:
            st.warning("⚠️ No faces found in uploaded images.")

# Upload reference
st.header("Step 2: Upload Reference Image")
ref_file = st.file_uploader("Reference photo", type=["jpg", "jpeg"])

# Face matching with improved download
if ref_file and st.button("🔍 Match Faces"):
    with st.spinner("Matching faces..."):
        zip_bytes, message = match_faces(ref_file)

    if zip_bytes is None:
        st.error(f"❌ {message}")
    else:
        st.success(f"✅ {message}")

        # Show file info
        zip_size_mb = len(zip_bytes) / (1024 * 1024)
        st.info(f"ZIP file size: {zip_size_mb:.2f} MB ({len(zip_bytes):,} bytes)")

        # Primary download button
        st.download_button(
            label="📦 Download Target Images",
            data=zip_bytes,
            file_name="target_images.zip",
            mime="application/zip",
            key="download_main",
            help="Click to download matched face images"
        )

        # Alternative download method if the button fails
        st.markdown("---")
        st.markdown("**Alternative Download Method:**")
        download_link = create_download_link(zip_bytes, "target_images.zip")
        st.markdown(download_link, unsafe_allow_html=True)

        # Show some statistics
        try:
            with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
                file_list = zf.namelist()
                st.success(f"ZIP contains {len(file_list)} images")

                # Show first few filenames
                if len(file_list) <= 10:
                    st.write("Files in ZIP:", ", ".join(file_list))
                else:
                    st.write(f"Files in ZIP: {', '.join(file_list[:5])} ... and {len(file_list) - 5} more")

        except Exception as e:
            st.warning(f"Could not read ZIP contents: {e}")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tips**: Use clear, well-lit photos for better face detection. The system works best with frontal face images.")