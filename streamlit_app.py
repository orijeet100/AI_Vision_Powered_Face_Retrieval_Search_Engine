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
DIST_THR = 0.7


# ───── Model Load ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face model (~600MB)...")
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


face_app = load_model()


# ───── In-memory storage helpers using session state ────────────────────
def _get_index():
    """Get or create FAISS index in session state"""
    if st.session_state.face_index is None:
        st.session_state.face_index = faiss.IndexFlatIP(512)
    return st.session_state.face_index


def get_current_face_count():
    """Get current face count from session state"""
    if st.session_state.face_index is None:
        return 0
    return st.session_state.face_index.ntotal


def _save_face_data(new_embeddings, new_image_data):
    """Add new face data to session state storage"""
    print(f"DEBUG: _save_face_data called with {len(new_embeddings)} embeddings")

    # Add embeddings to FAISS index
    if new_embeddings:
        # Convert to proper numpy array format
        embeddings_array = np.vstack(new_embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings_array)

        print(f"DEBUG: embeddings_array shape: {embeddings_array.shape}")

        # Get or create index and update session state
        face_index = _get_index()
        print(f"DEBUG: face_index before add: {face_index.ntotal}")

        # Add embeddings to FAISS index
        face_index.add(embeddings_array)
        print(f"DEBUG: face_index after add: {face_index.ntotal}")

        # Add to face_embeddings list in session state
        st.session_state.face_embeddings.extend(new_embeddings)
        print(f"DEBUG: face_embeddings length: {len(st.session_state.face_embeddings)}")

        # Track which image each face belongs to
        start_img_idx = len(st.session_state.image_data)
        for i, (img_data, face_count) in enumerate(new_image_data):
            st.session_state.image_data.append(img_data)
            # Map each face from this image to the image index
            for _ in range(face_count):
                st.session_state.face_to_image_map.append(start_img_idx + i)

        print(
            f"DEBUG: Final counts - FAISS: {face_index.ntotal}, embeddings: {len(st.session_state.face_embeddings)}, images: {len(st.session_state.image_data)}")

# ───── Add user images ───────────────────────────────────────────────────
def add_images(files):
    new_embeddings = []
    new_image_data = []

    for file in files:
        file.seek(0)  # Reset file pointer
        file_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        faces = face_app.get(img)
        if not faces:
            continue

        # Store original image bytes with filename
        file.seek(0)  # Reset pointer again
        original_bytes = file.read()
        face_count = len(faces)
        new_image_data.append(((file.name, original_bytes), face_count))

        # Extract face embeddings
        for face in faces:
            new_embeddings.append(face["embedding"])

    if new_embeddings:
        _save_face_data(new_embeddings, new_image_data)
        # Debug: verify data was saved
        final_count = get_current_face_count()
        return len(new_embeddings)
    return 0


# ───── Search with robust ZIP creation ──────────────────────────────────
def match_faces(reference_file) -> Tuple[Optional[bytes], str]:
    """
    Returns (zip_bytes, message) where zip_bytes is None on error
    """
    try:
        current_count = get_current_face_count()
        print(f"DEBUG: match_faces - current face count: {current_count}")
        print(f"DEBUG: face_index is None: {st.session_state.face_index is None}")
        if st.session_state.face_index:
            print(f"DEBUG: face_index.ntotal: {st.session_state.face_index.ntotal}")

        if current_count == 0:
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
        D, I = st.session_state.face_index.search(q, min(50, st.session_state.face_index.ntotal))

        # Collect unique matched images
        matched_images = {}  # Use dict to avoid duplicates by filename

        for d, i in zip(D[0], I[0]):
            if i == -1 or i >= len(st.session_state.face_to_image_map):  # Invalid index
                continue

            similarity = float(d)
            distance = 1 - similarity / 2

            if distance < DIST_THR:
                img_idx = st.session_state.face_to_image_map[i]
                if img_idx < len(st.session_state.image_data):
                    filename, img_bytes = st.session_state.image_data[img_idx]
                    # Keep the best similarity score for each image
                    if filename not in matched_images or similarity > matched_images[filename][1]:
                        matched_images[filename] = (img_bytes, similarity)

        if not matched_images:
            return None, f"No faces matched with threshold {DIST_THR}"

        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zipf:
            for filename, (img_bytes, similarity) in matched_images.items():
                zipf.writestr(filename, img_bytes)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

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


def create_mobile_friendly_download(zip_bytes: bytes, filename: str = "target_images.zip"):
    """Create mobile-friendly download options"""
    # Method 1: Streamlit's native download button (works best on mobile)
    st.download_button(
        label="📱 Download for Mobile",
        data=zip_bytes,
        file_name=filename,
        mime="application/zip",
        key="mobile_download",
        help="Works best on mobile devices"
    )

    # Method 2: Base64 link (for desktop)
    b64 = base64.b64encode(zip_bytes).decode()
    download_link = f'<a href="data:application/zip;base64,{b64}" download="{filename}">💻 Desktop Download Link</a>'
    st.markdown(download_link, unsafe_allow_html=True)

    # Method 3: Instructions for mobile users
    st.info("""
    📱 **Mobile Download Tips:**
    - Use the "Download for Mobile" button above
    - If that doesn't work, try opening this page in Chrome/Firefox instead of Safari
    - On iPhone: Long-press the download link → "Download Linked File"
    """)


# ───── Session State Management for face data ────────────────────────────
if 'database_uploaded' not in st.session_state:
    st.session_state.database_uploaded = False
if 'faces_indexed' not in st.session_state:
    st.session_state.faces_indexed = False
if 'reference_uploaded' not in st.session_state:
    st.session_state.reference_uploaded = False
if 'download_ready' not in st.session_state:
    st.session_state.download_ready = False
if 'zip_data' not in st.session_state:
    st.session_state.zip_data = None

# Initialize face data in session state
if 'face_index' not in st.session_state:
    st.session_state.face_index = None
if 'face_embeddings' not in st.session_state:
    st.session_state.face_embeddings = []
if 'image_data' not in st.session_state:
    st.session_state.image_data = []
if 'face_to_image_map' not in st.session_state:
    st.session_state.face_to_image_map = []

# ───── UI ────────────────────────────────────────────────────────────────
st.title("🔍 Face Retrieval System")

# Check current database status
current_faces = get_current_face_count()

# Only show database status if faces exist, and only once per session
if current_faces > 0 and 'initial_load_shown' not in st.session_state:
    st.session_state.initial_load_shown = True
    st.session_state.database_uploaded = True
    st.session_state.faces_indexed = True

# Reset session state based on actual database status
if current_faces == 0:
    st.session_state.database_uploaded = False
    st.session_state.faces_indexed = False
    st.session_state.reference_uploaded = False
    st.session_state.download_ready = False
    st.session_state.zip_data = None
    if 'initial_load_shown' in st.session_state:
        del st.session_state.initial_load_shown

# Step 1: Upload Database Images
st.header("Step 1: Upload Database Images")


img_files = st.file_uploader("Upload JPG images for database", type=["jpg", "jpeg"], accept_multiple_files=True,
                             key="database_upload")

if img_files and not st.session_state.database_uploaded:
        # Simulate database upload process
    st.session_state.database_uploaded = True
    st.success("✅ Database of images successfully loaded")

if st.session_state.database_uploaded and img_files and not st.session_state.faces_indexed:
    if st.button("📁 Start Face Indexing"):
        with st.spinner("Indexing faces, please wait..."):
            count = add_images(img_files)

        if count > 0:
            st.success(f"✅ Face indexing finished! {count} faces indexed")
            st.success("🎯 Please upload reference image")
            st.session_state.faces_indexed = True
            # Force refresh of face count
            # st.rerun()
        else:
            st.warning("⚠️ No faces found in uploaded images.")

# Step 2: Upload Reference Image (only show if faces are indexed)
if st.session_state.faces_indexed:
    st.header("Step 2: Upload Reference Image")

    ref_file = st.file_uploader("Reference photo", type=["jpg", "jpeg"], key="reference_upload")

    if ref_file and not st.session_state.reference_uploaded:
        st.session_state.reference_uploaded = True
        st.success("✅ Reference image uploaded successfully")

    # Face Matching (only show if reference is uploaded)
    if ref_file and st.button("🔍 Match Faces"):
        with st.spinner("Matching faces..."):
            zip_bytes, message = match_faces(ref_file)

        if zip_bytes is None:
            st.error(f"❌ {message}")
        else:
            st.success(f"✅ {message}")

            # Store ZIP data in session state
            st.session_state.zip_data = zip_bytes
            st.session_state.download_ready = True

            # Show file info
            zip_size_mb = len(zip_bytes) / (1024 * 1024)
            st.info(f"ZIP file size: {zip_size_mb:.2f} MB ({len(zip_bytes):,} bytes)")

# Download section (show if download is ready)
if st.session_state.download_ready and st.session_state.zip_data:
    st.markdown("---")
    st.subheader("📦 Download Matched Images")

    # Mobile-friendly download button
    st.download_button(
        label="📱 Download Images",
        data=st.session_state.zip_data,
        file_name="target_images.zip",
        mime="application/zip",
        key="mobile_download",
        type="primary"
    )

    # Alternative desktop download link
    b64 = base64.b64encode(st.session_state.zip_data).decode()
    download_link = f'<a href="data:application/zip;base64,{b64}" download="target_images.zip">💻 Alternative Download Link</a>'
    st.markdown(download_link, unsafe_allow_html=True)

    # Mobile download tips
    st.info("""
    📱 **Mobile Download Tips:**
    - Use the "Download Images" button above (works best on mobile)
    - If button doesn't work, try the alternative link
    - On iPhone: Long-press download link → "Download Linked File"
    - Consider using Chrome/Firefox instead of Safari for better download support
    """)

    # Show ZIP contents
    try:
        with zipfile.ZipFile(BytesIO(st.session_state.zip_data), 'r') as zf:
            file_list = zf.namelist()
            st.success(f"ZIP contains {len(file_list)} images")

            # Show filenames
            if len(file_list) <= 10:
                st.write("Files in ZIP:", ", ".join(file_list))
            else:
                st.write(f"Files in ZIP: {', '.join(file_list[:5])} ... and {len(file_list) - 5} more")

    except Exception as e:
        st.warning(f"Could not read ZIP contents: {e}")

# Reset Workflow (simple refresh instruction)
st.markdown("---")
st.info("🔄 **Reset Workflow**: Refresh this page to start over")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tips**: Use clear, well-lit photos for better face detection. The system works best with frontal face images.")