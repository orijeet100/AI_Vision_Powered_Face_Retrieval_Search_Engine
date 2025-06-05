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
import requests
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
import rawpy  # For RAW image formats
import pillow_heif  # For HEIC/HEIF formats
from dotenv import load_dotenv

load_dotenv()

# ───── Setup ─────────────────────────────────────────────────────────────
DIST_THR = 0.7

# Register HEIF opener for PIL
pillow_heif.register_heif_opener()

# Supported image formats
SUPPORTED_FORMATS = {
    # Standard formats
    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp',
    # iPhone formats
    'heic', 'heif',
    # RAW formats
    'nef', 'cr2', 'cr3', 'arw', 'dng', 'orf', 'raf', 'rw2', 'pef', 'srw',
    # Other formats
    'gif', 'ico', 'jfif', 'jpe', 'pbm', 'pgm', 'ppm', 'xbm', 'xpm'
}


# ───── Model Load ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face model (~600MB)...")
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


face_app = load_model()


# ───── Image Processing Helper Functions ────────────────────────────────
def convert_image_to_cv2(image_bytes, filename):
    """Convert various image formats to OpenCV format"""
    try:
        file_ext = Path(filename).suffix.lower().lstrip('.')

        # Try OpenCV first for standard formats
        if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img

        # Handle RAW formats
        if file_ext in ['nef', 'cr2', 'cr3', 'arw', 'dng', 'orf', 'raf', 'rw2', 'pef', 'srw']:
            try:
                with rawpy.imread(BytesIO(image_bytes)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
                    # Convert RGB to BGR for OpenCV
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    return bgr
            except Exception as e:
                st.warning(f"Failed to process RAW image {filename}: {e}")
                return None

        # Handle HEIC/HEIF and other formats using PIL
        try:
            pil_img = Image.open(BytesIO(image_bytes))

            # Auto-rotate based on EXIF data
            pil_img = ImageOps.exif_transpose(pil_img)

            # Convert to RGB if needed
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # Convert PIL to OpenCV format
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_bgr

        except Exception as e:
            st.warning(f"Failed to process image {filename}: {e}")
            return None

    except Exception as e:
        st.warning(f"Error processing {filename}: {e}")
        return None


def get_supported_formats_for_ui():
    """Get list of supported formats for file uploader"""
    # File uploader formats (remove some that might not work in browser)
    ui_formats = [
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp',
        'heic', 'heif', 'gif', 'jfif'
    ]
    return ui_formats


def create_drive_query():
    """Create Google Drive API query for all supported image formats"""
    mime_conditions = []

    # Standard MIME types
    standard_mimes = [
        'image/jpeg', 'image/png', 'image/bmp', 'image/tiff',
        'image/webp', 'image/gif', 'image/heic', 'image/heif'
    ]

    for mime in standard_mimes:
        mime_conditions.append(f"mimeType='{mime}'")

    # Also search by file extension for formats that might not have proper MIME types
    ext_conditions = []
    for ext in SUPPORTED_FORMATS:
        ext_conditions.append(f"name contains '.{ext}'")

    # Combine conditions
    all_conditions = mime_conditions + ext_conditions
    return ' or '.join(all_conditions)


# ───── Google Drive Helper Functions ────────────────────────────────────
def extract_drive_folder_id(drive_link):
    """Extract folder ID from Google Drive link"""
    patterns = [
        r'drive\.google\.com/drive/folders/([a-zA-Z0-9-_]+)',
        r'drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9-_]+)',
        r'drive\.google\.com/folderview\?id=([a-zA-Z0-9-_]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            return match.group(1)
    return None


def check_drive_folder_permissions(folder_id):
    """Check if Google Drive folder is publicly accessible"""
    try:
        # Try to access the folder metadata
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except:
        return False


def get_all_folders_recursive(folder_id, api_key, visited=None):
    """Recursively get all folder IDs including nested folders"""
    if visited is None:
        visited = set()

    if folder_id in visited:
        return []  # Avoid infinite loops

    visited.add(folder_id)
    all_folders = [folder_id]

    try:
        # Get all folders within this folder
        api_url = f"https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            'key': api_key,
            'fields': 'files(id,name)',
            'pageSize': 1000
        }

        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            folders = data.get('files', [])

            # Recursively get all nested folders
            for folder in folders:
                nested_folders = get_all_folders_recursive(folder['id'], api_key, visited)
                all_folders.extend(nested_folders)

        return all_folders
    except Exception as e:
        st.warning(f"Error getting nested folders: {e}")
        return all_folders

def get_drive_images_list(folder_id):
    """Get list of image files from Google Drive folder and all nested folders"""
    try:
        # Get API key from environment variable
        api_key = os.getenv('DRIVE_API_KEY')
        if not api_key:
            st.error("Google Drive API key not configured. Please contact administrator.")
            return None

        # Get all folder IDs (including nested ones)
        st.info("🔍 Scanning folder structure...")
        all_folder_ids = get_all_folders_recursive(folder_id, api_key)

        if len(all_folder_ids) > 1:
            st.success(f"📁 Found {len(all_folder_ids)} folders to scan (including subfolders)")

        api_url = f"https://www.googleapis.com/drive/v3/files"
        all_valid_files = []

        # Create comprehensive query for all image formats
        image_query = create_drive_query()

        # Search in all folders (including nested ones)
        for i, current_folder_id in enumerate(all_folder_ids):
            try:
                full_query = f"'{current_folder_id}' in parents and ({image_query})"

                params = {
                    'q': full_query,
                    'key': api_key,
                    'fields': 'files(id,name,mimeType,size,parents)',
                    'pageSize': 1000
                }

                response = requests.get(api_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    files = data.get('files', [])

                    # Filter by extension as additional check
                    for file in files:
                        file_ext = Path(file['name']).suffix.lower().lstrip('.')
                        if file_ext in SUPPORTED_FORMATS:
                            all_valid_files.append(file)

                # Show progress for multiple folders
                if len(all_folder_ids) > 1:
                    progress = (i + 1) / len(all_folder_ids)
                    st.progress(progress, text=f"Scanning folder {i + 1}/{len(all_folder_ids)}")

            except Exception as e:
                st.warning(f"Error scanning folder {current_folder_id}: {e}")
                continue

        # Remove duplicates (in case same file appears in multiple searches)
        unique_files = {}
        for file in all_valid_files:
            unique_files[file['id']] = file

        final_files = list(unique_files.values())

        if len(all_folder_ids) > 1:
            st.empty()  # Clear progress bar

        return final_files

    except Exception as e:
        st.error(f"Error accessing Drive API: {e}")
        return None


def download_drive_image(file_id, filename):
    """Download image from Google Drive"""
    try:
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to download {filename}: {e}")
        return None


def process_drive_folder(drive_link):
    """Process all images from Google Drive folder"""
    # Extract folder ID
    folder_id = extract_drive_folder_id(drive_link)
    if not folder_id:
        return None, "Invalid Google Drive link format"

    # Check permissions
    if not check_drive_folder_permissions(folder_id):
        return None, "Drive folder is not publicly accessible or doesn't exist"

    # Get list of images
    st.info("📂 Accessing Google Drive folder...")
    image_files = get_drive_images_list(folder_id)

    if image_files is None:
        return None, "Could not access folder contents. Make sure the folder is public."

    if not image_files:
        return None, "No image files found in the folder"

    st.success(f"📸 Found {len(image_files)} images in Drive folder")

    # Download and process images
    new_embeddings = []
    new_image_data = []
    processed_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_info in enumerate(image_files):
        file_id = file_info['id']
        filename = file_info['name']

        status_text.text(f"Processing {filename} ({i + 1}/{len(image_files)})")

        # Download image
        img_bytes = download_drive_image(file_id, filename)
        if img_bytes is None:
            continue

        # Process image for faces using enhanced conversion
        img = convert_image_to_cv2(img_bytes, filename)
        if img is None:
            continue

        faces = face_app.get(img)
        if not faces:
            continue

        # Store image data and embeddings
        face_count = len(faces)
        new_image_data.append(((filename, img_bytes), face_count))

        for face in faces:
            new_embeddings.append(face["embedding"])

        processed_count += 1
        progress_bar.progress((i + 1) / len(image_files))

    progress_bar.empty()
    status_text.empty()

    if new_embeddings:
        _save_face_data(new_embeddings, new_image_data)
        return len(new_embeddings), f"Successfully processed {processed_count} images with faces from Drive folder"
    else:
        return 0, "No faces found in any of the Drive folder images"


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

# Initialize method tracking in session state
if 'file_method_used' not in st.session_state:
    st.session_state.file_method_used = False
if 'drive_method_used' not in st.session_state:
    st.session_state.drive_method_used = False


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
    # Add embeddings to FAISS index
    if new_embeddings:
        # Convert to proper numpy array format
        embeddings_array = np.vstack(new_embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings_array)

        # Get or create index and update session state
        face_index = _get_index()

        # Add embeddings to FAISS index
        face_index.add(embeddings_array)

        # Add to face_embeddings list in session state
        st.session_state.face_embeddings.extend(new_embeddings)

        # Track which image each face belongs to
        start_img_idx = len(st.session_state.image_data)
        for i, (img_data, face_count) in enumerate(new_image_data):
            st.session_state.image_data.append(img_data)
            # Map each face from this image to the image index
            for _ in range(face_count):
                st.session_state.face_to_image_map.append(start_img_idx + i)


# ───── Add user images ───────────────────────────────────────────────────
def add_images(files):
    new_embeddings = []
    new_image_data = []
    processed_files = 0
    total_files = len(files)

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(files):
        status_text.text(f"Processing {file.name} ({i + 1}/{total_files})")

        file.seek(0)  # Reset file pointer
        file_bytes = file.read()

        # Use enhanced image conversion
        img = convert_image_to_cv2(file_bytes, file.name)
        if img is None:
            st.warning(f"Could not process {file.name} - unsupported format or corrupted file")
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

        processed_files += 1
        progress_bar.progress((i + 1) / total_files)

    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()

    # Show processing summary
    if processed_files > 0:
        st.info(f"📊 Processed {processed_files}/{total_files} files with faces")

    if new_embeddings:
        _save_face_data(new_embeddings, new_image_data)
        return len(new_embeddings)
    return 0


# ───── Search with robust ZIP creation ──────────────────────────────────
def match_faces(reference_file) -> Tuple[Optional[bytes], str]:
    """
    Returns (zip_bytes, message) where zip_bytes is None on error
    """
    try:
        current_count = get_current_face_count()

        if current_count == 0:
            return None, "No faces in database. Please add images first."

        # Process reference image with enhanced conversion
        reference_file.seek(0)
        file_bytes = reference_file.read()

        img = convert_image_to_cv2(file_bytes, reference_file.name)
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
    st.session_state.file_method_used = False
    st.session_state.drive_method_used = False
    if 'initial_load_shown' in st.session_state:
        del st.session_state.initial_load_shown

# Step 1: Upload Database Images
st.header("Step 1: Upload Database Images")

# Get supported formats (needed for both file upload and reference upload)
supported_formats = get_supported_formats_for_ui()

# Upload method selection toggle
upload_method = st.radio(
    "Choose upload method:",
    ["📁 Upload Files", "🔗 Google Drive Link"],
    key="upload_method",
    horizontal=True
)

st.markdown("---")

# Option A: Upload files directly
if upload_method == "📁 Upload Files":
    st.subheader("Upload Image Files")
    st.info(f"📸 **Supported formats**: {', '.join([fmt.upper() for fmt in supported_formats])}")
    st.info("💡 **Includes**: iPhone photos (HEIC), RAW files (NEF, CR2, ARW, etc.), and standard formats")

    img_files = st.file_uploader(
        "Upload images for database",
        type=supported_formats,
        accept_multiple_files=True,
        key="database_upload",
        disabled=st.session_state.get('drive_method_used', False)
    )

    # Process file uploads
    if img_files and not st.session_state.database_uploaded:
        with st.spinner("Please wait, the photos are being uploaded..."):
            st.session_state.database_uploaded = True
            st.session_state.file_method_used = True
        st.success("✅ Database of images successfully loaded")
        st.success("👆 Click 'Start Face Indexing' button below when ready")

    # Show indexing button for file uploads
    if st.session_state.get('database_uploaded') and img_files and not st.session_state.faces_indexed:
        if st.button("📁 Start Face Indexing", type="primary"):
            with st.spinner("Indexing faces, please wait..."):
                count = add_images(img_files)

            if count > 0:
                st.success(f"✅ Face indexing finished! {count} faces indexed")
                st.success("🎯 Now you can upload a reference image below")
                st.session_state.faces_indexed = True
                st.rerun()
            else:
                st.warning("⚠️ No faces found in uploaded images.")

# Option B: Google Drive folder
elif upload_method == "🔗 Google Drive Link":
    st.subheader("Google Drive Folder")
    st.info("""
    🔗 **How to use Google Drive folder:**
    1. Create a folder in Google Drive with your images
    2. Right-click the folder → Share → Change to "Anyone with the link"
    3. Copy the folder link and paste it below
    """)

    drive_link = st.text_input(
        "Google Drive folder link:",
        placeholder="https://drive.google.com/drive/folders/your-folder-id",
        disabled=st.session_state.get('file_method_used', False)
    )

    if drive_link and not st.session_state.database_uploaded:
        if st.button("🔗 Load Images from Drive", type="primary"):
            with st.spinner("Loading images from Google Drive..."):
                face_count, message = process_drive_folder(drive_link)
                if face_count is not None and face_count > 0:
                    st.session_state.database_uploaded = True
                    st.session_state.faces_indexed = True
                    st.session_state.drive_method_used = True
                    st.success(f"✅ {message}")
                    st.success(f"🎯 Face indexing finished! {face_count} faces indexed")
                    st.success("📤 Now you can upload a reference image below")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

    elif drive_link and st.session_state.database_uploaded:
        st.success("✅ Google Drive processing completed!")
        st.info(f"📊 Database ready with {get_current_face_count()} faces indexed")

# Step 2: Upload Reference Image (only show if faces are indexed)
if st.session_state.faces_indexed:
    st.header("Step 2: Upload Reference Image")

    ref_file = st.file_uploader(
        "Reference photo",
        type=supported_formats,
        key="reference_upload",
        help="Upload any supported image format including HEIC, RAW, etc."
    )

    if ref_file and not st.session_state.reference_uploaded:
        with st.spinner("Please wait, reference image is being processed..."):
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
            st.success("📦 Matched images package created successfully!")

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
