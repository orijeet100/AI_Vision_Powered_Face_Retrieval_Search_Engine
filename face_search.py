# face_search.py

import os
import cv2
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List
from insightface.app import FaceAnalysis

# ───── Config ─────────────────────────────────────────────────────────────
DATABASE_DIR = Path("database")          # initial folder of jpg images
DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "user_photos"       # internal vault
INDEX_FILE = DATA_DIR / "faces.faiss"
META_FILE = DATA_DIR / "faces.pkl"
DIST_THR = 0.7                          # cosine distance threshold

# ───── Model Init ─────────────────────────────────────────────────────────
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ───── Setup Directories ──────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# ───── Helpers ────────────────────────────────────────────────────────────
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

# ───── Add Images ─────────────────────────────────────────────────────────
def add_images_from_directory(directory: Path, max_images: int = 100):
    all_paths = list(directory.glob("*.jpg"))[:max_images]
    index = _load_or_create_index()
    meta = _load_meta()

    new_vecs, new_paths = [], []

    for path in all_paths:
        dst_path = IMG_DIR / path.name
        if dst_path.exists():
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"⚠️ Could not read {path}")
            continue

        faces = face_app.get(img)
        if not faces:
            continue

        # Save to vault
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
        print(f"✅ Indexed {len(new_vecs)} faces from {len(set(new_paths))} images.")
    else:
        print("⚠️ No new valid faces found.")

# ───── Search ─────────────────────────────────────────────────────────────
def search(reference_img_path: str, k: int = 50) -> List[tuple]:
    """
    Search for faces matching the reference image.
    Saves matches into target_photos/ and returns [(image_path, similarity)].
    """
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise RuntimeError("Index not found. Add images first.")

    index = _load_or_create_index()
    meta = _load_meta()
    TARGET_DIR = Path("target_photos")
    TARGET_DIR.mkdir(exist_ok=True)

    img = cv2.imread(reference_img_path)
    if img is None:
        raise FileNotFoundError(reference_img_path)

    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face detected in reference image.")

    q = faces[0]["embedding"].astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, min(k, index.ntotal))

    results = []
    for d, i in zip(D[0], I[0]):
        similarity = d  # cosine similarity
        distance = 1 - similarity / 2
        if distance < DIST_THR:
            source_path = meta[i]
            target_path = TARGET_DIR / Path(source_path).name
            if not target_path.exists():
                img_match = cv2.imread(source_path)
                if img_match is not None:
                    cv2.imwrite(str(target_path), img_match)
            results.append((source_path, similarity))

    return sorted(results, key=lambda x: -x[1])  # sort by similarity



# ───── CLI Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
Usage:
  python face_search.py init             # index images in /database
  python face_search.py find selfie.jpg # find matches for reference face
""")
        exit()

    cmd = sys.argv[1]

    if cmd == "init":
        if not DATABASE_DIR.exists():
            print("❌ 'database/' directory not found.")
            exit()
        add_images_from_directory(DATABASE_DIR)
    elif cmd == "find":
        if len(sys.argv) != 3:
            print("Usage: python face_search.py find reference.jpg")
            exit()
        try:
            matches = search(sys.argv[2])
            if matches:
                print(f"\n🔍 Found {len(matches)} matching images:")
                for path, score in matches:
                    print(f"  ➤ {path}   [similarity = {score:.4f}]")
            else:
                print("❌ No matching images found.")
        except Exception as e:
            print(f"⚠️ Error: {e}")
    else:
        print("Unknown command.")

# python face_search.py init
# python face_search.py find reference.jpg