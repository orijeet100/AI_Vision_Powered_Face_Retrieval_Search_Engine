import shutil, os
from pathlib import Path
from app.model_setup import load_model, get_embeddings, build_faiss_index, search_faiss

def process_uploads(image_dir: str):
    model = load_model()
    image_paths = list(Path(image_dir).glob("*.jpg"))
    embeddings, paths = get_embeddings(image_paths, model)
    index = build_faiss_index(embeddings)
    return index, paths

def match_faces(session_dir: str, ref_path: str):
    upload_dir = f"{session_dir}/uploads"
    target_dir = f"{session_dir}/target_photos"
    os.makedirs(target_dir, exist_ok=True)

    model = load_model()
    index, paths = process_uploads(upload_dir)
    ref_embedding, _ = get_embeddings([ref_path], model)
    matches = search_faiss(ref_embedding[0], index, paths, threshold=0.7)

    for match in matches:
        shutil.copy(match, f"{target_dir}/{Path(match).name}")

    # 🛑 If no matches, return None
    if not os.listdir(target_dir):
        return None

    return shutil.make_archive(f"{session_dir}/matched_faces", 'zip', target_dir)
