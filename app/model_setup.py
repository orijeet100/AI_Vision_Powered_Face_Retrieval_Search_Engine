import faiss, cv2, numpy as np
from insightface.app import FaceAnalysis

def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

def get_embeddings(image_paths, model):
    embeddings, valid_paths = [], []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        faces = model.get(img)
        if not faces: continue
        embeddings.append(faces[0]['embedding'])
        valid_paths.append(str(p))
    return np.array(embeddings).astype("float32"), valid_paths

def build_faiss_index(vectors):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def search_faiss(query, index, paths, threshold=0.7):
    query = query.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query)
    D, I = index.search(query, k=len(paths))
    return [paths[i] for d, i in zip(D[0], I[0]) if (1 - d/2) < threshold]
