from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil, uuid, os
from main.face_matcher import process_uploads, match_faces

app = FastAPI()

@app.post("/upload/")
async def upload_images(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    os.makedirs(f"/tmp/{session_id}/uploads", exist_ok=True)

    for file in files[:100]:
        path = f"/tmp/{session_id}/uploads/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    return {"session_id": session_id, "uploaded": len(files)}

@app.post("/reference/")
async def reference_face(session_id: str, reference: UploadFile = File(...)):
    session_dir = f"/tmp/{session_id}"
    os.makedirs(session_dir, exist_ok=True)
    ref_path = f"{session_dir}/reference.jpg"

    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    zip_path = match_faces(session_dir, ref_path)
    return FileResponse(zip_path, filename="matched_faces.zip")
