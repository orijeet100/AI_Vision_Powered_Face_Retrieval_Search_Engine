from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil, uuid, os
from app.face_matcher import process_uploads, match_faces

app = FastAPI()

@app.post("/upload/")
async def upload_images(files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    os.makedirs(f"/tmp/{session_id}/uploads", exist_ok=True)
    for file in files[:100]:
        with open(f"/tmp/{session_id}/uploads/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
    return {"session_id": session_id, "uploaded": len(files)}

@app.post("/reference/")
async def reference_face(session_id: str, reference: UploadFile = File(...)):
    ref_path = f"/tmp/{session_id}/reference.jpg"
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)
    zip_path = match_faces(f"/tmp/{session_id}", ref_path)
    return FileResponse(zip_path, filename="target_photos.zip")
