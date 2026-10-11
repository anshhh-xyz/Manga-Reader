# backend/app.py
"""
FastAPI Server for Manga Task Reader
====================================
Accepts 3 manga page images, executes the end-to-end reading pipeline
(Panel detection -> OCR -> Story Filter -> RTL Order -> Speaker Attribution),
and provides a clean, minimal black & white web interface.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline_service import run_pipeline_on_images

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Manga Task Reader AML-2",
    description="End-to-End Japanese Manga Dialogue Extraction & Character Attribution",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Manga Task Reader"}


@app.get("/api/sample")
def get_sample_sequence():
    """Runs pipeline on pre-packaged 3-page development sample sequence."""
    sample_dir = ROOT / "dataset" / "development" / "images" / "seq_2032620aa4e4ac7f"
    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="Sample sequence not found")

    imgs = sorted(sample_dir.glob("*.png"))[:3]
    if len(imgs) != 3:
        raise HTTPException(status_code=400, detail="Sample directory does not contain 3 pages")

    result = run_pipeline_on_images(imgs, seq_name="seq_2032620aa4e4ac7f (Sample)")
    return JSONResponse(content=result)


@app.post("/api/analyze")
async def analyze_three_pages(
    page1: UploadFile = File(...),
    page2: UploadFile = File(...),
    page3: UploadFile = File(...),
    sequence_name: str = Form("custom_sequence")
):
    """
    Accepts exactly 3 manga page images and returns:
    - Structured competition-format JSON
    - Reading script transcript
    - Annotated image overlays (panels + speaker tags)
    - Sequence statistics
    """
    uploads = [page1, page2, page3]
    for idx, u in enumerate(uploads, 1):
        if not u.filename:
            raise HTTPException(status_code=400, detail=f"Page {idx} file is missing")

    temp_dir = Path(tempfile.mkdtemp(prefix="manga_reader_"))
    try:
        saved_paths = []
        for idx, upload in enumerate(uploads, 1):
            ext = Path(upload.filename).suffix or ".png"
            dest = temp_dir / f"page_{idx}{ext}"
            content = await upload.read()
            dest.write_bytes(content)
            saved_paths.append(dest)

        result = run_pipeline_on_images(saved_paths, seq_name=sequence_name)
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Mount static directory for frontend assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Manga Reader Frontend Not Found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)
