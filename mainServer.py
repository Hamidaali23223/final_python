import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes import paper_checking, time_table, paper_generate, subjective_Paper_checking
from config import env
app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs(env.UPLOAD_DIR, exist_ok=True)
os.makedirs(env.Y_REF_DIR, exist_ok=True)
os.makedirs(env.PROCESSED_REF_DIR, exist_ok=True)
os.makedirs(env.PROCESSED_STUDENT_DIR, exist_ok=True)

app.mount("/uploaded_refrence_images", StaticFiles(directory=env.UPLOAD_DIR), name="uploaded_refrence_images")
app.mount("/Y_reference_images", StaticFiles(directory=env.Y_REF_DIR), name="Y_reference_images")
app.mount("/processed_reference_images", StaticFiles(directory=env.PROCESSED_REF_DIR), name="processed_reference_images")
app.mount("/student_processed_images", StaticFiles(directory=env.PROCESSED_STUDENT_DIR), name="student_processed_images")
# Include routers
app.include_router(paper_checking.router, prefix="/api", tags=["MCQ Processing"])
app.include_router(time_table.router, prefix="/time_table", tags=["Get TimeTable"])
app.include_router(paper_generate.router, prefix="/paper_generate", tags=["Generate Paper"])
app.include_router(subjective_Paper_checking.router, prefix="/subjective_Paper_checking", tags=["Subjective Paper Checking"])

@app.get("/")
def health_check():
    return {"status": "active"}