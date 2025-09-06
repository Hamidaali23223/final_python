# extract_api.py
from fastapi import APIRouter, FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pdfplumber
from typing import List
import uvicorn
import tempfile

router = APIRouter()

def extract_tables_from_pdf(pdf_path: str):
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                all_tables.append(table)
    return all_tables
@router.post("/extract-tables")
async def extract_tables(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        tables = extract_tables_from_pdf(tmp_path)
        
        return JSONResponse(content={"tables": tables})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
