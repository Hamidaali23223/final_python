import asyncio
import os
import io
import re
import json
import base64
import time
import unicodedata
from typing import List, Optional, Tuple

from bson import ObjectId
import pdfplumber
from fastapi import APIRouter, Body, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from openai import OpenAI
from rapidfuzz import fuzz, distance
from config import env
from database.db import (
   subjectivePaper_collection,
   jobs_collection
)



router = APIRouter()

OPENAI_BASE_URL=env.ENDPOINT
OPENAI_API_KEY=env.TOKEN
OPENAI_MODEL=env.MODEL_NAME


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# ------------------- Schemas -------------------
class SubjectivePayload(BaseModel):
    student_name: Optional[str]
    roll_number: Optional[str]
    subject_name: Optional[str]
    term: Optional[str]
    semester: Optional[str]
    short_question_marks: List[int] = Field(default_factory=list)
    long_question_marks: List[int] = Field(default_factory=list)

    @validator("student_name", "roll_number", "subject_name", "term", "semester", pre=True)
    def blank_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


# ------------------- Utils: Normalization -------------------
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

def norm_name(name: str) -> str:
    name = strip_accents(name or "")
    name = name.upper()
    name = re.sub(r"[^A-Z\s\.\'-]", " ", name)
    return normalize_spaces(name)

def norm_roll(roll: str) -> str:
    roll = (roll or "").upper()
    roll = re.sub(r"[^A-Z0-9]", "", roll)  # keep only A-Z, 0-9
    return roll


# ------------------- PDF Parsing -------------------
ROLL_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{6,}\b")

def extract_students_from_pdf(pdf_bytes: bytes) -> List[dict]:
    students = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text:
                continue
            lines = [normalize_spaces(line) for line in text.splitlines() if line.strip()]
            for line in lines:
                rolls = ROLL_PATTERN.findall(line)
                if not rolls:
                    continue
                roll = rolls[0]
                idx = line.find(roll)
                trailing = line[idx + len(roll):].strip()
                trailing = re.sub(r"^\d+\s*", "", trailing)

                if not trailing:
                    tokens = line.split()
                    try:
                        rpos = tokens.index(roll)
                        trailing = " ".join(tokens[rpos + 1:])
                    except ValueError:
                        trailing = ""

                if trailing:
                    students.append({
                        "roll_number": norm_roll(roll),
                        "student_name": norm_name(trailing)
                    })

    seen = set()
    unique = []
    for s in students:
        if s["roll_number"] not in seen:
            seen.add(s["roll_number"])
            unique.append(s)
    return unique


# ------------------- Matching -------------------
def roll_suffix_match(extracted_roll: str, candidate_roll: str, length: int) -> bool:
    return extracted_roll[-length:] == candidate_roll[-length:]

def best_match_from_pdf(
    extracted_name: Optional[str],
    extracted_roll: Optional[str],
    pdf_students: List[dict]
) -> Tuple[Optional[dict], dict]:
    diagnostics = {"candidates_checked": 0, "top_alt": []}

    if not pdf_students:
        return None, {**diagnostics, "reason": "empty_pdf_list"}

    extracted_roll_n = norm_roll(extracted_roll or "")
    extracted_name_n = norm_name(extracted_name or "")

    def score_student(s):
        roll_sim = 0.0
        name_sim = 0.0
        if extracted_roll_n:
            roll_sim = 100.0 * distance.Levenshtein.normalized_similarity(extracted_roll_n, s["roll_number"])
        if extracted_name_n:
            name_sim = float(fuzz.token_set_ratio(extracted_name_n, s["student_name"]))

        if extracted_roll_n and extracted_name_n:
            combined = 0.7 * roll_sim + 0.3 * name_sim
        elif extracted_roll_n:
            combined = roll_sim
        else:
            combined = name_sim
        return combined, roll_sim, name_sim

    scored = []
    for s in pdf_students:
        combined, roll_sim, name_sim = score_student(s)
        scored.append((combined, roll_sim, name_sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    diagnostics["candidates_checked"] = len(scored)
    diagnostics["top_alt"] = [
        {
            "roll_number": sc[3]["roll_number"],
            "student_name": sc[3]["student_name"],
            "combined": round(sc[0], 2),
            "roll_sim": round(sc[1], 2),
            "name_sim": round(sc[2], 2),
        }
        for sc in scored[:5]
    ]

    best = scored[0]
    best_combined, best_roll_sim, best_name_sim, best_student = best

    # ---- Confidence thresholds ----
    if extracted_roll_n:
        if best_roll_sim >= 90.0:
            return best_student, {**diagnostics, "decision": "accept_by_roll>=90"}

        # Rule 1: last 4 digits match + decent name
        if roll_suffix_match(extracted_roll_n, best_student["roll_number"], 4):
            if best_name_sim >= 80.0:
                return best_student, {**diagnostics, "decision": "accept_by_suffix4+name"}

        # Rule 2: last 2 digits match + strong name
        if roll_suffix_match(extracted_roll_n, best_student["roll_number"], 2):
            if best_name_sim >= 95.0:
                return best_student, {**diagnostics, "decision": "accept_by_suffix2+name"}

        # Rule 3: name perfect → trust name
        if best_name_sim >= 98.0:
            return best_student, {**diagnostics, "decision": "accept_by_name_override"}

        # Rule 4: fallback combined score
        if best_combined >= 87.0 and best_name_sim >= 80.0:
            return best_student, {**diagnostics, "decision": "accept_by_combined_and_name"}

        return None, {**diagnostics, "reason": "low_confidence_roll", "best": diagnostics["top_alt"][0]}

    if best_name_sim >= 90.0:
        return best_student, {**diagnostics, "decision": "accept_by_name>=90"}

    return None, {**diagnostics, "reason": "low_confidence_name", "best": diagnostics["top_alt"][0]}


# ------------------- GPT Extraction -------------------
def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

EXTRACTION_PROMPT = """
You are a professional examiner. You will receive a photo of a university exam result paper.
Extract the following fields from the image in strict JSON (no markdown, no comments):
1. Student Name
2. Roll Number
3. Subject Name
4. Term (e.g., Final, Mid)
5. Semester
6. Short Question Marks (First position award section)
7. Long Question Marks (Second position award section)
{
  "student_name": "...",
  "roll_number": "...",
  "subject_name": "...",
  "term": "...",
  "semester": "...",
  "short_question_marks": [0],
  "long_question_marks": [0]
}

Rules:
- Output STRICT JSON only.
- short_question_marks must be a single integer value inside an array (example: [30]).
- long_question_marks must be a single integer value inside an array (example: [10]).
- If either is missing, return an empty array [].
- If GPT mistakenly detects multiple numbers in short_question_marks, 
  assign the first one to short_question_marks and the second one to long_question_marks.
"""

def extract_from_image_with_gpt(image_bytes: bytes) -> SubjectivePayload:
    b64 = encode_image_bytes(image_bytes)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}" }},
                ],
            }
        ],
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or ""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}")

    try:
        payload = SubjectivePayload(**data)

        # --- Post-processing rules ---
        short_vals = payload.short_question_marks or []
        long_vals = payload.long_question_marks or []

        # If GPT gave multiple values in short, move second into long
        if len(short_vals) > 1:
            if not long_vals:
                long_vals = [short_vals[1]]
            short_vals = [short_vals[0]]

        # If GPT gave multiple values in long, keep only the first
        if len(long_vals) > 1:
            long_vals = [long_vals[0]]

        payload.short_question_marks = short_vals
        payload.long_question_marks = long_vals

        return payload

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extracted JSON failed validation: {e}")

def convert_objectid(doc):
    if isinstance(doc, list):
        return [convert_objectid(x) for x in doc]
    elif isinstance(doc, dict):
        return {k: convert_objectid(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc


# ------------------- Endpoint -------------------


job_queue = asyncio.Queue()
average_processing_time = 15  # seconds per job (adjust after measuring)

async def worker():
    while True:
        job = await job_queue.get()
        try:
            job_id, image_bytes, pdf_bytes, teacherId, subjectName, selectedDept, selectedClass = job
            print(f"Processing job {job_id}...")

            extracted = extract_from_image_with_gpt(image_bytes)
            pdf_students = extract_students_from_pdf(pdf_bytes)
            best, diag = best_match_from_pdf(
                extracted_name=extracted.student_name,
                extracted_roll=extracted.roll_number,
                pdf_students=pdf_students
            )

            if best:
                corrected = extracted.dict()
                corrected["student_name"] = best["student_name"]
                corrected["roll_number"] = best["roll_number"]

                final_doc = {
                    **corrected,
                    "teacherId": teacherId,
                    "subjectName": subjectName,
                    "selectedDept": selectedDept,
                    "selectedClass": selectedClass,
                    "status": "completed",
                    "createdAt": time.time()
                }
                subjectivePaper_collection.insert_one(final_doc)

                jobs_collection.update_one(
                    {"_id": ObjectId(job_id)},
                    {"$set": {"status": "completed", "result": final_doc, "diagnostics": diag}}
                )
            else:
                jobs_collection.update_one(
                    {"_id": ObjectId(job_id)},
                    {"$set": {
                        "status": "failed",
                        "extracted": extracted.dict(),
                        "diagnostics": diag
                    }}
                )
        except Exception as e:
            jobs_collection.update_one(
                {"_id": ObjectId(job_id)},
                {"$set": {"status": "error", "error": str(e)}}
            )
        finally:
            job_queue.task_done()



# Kick off worker task on startup
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())


# ----------------- Submit Job -----------------
@router.post("/check-subjective-paper-with-pdf")
async def check_subjective_paper_with_pdf(
    image_file: UploadFile = File(...),
    pdf_file: UploadFile = File(...),
    teacherId: str = Form(...),
    subjectName: str = Form(...),
    selectedDept: str = Form(...),
    selectedClass: str = Form(...),
):
    image_bytes = await image_file.read()
    pdf_bytes = await pdf_file.read()

    # Insert job record
    doc = {
        "teacherId": teacherId,
        "subjectName": subjectName,
        "selectedDept": selectedDept,
        "selectedClass": selectedClass,
        "status": "pending",
        "createdAt": time.time()
    }
    result = jobs_collection.insert_one(doc)
    job_id = str(result.inserted_id)

    # Add to queue
    queue_length = job_queue.qsize()
    await job_queue.put((job_id, image_bytes, pdf_bytes, teacherId, subjectName, selectedDept, selectedClass))
    position = queue_length + 1
    eta = position * average_processing_time

    return {
        "success": True,
        "message": "Your request is queued for processing",
        "jobId": job_id,
        "position": position,
        "estimated_time_seconds": eta
    }







# jobs route

@router.get("/jobs/by-teacher/{teacherId}/{className}")
async def get_jobs_for_teacher_and_class(teacherId: str, className: str):
    query = {"teacherId": teacherId, "selectedClass": className}
    docs = list(jobs_collection.find(query).sort("createdAt", -1))
    docs = convert_objectid(docs)
    return {"success": True, "count": len(docs), "data": docs}

# ----------------- Resolve Failed Job -----------------
@router.put("/jobs/{jobId}/resolve")
async def resolve_failed_job(jobId: str, choice: dict = Body(...)):
    """
    choice example:
    {
      "accept": "gpt" | "best" | "manual",
      "data": {
        "student_name": "Correct Name",
        "roll_number": "Correct Roll"
      }
    }
    """
    job = jobs_collection.find_one({"_id": ObjectId(jobId)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "failed":
        raise HTTPException(status_code=400, detail="Job is not in failed state")

    extracted = job.get("extracted", {})
    best = job.get("diagnostics", {}).get("best", {})

    accept = choice.get("accept")
    if accept == "gpt":
        final_name = extracted.get("student_name")
        final_roll = extracted.get("roll_number")
    elif accept == "best":
        final_name = best.get("student_name")
        final_roll = best.get("roll_number")
    elif accept == "manual":
        manual = choice.get("data", {})
        final_name = manual.get("student_name")
        final_roll = manual.get("roll_number")
    else:
        raise HTTPException(status_code=400, detail="Invalid resolution choice")

    final_doc = {
        **extracted,
        "student_name": final_name,
        "roll_number": final_roll,
        "teacherId": job["teacherId"],
        "subjectName": job["subjectName"],
        "selectedDept": job["selectedDept"],
        "selectedClass": job["selectedClass"],
        "status": "completed",
        "resolvedFrom": accept,
        "createdAt": time.time()
    }

    # Save to final collection
    subjectivePaper_collection.insert_one(final_doc)

    # Update job record
    jobs_collection.update_one(
        {"_id": ObjectId(jobId)},
        {"$set": {
            "status": "resolved",
            "resolution": accept,
            "result": final_doc
        }}
    )

    return {"success": True, "message": "Job resolved", "data": convert_objectid(final_doc)}

# ----------------- Job spacific result -----------------
@router.get("/results/status/{jobId}")
async def get_job_status(jobId: str):
    doc = jobs_collection.find_one({"_id": ObjectId(jobId)})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True, "data": convert_objectid(doc)}



@router.get("/results")
async def get_all_results():
    docs = list(subjectivePaper_collection.find({}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return {"success": True, "count": len(docs), "data": docs}

@router.get("/results/by-teacher-class")
async def get_results_by_teacher_class(teacherId: str, selectedClass: str):
    query = {"teacherId": teacherId, "selectedClass": selectedClass}
    docs = list(subjectivePaper_collection.find(query))
    for d in docs:
        d["_id"] = str(d["_id"])
    return {"success": True, "count": len(docs), "data": docs}

@router.delete("/results/{id}")
async def delete_result(id: str):
    try:
        result = subjectivePaper_collection.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        return {"success": True, "message": "Result deleted successfully"}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")

@router.put("/results/{id}")
async def update_result(id: str, update_data: dict = Body(...)):
    try:
        result = subjectivePaper_collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": update_data}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Result not found")
        updated_doc = subjectivePaper_collection.find_one({"_id": ObjectId(id)})
        updated_doc["_id"] = str(updated_doc["_id"])
        return {"success": True, "data": updated_doc}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")