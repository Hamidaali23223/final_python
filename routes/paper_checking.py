import platform
from bson import ObjectId
import cv2
import numpy as np
import base64
import pytesseract
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, BackgroundTasks
from openai import OpenAI
import uuid
import os
from datetime import datetime
from config import env
from database.db import (
    mcq_collection,
    collection,
    teacher_reference_data,
    image_queue,
    processing_active
)

router = APIRouter()


UPLOAD_DIR=env.UPLOAD_DIR
Y_REF_DIR=env.Y_REF_DIR
PROCESSED_REF_DIR=env.PROCESSED_REF_DIR
PROCESSED_STUDENT_DIR=env.PROCESSED_STUDENT_DIR

ENDPOINT=env.ENDPOINT
TOKEN=env.TOKEN
MODEL_NAME=env.MODEL_NAME

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = env.TESSERACT_PATH
else:
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

def set_teacher_reference_data(teacher_id, subject_id, image=None, answers=None, positions=None):
    key = (teacher_id, subject_id)
    if key not in teacher_reference_data:
        teacher_reference_data[key] = {}

    if image is not None:
        teacher_reference_data[key]['image'] = image
    if answers is not None:
        teacher_reference_data[key]['answers'] = answers
    if positions is not None:
        teacher_reference_data[key]['positions'] = positions

def get_teacher_reference_data(teacher_id, subject_id):
    return teacher_reference_data.get((teacher_id, subject_id), {})

def get_teacher_reference_value(teacher_id, subject_id, key_name):
    return teacher_reference_data.get((teacher_id, subject_id), {}).get(key_name)

def reset_teacher_reference_data(teacher_id, subject_id):
    key = (teacher_id, subject_id)
    if key in teacher_reference_data:
        del teacher_reference_data[key]

def encode_image(image_data):
    """Encode image to base64"""
    return base64.b64encode(image_data).decode("utf-8")

def find_first_question_position(image_data):
    """Find Y-coordinate of first question using OCR"""
    try:
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        processed = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
        for i, text in enumerate(ocr_data["text"]):
            if text.strip().lower() in ["1.", "i.", "q.1", "1)", "Q.1"]:
                return ocr_data["top"][i]
    except Exception as e:
        print(f"OCR Error: {e}")
    return None

def crop_mcq_section(image_data, question_y):
    """Crop image from first question's position"""
    if question_y is None:
        raise ValueError("First question position not detected by OCR.")

    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    cropped_image = image[question_y - 10:, :]
    return cv2.imencode('.jpg', cropped_image)[1]

def preprocess_image(image_data, width=840):
    """Resize and preprocess image for detection"""
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(width / aspect_ratio)
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    return resized, hsv

def align_image(ref_image, scanned_image):
    """Align scanned image to reference image"""
    try:
        gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        gray_scanned = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_scanned, None)

        if des1 is None or des2 is None:
            raise ValueError("Could not compute image descriptors")

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 80:
            raise ValueError(f"Not enough good matches found: {len(good_matches)}")

        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        aligned_image = cv2.warpPerspective(scanned_image, H, (ref_image.shape[1], ref_image.shape[0]))

        return aligned_image
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {e}")

def get_answer_choice(x, y, answer_positions, tolerance=10):
    for i, (x_min, y_min, x_max, y_max, cx, cy, x_rel, y_rel) in enumerate(answer_positions):
        if (x_min - tolerance <= x <= x_max + tolerance) and (y_min - tolerance <= y <= y_max + tolerance):
            # Match found, return current index and choice based on x_rel
            closest_choice = min(env.MCQ_GRID, key=lambda k: abs(env.MCQ_GRID[k] - x_rel))
            return i, closest_choice
    return None, None

def detect_marks(hsv, image_width, image_height):
    color_ranges = {
        "blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
        "red1": (np.array([0, 70, 50]), np.array([10, 255, 255])),
        "red2": (np.array([170, 70, 50]), np.array([180, 255, 255])),
    }
    
    mask_combined = np.zeros_like(hsv[:, :, 0])
    for (lower, upper) in color_ranges.values():
        mask = cv2.inRange(hsv, lower, upper)
        mask_combined = cv2.bitwise_or(mask_combined, mask)

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answer_positions = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2, y2 = x1 + w, y1 + h
            
            # Calculate Center Position (X, Y)
            X = (x1 + x2) // 2
            Y = (y1 + y2) // 2

            # Normalize Position (Relative to MCQ Sheet)
            X_rel = round(X / image_width, 4)
            Y_rel = round(Y / image_height, 4)

            # Store bounding box + absolute & relative positions
            answer_positions.append((x1, y1, x2, y2, X, Y, X_rel, Y_rel))

    return sorted(answer_positions, key=lambda pos: (pos[7], pos[6]))

def extract_reference_answers(image_data):
    """Enhanced reference answer extraction"""

    image, hsv = preprocess_image(image_data)
    height, width = image.shape[:2]
    
    # Detect all marks
    reference_positions = detect_marks(hsv, width, height)

    # Fail fast if no marks found
    if not reference_positions:
        raise HTTPException(
            status_code=400,
            detail="No answer bubbles detected in the reference MCQ image. Please upload a clearer image."
        )

    detected_answers = ["Missing"] * len(reference_positions)
    final_answers = detected_answers.copy()

    for (x1, y1, x2, y2, X, Y, X_rel, Y_rel) in reference_positions:
        q_index, choice = get_answer_choice(x1, y1, reference_positions)
    
        if q_index is not None and choice is not None:
            # Only write if it's still marked as 'Missing'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(image, f"Q{q_index+1}: {choice}", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if detected_answers[q_index] == "Missing":
                detected_answers[q_index] = choice
        else:
            print(f"⚠️ Warning: Question {q_index+1} already has an answer: {detected_answers[q_index]}. Skipping duplicate.")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red boxes
            cv2.putText(image, f"({X_rel}, {Y_rel})", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        final_answers = detected_answers  # No need to recreate it

    print("\n✅ Final Detected Answers:")
    for i, ans in enumerate(final_answers):
        print(f"Q{i+1}: {ans}")

    return final_answers, image, reference_positions

def compare_answers(reference_answers, student_answers):
    """Compare reference and student answers"""
    results = {
        "total_questions": len(reference_answers),
        "correct_answers": 0,
        "incorrect_answers": 0,
        "missing_answers": 0,
        "answer_details": []
    }
    
    for i, (ref_ans, student_ans) in enumerate(zip(reference_answers, student_answers)):
        detail = {
            "question_number": i + 1,
            "reference_answer": ref_ans,
            "student_answer": student_ans,
            "status": "Unknown"
        }
        
        if student_ans == "Missing":
            results["missing_answers"] += 1
            detail["status"] = "Missing"
        elif student_ans == ref_ans:
            results["correct_answers"] += 1
            detail["status"] = "Correct"
        else:
            results["incorrect_answers"] += 1
            detail["status"] = "Incorrect"
        
        results["answer_details"].append(detail)
    
    return results

def process_student_mcq(aligned_image, reference_positions):
    """Process student's MCQ and detect answers"""
    try:
        # Convert to bytes and back to ensure consistent processing
        _, buffer = cv2.imencode('.jpg', aligned_image)
        image_data = np.frombuffer(buffer, np.uint8)
        image, hsv = preprocess_image(image_data)
        
        color_ranges = {
            "blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
            "red1": (np.array([0, 70, 50]), np.array([10, 255, 255])),
            "red2": (np.array([170, 70, 50]), np.array([180, 255, 255])),
        }
    
        mask_combined = np.zeros_like(hsv[:, :, 0])
        for (lower, upper) in color_ranges.values():
            mask = cv2.inRange(hsv, lower, upper)
            mask_combined = cv2.bitwise_or(mask_combined, mask)

        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       
        detected_answers = {}
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                q_index, choice = get_answer_choice(x, y, reference_positions)
                if choice is not None and q_index is not None:
                    detected_answers[q_index] = choice  # map directly by 
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
                    cv2.putText(image, f"Q{q_index+1}: {choice}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        final_answers = [detected_answers.get(i, "Missing") for i in range(len(reference_positions))]

        print("\n✅ Final Detected Answers:")
        for i, ans in enumerate(final_answers, start=1):
            print(f"Q{i}: {ans}")
        return final_answers, image
    except Exception as e:
        print(f"MCQ Processing Error: {e}")
        return ["Missing"] * len(reference_positions)

async def extract_name_roll(image_data):
    """Extract name and roll number using OpenAI"""
    try:
        client = OpenAI(base_url=ENDPOINT, api_key=TOKEN)
        base64_image = encode_image(image_data)
        
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract Name and Roll Number of the student."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
                ]},
            ],
            model=MODEL_NAME
        )
        print("response.choices[0].message.content",response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Name Extraction Error: {e}")
        return "Name: Unknown\nRoll Number: Unknown"

def parse_name_roll(text: str) -> tuple[str, str]:
    """Parse name and roll number from text"""
    name, roll = "Unknown", "Unknown"
    for line in text.split('\n'):
        if "name:" in line.lower():
            name = line.split(":")[-1].strip()
        elif "roll" in line.lower():
            roll = line.split(":")[-1].strip()
    return name, roll

def save_to_mongodb(result_data):
    """Save results to MongoDB"""
    result = collection.insert_one(result_data)
    result_data["_id"] = str(result.inserted_id)
    return result_data

async def process_queue():
    global processing_active, image_queue

    processing_active = True
    while image_queue:
        current_image, aligned_image, reference_paper_id, teacherId, subjectId = image_queue.pop(0)
        try: 
            name_roll_text = await extract_name_roll(current_image)
            name, roll_number = parse_name_roll(name_roll_text)
            
            # Get only reference_positions
            reference_positions = get_teacher_reference_value(teacherId, subjectId, "positions")
            student_answers, aligned_image = process_student_mcq(aligned_image, reference_positions)

            processed_unique_filename = f"{uuid.uuid4()}.jpg"
            processed_filename = f"processed_{processed_unique_filename}"
            processed_file_path = os.path.join(PROCESSED_STUDENT_DIR, processed_filename)
            cv2.imwrite(processed_file_path, aligned_image)



            # Get only reference_answers
            reference_answers = get_teacher_reference_value(teacherId, subjectId, "answers")
            comparison_results = compare_answers(reference_answers, student_answers)
            
            save_to_mongodb({
                "name": name,
                "roll_number": roll_number,
                "reference_answers": reference_answers,
                "student_answers": student_answers,
                "results": comparison_results,
                "processed_filename": processed_filename,
                "processed_file_path": processed_file_path,
                "reference_paper_id": reference_paper_id,
                "teacherId": teacherId,
                "subjectId": subjectId,
                "upload_time": datetime.utcnow(),
            })

        except Exception as e:
            print(f"Failed to process image: {e}")
    
    processing_active = False

@router.post("/upload-reference-mcq")
async def upload_reference_mcq(
    file: UploadFile = File(...),
    teacherId: str = Form(...),
    subjectId: str = Form(...),
    classes: str = Form(...),
    term: str = Form(...),
):
    """Upload reference MCQ image"""
    

    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[-1]
    y_reference_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, y_reference_filename)
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Read image back as bytes
    with open(file_path, "rb") as f:
        image_data = f.read()
    
    # Image processing
    question_y = find_first_question_position(image_data)
    if question_y is None:
        raise HTTPException(status_code=400, detail="Failed to detect first question in the image. Please upload a clearer image.")

    try:
        cropped_image_data = crop_mcq_section(image_data, question_y)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
    # Process the reference image
    reference_image = cv2.imdecode(np.frombuffer(cropped_image_data, np.uint8), cv2.IMREAD_COLOR)
    reference_answers, processed_image, reference_positions = extract_reference_answers(cropped_image_data)

    set_teacher_reference_data(teacherId, subjectId, image=reference_image, answers=reference_answers, positions=reference_positions)

    # save processed image
    file_ext = os.path.splitext(file.filename)[-1]
    processed_unique_filename = f"{uuid.uuid4()}{file_ext}"
    processed_filename = f"processed_{processed_unique_filename}"
    processed_file_path = os.path.join(PROCESSED_REF_DIR, processed_filename)
    cv2.imwrite(processed_file_path, processed_image)



    # Save the processed image to backend
    processed_filename = f"cropped_{y_reference_filename}"
    y_reference_file_path = os.path.join(Y_REF_DIR, processed_filename)
    cv2.imwrite(y_reference_file_path, reference_image)


    mcq_doc = {
        "teacherId": teacherId,
        "subjectId": subjectId,
        "class": classes,
        "term": term,
        "y_reference_filename": y_reference_filename,
        "y_reference_filepath": y_reference_file_path,  # Store the path to the processed image
        "processed_referenceImage_filename": processed_unique_filename,
        "processed_referenceImage_filepath": processed_file_path,
        "upload_time": datetime.utcnow(),
        "num_questions": len(reference_answers),
        "reference_answers": reference_answers,
        'reference_positions':reference_positions,
    }

    result = mcq_collection.insert_one(mcq_doc)
    reference_paper_id = str(result.inserted_id)
    
    return {
        "message": "Reference MCQ uploaded and saved",
        "reference_paper_id": reference_paper_id,
        "file_id": str(result.inserted_id),
        "reference_answers": reference_answers,
        "num_questions": len(reference_answers)
    }

@router.post("/process-student-mcq")
async def process_student_mcq_route(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    reference_paper_id: str = Form(...),
    teacherId: str = Form(...),
    subjectId: str = Form(...),
):
    print("teacher_reference_data",teacher_reference_data)
    """Add student MCQ to processing queue"""
    # Get only reference_image for a teacher
    reference_MCQ_image = get_teacher_reference_value(teacherId, subjectId, "image")
    if reference_MCQ_image is None:
        raise HTTPException(status_code=400, detail="Reference MCQ not uploaded")
    
    image_data = await file.read()
    scanned_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    try:
        aligned_image = align_image(reference_MCQ_image, scanned_image)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image_queue.append((image_data, aligned_image, reference_paper_id, teacherId, subjectId))
    if not processing_active:
        background_tasks.add_task(process_queue)
    
    return {
        "status": "success",
        "position": len(image_queue),
        "estimated_time_sec": len(image_queue) * 15
    }

@router.get("/queue-status")
async def get_queue_status():
    """Get current queue status"""
    return {
        "queue_length": len(image_queue),
        "processing_active": processing_active,
        "estimated_time_sec": len(image_queue) * 15
    }

@router.get("/get-results")
async def get_results():
    """Retrieve all processed results"""
    try:
        results = list(collection.find({}))
        for item in results:
            item["_id"] = str(item["_id"])
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reference-mcqs")
def get_all_reference_mcqs():
    try:
        data = list(mcq_collection.find({}))  # keep _id
        for item in data:
            item["_id"] = str(item["_id"])  # convert ObjectId to string
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/get-student-mcqs/{reference_paper_id}")
async def get_student_mcqs_by_ref(reference_paper_id: str):
    mcqs = list(collection.find({"reference_paper_id": reference_paper_id}))
    for mcq in mcqs:
        mcq["_id"] = str(mcq["_id"])  # Convert ObjectId to string for JSON serialization

    return {"status": "success", "data": mcqs}

@router.get("/get-single_student-mcqs/{student_processed_id}")
async def get_student_mcqs_by_ref(student_processed_id: str):
    try:        
        mcq = collection.find_one({"_id": ObjectId(student_processed_id)})
        if mcq:
            mcq["_id"] = str(mcq["_id"])
            return {"status": "success", "data": mcq}
        else:
            return {"status": "error", "message": "Student MCQ not found", "data": None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/reference-mcq/by-teacher_and_subject/{teacher_id}/{subject_id}")
async def get_reference_mcqs_by_teacher(teacher_id: str, subject_id: str):
    mcq = mcq_collection.find_one({
        "teacherId": teacher_id,
        "subjectId": subject_id
    })

    if not mcq:
        raise HTTPException(status_code=404, detail="No reference MCQs found for this teacher and subject")

    try:
        mcq["id"] = str(mcq.pop("_id"))
        if "upload_time" in mcq and isinstance(mcq["upload_time"], datetime):
            mcq["upload_time"] = mcq["upload_time"].isoformat()

        return {"data": mcq}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reference MCQs: {str(e)}")

@router.get("/student-mcq/by-teacher_and_reference_paper_id/{teacher_id}/{reference_paper_id}")
async def get_reference_mcqs_by_teacher(teacher_id: str, reference_paper_id: str):
    try:
        mcqs_cursor = collection.find({
            "teacherId": teacher_id,
            "reference_paper_id": reference_paper_id
        })

        mcqs_list = []
        for mcq in mcqs_cursor:
            mcq["id"] = str(mcq.pop("_id"))
            mcq["upload_time"] = mcq["upload_time"].isoformat() if isinstance(mcq["upload_time"], datetime) else mcq["upload_time"]
            mcqs_list.append(mcq)

        if not mcqs_list:
            raise HTTPException(status_code=404, detail="No student MCQs found for this teacher and subject")

        return {"data": mcqs_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch student MCQs: {str(e)}")

@router.delete("/delete-reference-mcq/{reference_id}")
async def delete_reference_mcq_by_id(reference_id: str):
    """
    Delete a reference MCQ by MongoDB _id, remove related files, and clear in-memory data.
    """
    try:
        # Validate reference_id
        if not ObjectId.is_valid(reference_id):
            raise HTTPException(status_code=400, detail="Invalid reference ID format")

        # Fetch document by _id
        reference_record = mcq_collection.find_one({"_id": ObjectId(reference_id)})
        if not reference_record:
            raise HTTPException(status_code=404, detail="Reference MCQ not found")

        # === Delete associated image files ===
        try:
            # Full path to cropped reference (Y image)
            y_ref_path = reference_record.get("y_reference_filepath")
            if y_ref_path and os.path.exists(y_ref_path):
                os.remove(y_ref_path)
                print(f"Deleted: {y_ref_path}")

            # Full path to processed reference
            processed_ref_path = reference_record.get("processed_referenceImage_filepath")
            if processed_ref_path and os.path.exists(processed_ref_path):
                os.remove(processed_ref_path)
                print(f"Deleted: {processed_ref_path}")

            # Original uploaded image (optional cleanup)
            uploaded_filename = reference_record.get("y_reference_filename")
            if uploaded_filename:
                uploaded_path = os.path.join(UPLOAD_DIR, uploaded_filename)
                if os.path.exists(uploaded_path):
                    os.remove(uploaded_path)
                    print(f"Deleted: {uploaded_path}")

        except Exception as e:
            print(f"⚠️ Error deleting reference files: {e}")

        # Delete the document from MongoDB
        mcq_collection.delete_one({"_id": ObjectId(reference_id)})

        teacher_id = reference_record.get("teacherId")  # use "teacherId" not "teacher_id"
        subject_id = reference_record.get("subjectId")  # assuming subjectId is available

        key = (teacher_id, subject_id)
        if teacher_id and subject_id and key in teacher_reference_data:
            del teacher_reference_data[key]
            print(f"Cleared in-memory reference data for teacher {teacher_id}, subject {subject_id}")

        return {"message": "Reference MCQ and related data deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error deleting reference MCQ: {str(e)}")

@router.delete("/delete-processed-student-mcq/{_id}")
def delete_processed_student_mcq(_id: str):
    try:
        # Check if record exists
        student_data = collection.find_one({"_id": ObjectId(_id)})
        if not student_data:
            raise HTTPException(status_code=404, detail="Student MCQ not found")

        # Delete the processed image file
        processed_file_path = student_data.get("processed_file_path")
        if processed_file_path and os.path.exists(processed_file_path):
            os.remove(processed_file_path)

        # Remove from MongoDB
        collection.delete_one({"_id": ObjectId(_id)})

        return {"status": "success", "message": "Student MCQ and image deleted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
