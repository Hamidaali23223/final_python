import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
class ENV:
    # Database
    MONGO_URI: str = os.getenv("MONGO_URI")
    API_URI: str = os.getenv("API_URI")
    # API Config
    TOKEN: str = os.getenv("TOKEN2")
    ENDPOINT: str = os.getenv("ENDPOINT")
    MODEL_NAME: str = os.getenv("MODEL_NAME")
    
    # OCR Config
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH")
    
    # Directory Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR")
    Y_REF_DIR: str = os.getenv("Y_REF_DIR")
    PROCESSED_REF_DIR: str = os.getenv("PROCESSED_REF_DIR")
    PROCESSED_STUDENT_DIR: str = os.getenv("PROCESSED_STUDENT_DIR")
    
    MCQ_GRID = {
        "A": 0.1,
        "B": 0.33, 
        "C": 0.54,
        "D": 0.75
    }

env = ENV()