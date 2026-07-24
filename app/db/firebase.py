import os
import uuid
import json
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, auth
from app.core.config import settings

UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads"))
os.makedirs(UPLOADS_DIR, exist_ok=True)

def init_firebase():
    if not firebase_admin._apps:
        try:
            if os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
                cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred)
            else:
                # Fallback for environments where default credentials are used (e.g. Google Cloud)
                # Or just print a warning if missing
                print(f"Warning: {settings.FIREBASE_CREDENTIALS_PATH} not found. Attempting default auth.")
                firebase_admin.initialize_app()
        except Exception as e:
            print(f"Firebase initialization failed: {e}")

def get_db():
    return firestore.client()

def get_auth():
    return auth

def save_analysis_record(data: dict, user_id: str) -> dict:
    db = get_db()
    record_id = str(uuid.uuid4())
    data["id"] = record_id
    data["user_id"] = user_id
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    
    # ensure no None values that Firestore rejects
    safe_data = json.loads(json.dumps(data))
    
    # Write to users/{user_id}/cases/{record_id} to match Android app's path, 
    # OR write to analyzed_cases to match old backend format. Let's write to both or just users.
    # The requirement says: "Storage: Firestore + Room local database cache must remain synchronized."
    db.collection("users").document(user_id).collection("cases").document(record_id).set(safe_data)
    
    return safe_data

def get_user_analysis_history(user_id: str) -> list:
    db = get_db()
    docs = db.collection("users").document(user_id).collection("cases").order_by("created_at", direction=firestore.Query.DESCENDING).limit(50).stream()
    return [doc.to_dict() for doc in docs]

def get_analysis_by_id(record_id: str) -> dict:
    db = get_db()
    # To find it we might need user_id, but the old implementation didn't take user_id. 
    # Let's query across all cases if we don't know the user, but firestore requires group collection query.
    docs = db.collection_group("cases").where("id", "==", record_id).limit(1).stream()
    for doc in docs:
        return doc.to_dict()
    return None

def upload_image_to_storage(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Save image file to local uploads directory and return dynamic retrieval URL.
    """
    try:
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOADS_DIR, unique_filename)
        with open(filepath, "wb") as f:
            f.write(file_bytes)
        
        # Return standard URL mapped to static endpoint (using configurable domain)
        base_url = os.environ.get("BASE_URL", "http://10.54.37.107:8000").rstrip("/")
        return f"{base_url}/uploads/{unique_filename}"
    except Exception as e:
        print(f"Local Storage Write Error: {e}")
        raise ValueError("Failed to write image to storage.")
