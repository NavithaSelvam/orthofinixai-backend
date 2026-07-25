import os
import uuid
import json
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage
from app.core.config import settings

def init_firebase():
    if not firebase_admin._apps:
        try:
            cred = None
            firebase_path = os.getenv("FIREBASE_CREDENTIALS_PATH", settings.FIREBASE_CREDENTIALS_PATH)
            print(f"DEBUG: firebase_path={firebase_path}, exists={os.path.exists(firebase_path) if firebase_path else 'N/A'}")
            if firebase_path and os.path.exists(firebase_path):
                cred = credentials.Certificate(firebase_path)
            else:
                cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
                if cred_json:
                    try:
                        cred_dict = json.loads(cred_json)
                        cred = credentials.Certificate(cred_dict)
                    except json.JSONDecodeError as e:
                        print(f"Warning: FIREBASE_CREDENTIALS_JSON is malformed, ignoring it: {e}")

            if cred is None:
                print("Warning: No valid Firebase credentials found. Attempting default auth.")

            bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET", "orthofinixai.appspot.com")

            if cred:
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
            else:
                firebase_admin.initialize_app(options={'storageBucket': bucket_name})
        except Exception as e:
            print(f"Firebase initialization failed: {e}")

def get_db():
    return firestore.client()

def get_auth():
    return auth

def save_analysis_record(data: dict, user_id: str, provided_case_id: str = "") -> dict:
    db = get_db()
    record_id = provided_case_id if provided_case_id else str(uuid.uuid4())
    data["id"] = record_id
    data["user_id"] = user_id
    data["created_at"] = datetime.now(timezone.utc).isoformat()

    safe_data = json.loads(json.dumps(data))

    db.collection("users").document(user_id).collection("cases").document(record_id).set(safe_data)

    return safe_data

def get_user_analysis_history(user_id: str) -> list:
    db = get_db()
    docs = db.collection("users").document(user_id).collection("cases").order_by("created_at", direction=firestore.Query.DESCENDING).limit(50).stream()
    return [doc.to_dict() for doc in docs]

def get_analysis_by_id(record_id: str) -> dict:
    db = get_db()
    docs = db.collection_group("cases").where("id", "==", record_id).limit(1).stream()
    for doc in docs:
        return doc.to_dict()
    return None

def upload_image_to_storage(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    try:
        bucket = storage.bucket()
        unique_filename = f"uploads/{uuid.uuid4()}_{filename}"
        blob = bucket.blob(unique_filename)
        blob.upload_from_string(file_bytes, content_type=content_type)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Firebase Storage Write Error: {e}")
        raise ValueError(f"Failed to write image to storage: {e}")
