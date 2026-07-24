import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys

# Add the project root to the path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings

def cleanup_database():
    if not firebase_admin._apps:
        if os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
        else:
            print("Error: Firebase credentials not found.")
            return

    db = firestore.client()
    collections = ['patients', 'cases', 'ai_reports', 'analyses']
    
    for collection_name in collections:
        print(f"Cleaning up collection: {collection_name}...")
        docs = db.collection(collection_name).stream()
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1
        print(f"Deleted {count} documents from {collection_name}.")

if __name__ == "__main__":
    cleanup_database()
