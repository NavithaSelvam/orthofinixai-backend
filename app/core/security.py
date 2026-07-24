# This file previously contained local JWT authentication logic using python-jose.
# It has been deprecated as the backend has completely migrated to Firebase Authentication.
# Local JWT generation, SECRET_KEY, and hash validation have been removed.

def hash_password(password: str) -> str:
    raise NotImplementedError("Local password hashing is deprecated. Use Firebase Auth.")

def verify_password(plain: str, hashed: str) -> bool:
    raise NotImplementedError("Local password verification is deprecated. Use Firebase Auth.")

def create_access_token(subject: str, email: str, name: str) -> str:
    raise NotImplementedError("Local JWT generation is deprecated. Tokens are managed by Firebase Auth.")

def decode_token(token: str) -> dict:
    raise NotImplementedError("Local JWT decoding is deprecated. Use firebase_admin.auth.verify_id_token.")
