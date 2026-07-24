from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.models.schemas import UserInfo
from firebase_admin import auth


security = HTTPBearer()


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserInfo:

    print("AUTH HEADER RECEIVED = true")

    token = credentials.credentials

    try:
        decoded = auth.verify_id_token(
            token,
            clock_skew_seconds=10
        )

        uid = decoded.get("uid")

        if not uid:
            raise Exception("Firebase UID missing")

        email = decoded.get("email", "")
        name = decoded.get("name", "")

        print("TOKEN VERIFIED = true")
        print(f"FIREBASE UID = {uid}")

        return UserInfo(
            uid=uid,
            email=email,
            display_name=name
        )

    except Exception as e:

        print("TOKEN VERIFIED = false")
        print("AUTH ERROR:", e)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )


def get_current_user(
    user: UserInfo = Depends(verify_token)
):
    return user