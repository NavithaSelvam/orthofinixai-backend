from fastapi import APIRouter, Depends
from app.models.schemas import UserInfo
from app.api.dependencies import get_current_user

router = APIRouter()

# The /login and /register endpoints have been removed as authentication is now
# completely handled by Firebase Authentication on the client side.

@router.get("/me", response_model=UserInfo)
def get_me(current_user: UserInfo = Depends(get_current_user)):
    """
    Get the currently authenticated Firebase user profile based on Bearer token.
    """
    return current_user
