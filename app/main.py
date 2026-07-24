from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.firebase import init_firebase
from app.db.sqlalchemy_db import init_sqlalchemy

from app.api.routes import (
    auth,
    patients,
    cases,
    ai,
    analysis,
    summit_auth,
    summit_analysis,
)

# Initialize services
init_firebase()
init_sqlalchemy()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Orthodontic AI Analysis Backend API",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads folder if missing
import os
os.makedirs("uploads", exist_ok=True)

# Static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --------------------------
# Include Routers
# --------------------------

app.include_router(auth.router, tags=["Auth"])
app.include_router(patients.router, tags=["Patients"])
app.include_router(cases.router, tags=["Cases"])
app.include_router(ai.router, tags=["AI"])
app.include_router(analysis.router, tags=["Analysis"])

# Summit Routes
app.include_router(summit_auth.router, tags=["Summit Auth"])
app.include_router(
    summit_analysis.router,
    prefix="/analysis",
    tags=["Summit Analysis"]
)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the OrthofinixAi Backend",
        "status": "active",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )