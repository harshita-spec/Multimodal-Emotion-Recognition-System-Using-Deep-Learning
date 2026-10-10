import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.router import api_router

app = FastAPI(
    title="Multimodal Emotion Recognition System",
    description="Deep Learning API for Facial, Audio, and Text Emotion Recognition with Attention Fusion & XAI",
    version="1.0.0"
)

# CORS Middleware enabling React frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/")
def root():
    return {
        "message": "Multimodal Emotion Recognition API is active.",
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
