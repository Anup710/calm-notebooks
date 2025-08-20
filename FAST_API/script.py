# DEBUGGING FASTAPI APPLICATION
# This guide will help you identify and fix common issues

# ============================================================================
# STEP 1: INSTALL REQUIRED DEPENDENCIES
# ============================================================================
"""
First, make sure you have all dependencies installed:

pip install fastapi[all]
# OR individually:
pip install fastapi uvicorn[standard] python-multipart email-validator

The [all] includes uvicorn server and other optional dependencies
"""

# ============================================================================
# STEP 2: FIXED VERSION OF YOUR CODE
# ============================================================================

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn
import os

# ============================================================================
# FIX 1: PROPER LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("🚀 FastAPI app is starting up...")
    print("✅ Creating static directory...")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    yield  # App runs while inside this context

    # Shutdown code
    print("🛑 FastAPI app is shutting down...")

# ============================================================================
# APP INITIALIZATION & CONFIGURATION
# ============================================================================

app = FastAPI(
    title="AI Model API",
    description="Complete FastAPI example showing all key components",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
    lifespan=lifespan
)

# ============================================================================
# CORS HANDLING
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # FIX 2: Allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ============================================================================
# STATIC FILE SERVING
# ============================================================================

# Mount static files directory (created in lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    # FIX 3: Make EmailStr optional or install email-validator
    email: str  # Changed from EmailStr to str for simplicity
    age: int = Field(..., ge=0, le=150)
    preferences: Optional[List[str]] = []

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: int
    is_active: bool = True

class MLPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_name: str = "default"
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

class MLPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str

# ============================================================================
# AUTHENTICATION & SECURITY
# ============================================================================

security = HTTPBearer(auto_error=False)  # FIX 4: Make auth optional for testing

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Simple token verification - made optional for testing"""
    if credentials is None:
        # For testing, allow requests without auth
        print("⚠️  No authentication provided - allowing for testing")
        return "test-mode"
    
    token = credentials.credentials
    if token != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

# ============================================================================
# GLOBAL STORAGE
# ============================================================================

users_db = []
user_counter = 1

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - basic health check"""
    return {
        "message": "🎉 AI API is running successfully!",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "users": "/users",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Model API",
        "version": "1.0.0",
        "users_count": len(users_db)
    }

# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    """Create a new user with automatic validation"""
    global user_counter
    
    # Check if email already exists
    if any(u["email"] == user.email for u in users_db):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    new_user = {
        "id": user_counter,
        "name": user.name,
        "email": user.email,
        "age": user.age,
        "is_active": True
    }
    users_db.append(new_user)
    user_counter += 1
    
    return new_user

@app.get("/users", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 10):
    """Get users with pagination"""
    return users_db[skip:skip + limit]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """Get specific user by ID"""
    user = next((u for u in users_db if u["id"] == user_id), None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user

# ============================================================================
# ML MODEL ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=MLPredictionResponse)
async def predict(
    request: MLPredictionRequest,
    token: str = Depends(verify_token)  # Optional for testing
):
    """ML prediction endpoint"""
    
    # Simulate ML model inference
    prediction_confidence = 0.85
    
    if prediction_confidence < request.confidence_threshold:
        prediction = "uncertain"
    else:
        prediction = "positive" if len(request.text) % 2 == 0 else "negative"
    
    return MLPredictionResponse(
        prediction=prediction,
        confidence=prediction_confidence,
        model_used=request.model_name
    )

@app.post("/predict/batch")
async def batch_predict(
    requests: List[MLPredictionRequest],
    token: str = Depends(verify_token)
):
    """Batch prediction endpoint"""
    results = []
    for req in requests:
        result = {
            "text": req.text,
            "prediction": "positive" if len(req.text) % 2 == 0 else "negative",
            "confidence": 0.80
        }
        results.append(result)
    
    return {"predictions": results, "processed_count": len(results)}

# ============================================================================
# FILE UPLOAD HANDLING
# ============================================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads"""
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "text/plain", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file.content_type} not allowed"
        )
    
    # Save file
    file_path = f"static/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
        "url": f"/static/{file.filename}"
    }

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Custom error handler for ValueError"""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )

# ============================================================================
# RUNNING THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    print(f"🚀 Starting {app.title}")
    print("📖 Visit http://localhost:8000/docs for API documentation")
    print("🏠 Visit http://localhost:8000 for basic info")
    
    uvicorn.run(
        "script:app",  # FIX 5: Change this to match your filename
        host="127.0.0.1",  # FIX 6: Use localhost instead of 0.0.0.0
        port=8000,
        reload=True,
        workers=1
    )

# ============================================================================
# DEBUGGING COMMANDS
# ============================================================================

"""
STEP-BY-STEP DEBUGGING:

1. Save this code as script.py

2. Install dependencies:
   pip install fastapi[all]

3. Run the application:
   python main.py
   
   OR
   
   uvicorn main:app --reload

4. Test endpoints:
   - Basic test: http://localhost:8000
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

5. Common errors and fixes:
   - ModuleNotFoundError: Install missing packages
   - Port already in use: Change port to 8001
   - Permission denied: Use 127.0.0.1 instead of 0.0.0.0
   - Email validation error: Install email-validator or use str instead of EmailStr

TESTING THE API:

1. Go to http://localhost:8000/docs
2. Try the GET / endpoint
3. Try creating a user with POST /users
4. Try the prediction endpoint POST /predict

Sample user creation:
{
  "name": "John Doe",
  "email": "john@example.com", 
  "age": 30,
  "preferences": ["AI", "ML"]
}

Sample prediction:
{
  "text": "This is a test message",
  "model_name": "default",
  "confidence_threshold": 0.5
}
"""