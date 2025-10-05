from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional, Dict, Any


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisSessionBase(BaseModel):
    filename: str
    status: str


class AnalysisSessionCreate(BaseModel):
    user_id: int
    filename: str


class AnalysisSession(AnalysisSessionBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ExoplanetPredictionBase(BaseModel):
    planet_data: Dict[str, Any]
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    features: Optional[Dict[str, Any]] = None


class ExoplanetPredictionCreate(ExoplanetPredictionBase):
    session_id: int


class ExoplanetPrediction(ExoplanetPredictionBase):
    id: int
    session_id: int

    class Config:
        from_attributes = True


class UploadedFileBase(BaseModel):
    filename: str
    file_type: str


class UploadedFileCreate(UploadedFileBase):
    user_id: int
    content: str


class UploadedFile(UploadedFileBase):
    id: int
    user_id: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


class MLPredictionRequest(BaseModel):
    planet_data: List[Dict[str, Any]]


class MLPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_version: str


class AnalysisRequest(BaseModel):
    file_id: Optional[int] = None
    raw_data: Optional[List[Dict[str, Any]]] = None


class AnalysisResponse(BaseModel):
    task_id: int
    status: str
