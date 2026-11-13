from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: EmailStr
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class TextDetectRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    title: str | None = Field(default=None, max_length=500)


class DetectionResult(BaseModel):
    id: int
    prediction: Literal["REAL", "FAKE"]
    confidence_score: float
    modality: Literal["text", "image", "video"]
    inference_time_ms: float
    models_used: list[str]
    details: dict[str, Any]
    created_at: datetime


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    modality: str
    input_summary: str
    prediction: str
    confidence_score: float
    inference_time_ms: float
    models_used: list[str]
    created_at: datetime


class HistoryPage(BaseModel):
    total: int
    items: list[HistoryItem]


class HistoryStats(BaseModel):
    total: int
    fake: int
    real: int
    by_modality: dict[str, int]


class MetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    modality: str
    dataset: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    num_samples: int
    evaluated_at: datetime
