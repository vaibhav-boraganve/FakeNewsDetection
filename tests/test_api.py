"""End-to-end API tests. Run with: pytest -q"""
import os
import tempfile

os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp()}/test.db")
os.environ.setdefault("ENABLE_TRANSFORMER", "false")
os.environ.setdefault("ENABLE_PRETRAINED_DOWNLOAD", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/api/auth/register", json={"full_name": "Test User", "email": "t@example.com", "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _check_result(body, modality):
    assert body["prediction"] in {"REAL", "FAKE"}
    assert 50.0 <= body["confidence_score"] <= 100.0
    assert body["modality"] == modality
    assert body["inference_time_ms"] >= 0


def test_auth_flow(client, auth):
    assert client.post("/api/auth/login", json={"email": "t@example.com", "password": "wrong-pass"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "T@example.com", "password": "password123"}).status_code == 200
    assert client.get("/api/auth/me", headers=auth).json()["email"] == "t@example.com"
    assert client.get("/api/history").status_code == 401


def test_text_detection(client, auth):
    fake = "SHOCKING!!! Doctors don't want you to know this miracle cure. Share before it gets deleted! The mainstream media is hiding the truth!!!"
    real = ("The central bank kept its benchmark interest rate unchanged at 6.5 percent on Thursday, the governor said "
            "in a statement. According to officials, inflation is expected to ease next quarter.")
    rf = client.post("/api/detect/text", json={"text": fake}, headers=auth).json()
    rr = client.post("/api/detect/text", json={"text": real}, headers=auth).json()
    _check_result(rf, "text")
    _check_result(rr, "text")
    assert rf["prediction"] == "FAKE"
    assert rr["prediction"] == "REAL"


def test_image_detection(client, auth):
    img = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/detect/image", files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")}, headers=auth)
    assert r.status_code == 200, r.text
    _check_result(r.json(), "image")

    bad = client.post("/api/detect/image", files={"file": ("x.txt", b"hello", "text/plain")}, headers=auth)
    assert bad.status_code == 415


def test_video_detection(client, auth, tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(30):
        frame = np.full((120, 160, 3), 40, np.uint8)
        cv2.circle(frame, (20 + i * 4, 60), 15, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        r = client.post("/api/detect/video", files={"file": ("clip.mp4", f, "video/mp4")}, headers=auth)
    assert r.status_code == 200, r.text
    _check_result(r.json(), "video")


def test_history_and_metrics(client, auth):
    page = client.get("/api/history", headers=auth).json()
    assert page["total"] >= 4
    stats = client.get("/api/history/stats", headers=auth).json()
    assert stats["total"] == page["total"]
    item_id = page["items"][0]["id"]
    assert client.delete(f"/api/history/{item_id}", headers=auth).status_code == 204
    metrics = client.get("/api/metrics", headers=auth).json()
    assert any(m["modality"] == "text" for m in metrics)
