"""Convenience launcher: `python run.py` (add --reload during development)."""
import argparse

import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    a = p.parse_args()
    uvicorn.run("backend.main:app", host=a.host, port=a.port, reload=a.reload)
