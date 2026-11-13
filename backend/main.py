import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DEFAULT_SECRET, settings
from backend.database import init_db
from backend.metrics_store import ensure_metrics
from backend.routes import auth_routes, detect_routes, history_routes, system_routes
from models.registry import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    registry.load_fast()
    m = registry.text.baseline.bootstrap_metrics
    if m:
        ensure_metrics(registry.text.baseline.name, "text", m)
    if settings.SECRET_KEY == DEFAULT_SECRET:
        log.warning("Using the default SECRET_KEY - set SECRET_KEY in .env before sharing this deployment")
    registry.warmup_async()
    log.info("Startup complete. Open http://127.0.0.1:8000")
    yield


app = FastAPI(title=settings.APP_NAME, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth_routes, detect_routes, history_routes, system_routes):
    app.include_router(r.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(settings.FRONTEND_DIR / "img" / "logo.svg", media_type="image/svg+xml")


app.mount("/", StaticFiles(directory=settings.FRONTEND_DIR, html=True), name="frontend")
