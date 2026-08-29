import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.weaviate_db import connect_to_weaviate_with_retry
from app.retrieval_api import register_retrieval_routes
from app.generation_api import register_generation_routes


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """
    FastAPI lifecycle:

    Startup:
        Connect to Weaviate and save the client.

    Shutdown:
        Close the Weaviate HTTP and gRPC connections.
    """

    logger.info("Starting %s", settings.app_name)

    weaviate_client = connect_to_weaviate_with_retry(
        settings
    )

    app.state.weaviate = weaviate_client

    try:
        yield
    finally:
        logger.info("Closing Weaviate connection.")

        weaviate_client.close()

        logger.info("Application shutdown completed.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Ontology-driven GraphRAG API over the Jordanian "
        "Labour Law Knowledge Graph with graph-only legal evidence retrieval."
    ),
    lifespan=lifespan,
)


register_retrieval_routes(app)
register_generation_routes(app)

@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, Any]:
    """
    Basic API information.
    """

    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_environment,
        "message": "Jordan Legal KG API is running.",
    }


@app.get(
    "/health",
    tags=["System"],
)
def health(request: Request) -> JSONResponse:
    """
    Verify both:

    1. FastAPI is running.
    2. The API can communicate with Weaviate.
    """

    try:
        client = request.app.state.weaviate
        weaviate_ready = bool(client.is_ready())

    except Exception as exc:
        logger.exception(
            "Weaviate health check failed."
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "api": "running",
                "weaviate": "unavailable",
                "error": str(exc),
            },
        )

    if not weaviate_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "api": "running",
                "weaviate": "not ready",
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "api": "running",
            "weaviate": "ready",
            "weaviate_http_host": (
                settings.weaviate_http_host
            ),
            "weaviate_http_port": (
                settings.weaviate_http_port
            ),
            "weaviate_grpc_host": (
                settings.weaviate_grpc_host
            ),
            "weaviate_grpc_port": (
                settings.weaviate_grpc_port
            ),
        },
    )