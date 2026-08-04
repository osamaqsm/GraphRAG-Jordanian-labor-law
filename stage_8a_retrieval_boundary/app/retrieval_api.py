from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException

from app.retrieval_contract import RetrievalRequestV1, RetrievalResultV1
from app.retrieval_pipeline import RetrievalOnlyPipeline


router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalResultV1)
def retrieve(payload: RetrievalRequestV1) -> RetrievalResultV1:
    """Run routing and retrieval only. This endpoint never generates an answer."""

    try:
        with RetrievalOnlyPipeline() as pipeline:
            return pipeline.retrieve(
                payload.question,
                include_debug=payload.include_debug,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/model failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def register_retrieval_routes(app: FastAPI) -> None:
    """Register Stage 8-A without changing existing /ask behavior."""

    if not any(getattr(route, "path", None) == "/retrieve" for route in app.routes):
        app.include_router(router)
