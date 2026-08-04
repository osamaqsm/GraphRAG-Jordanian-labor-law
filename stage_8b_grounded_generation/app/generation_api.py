from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException

from app.generation_contract import (
    GenerationRequestV1,
    GroundedAnswerResultV1,
)
from app.grounded_answer_generator import GroundedAnswerGenerator


router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GroundedAnswerResultV1)
def generate(payload: GenerationRequestV1) -> GroundedAnswerResultV1:
    """
    Generate from an already completed retrieval.v1 object.

    This endpoint does not connect to Weaviate and does not rerun retrieval.
    """

    try:
        generator = GroundedAnswerGenerator()
        return generator.generate(
            payload.retrieval,
            include_debug=payload.include_debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/model failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def register_generation_routes(app: FastAPI) -> None:
    """Register Stage 8-B without changing /retrieve or existing /ask."""

    if not any(getattr(route, "path", None) == "/generate" for route in app.routes):
        app.include_router(router)
