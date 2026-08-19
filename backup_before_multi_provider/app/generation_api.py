from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException

from app.generation_contract import GroundedAnswerResultV1
from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import RetrievalResultV1


router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GroundedAnswerResultV1)
def generate(
    retrieval: RetrievalResultV1,
    include_debug: bool = False,
) -> GroundedAnswerResultV1:
    """
    Generate directly from the complete retrieval.v1 object
    produced by Step 7.

    The request body must be the exact output returned by /retrieve.

    This endpoint does not connect to Weaviate and does not
    rerun retrieval.
    """

    try:
        generator = GroundedAnswerGenerator()

        return generator.generate(
            retrieval,
            include_debug=include_debug,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:  # pragma: no cover - network/model failure
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


def register_generation_routes(app: FastAPI) -> None:
    """Register Stage 8 without changing /retrieve or existing /ask."""

    if not any(
        getattr(route, "path", None) == "/generate"
        for route in app.routes
    ):
        app.include_router(router)