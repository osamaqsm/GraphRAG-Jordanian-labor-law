from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException, Request

from app.retrieval_contract import RetrievalRequestV2, RetrievalResultV2
from app.retrieval_pipeline import RetrievalOnlyPipeline


router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalResultV2)
def retrieve(payload: RetrievalRequestV2, request: Request) -> RetrievalResultV2:
    """Run planner -> concept linking -> KG traversal -> evidence reranking."""
    try:
        with RetrievalOnlyPipeline(
            weaviate_client=request.app.state.weaviate,
        ) as pipeline:
            return pipeline.retrieve(
                payload.question,
                include_debug=payload.include_debug,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # provider/network/model failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def register_retrieval_routes(app: FastAPI) -> None:
    if not any(getattr(route, "path", None) == "/retrieve" for route in app.routes):
        app.include_router(router)
