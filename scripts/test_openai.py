from __future__ import annotations

import json
import sys

from app.config import get_settings
from app.openai_service import OpenAIService


def main() -> int:
    settings = get_settings()

    service: OpenAIService | None = None

    try:
        service = OpenAIService(settings)

        result = service.embed_texts(
            [
                (
                    "تأخر صاحب العمل في دفع "
                    "أجر العامل لمدة عشرة أيام"
                )
            ]
        )

        output = {
            "status": "success",
            "model": (
                settings.openai_embedding_model
            ),
            "number_of_vectors": len(
                result.vectors
            ),
            "dimensions": result.dimensions,
            "input_tokens": result.input_tokens,
            "first_five_values": (
                result.vectors[0][:5]
            ),
        }

        print(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        return 1

    finally:
        if service is not None:
            service.close()


if __name__ == "__main__":
    sys.exit(main())