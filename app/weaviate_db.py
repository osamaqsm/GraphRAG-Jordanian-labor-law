import logging
import time
from typing import Any

import weaviate
from weaviate.classes.init import Auth
from weaviate.config import AdditionalConfig, Timeout

from app.config import Settings


logger = logging.getLogger(__name__)


def create_weaviate_client(settings: Settings) -> Any:
    """
    Create a Weaviate Python client.

    The HTTP and gRPC hosts use the Docker Compose service
    name 'weaviate', not localhost.

    Inside the API container:
        localhost = the API container itself
        weaviate = the Weaviate container
    """

    return weaviate.connect_to_custom(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=False,

        grpc_host=settings.weaviate_grpc_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=False,

        auth_credentials=Auth.api_key(
            settings.weaviate_api_key
        ),

        additional_config=AdditionalConfig(
            timeout=Timeout(
                init=30,
                query=60,
                insert=120,
            )
        ),
    )


def connect_to_weaviate_with_retry(
    settings: Settings,
) -> Any:
    """
    Try connecting repeatedly because Docker may start the API
    container before Weaviate has completed its initialization.

    If Weaviate is unavailable after all attempts, the API startup
    fails with a clear error.
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        settings.weaviate_connection_attempts + 1,
    ):
        client = None

        try:
            logger.info(
                "Connecting to Weaviate: attempt %s/%s",
                attempt,
                settings.weaviate_connection_attempts,
            )

            client = create_weaviate_client(settings)

            if client.is_ready():
                logger.info(
                    "Connected successfully to Weaviate."
                )
                return client

            last_error = RuntimeError(
                "Weaviate responded but is not ready."
            )

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Weaviate connection attempt %s failed: %s",
                attempt,
                exc,
            )

        if client is not None:
            try:
                client.close()
            except Exception:
                logger.exception(
                    "Failed to close an unsuccessful "
                    "Weaviate client connection."
                )

        if attempt < settings.weaviate_connection_attempts:
            time.sleep(
                settings.weaviate_connection_delay_seconds
            )

    raise RuntimeError(
        "Could not connect to Weaviate after "
        f"{settings.weaviate_connection_attempts} attempts."
    ) from last_error