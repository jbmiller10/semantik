"""Examples of using embedding service context managers.

This module demonstrates the proper usage of context managers for
embedding service lifecycle management.
"""

import asyncio
from typing import Any

from shared.embedding import ManagedEmbeddingService, embedding_service_context, temporary_embedding_service


async def basic_context_manager_example() -> None:
    """Basic usage of embedding service context manager."""
    # The service will be automatically cleaned up after use
    async with embedding_service_context() as service:
        # Initialize with a model
        await service.initialize("BAAI/bge-base-en-v1.5")

        # Generate embeddings
        embeddings = await service.embed_texts(
            ["Context managers ensure proper cleanup", "Even if an exception occurs"]
        )

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {service.get_dimension()}")

    # Service is automatically cleaned up here


async def temporary_model_example() -> None:
    """Using a temporary model without affecting global state."""
    # Load a model just for this operation
    async with temporary_embedding_service("sentence-transformers/all-MiniLM-L6-v2") as temp_service:
        # This service is isolated from the global singleton
        await temp_service.embed_texts(["This uses a different model", "Without affecting the main service"])

        model_info = temp_service.get_model_info()
        print(f"Temporary model: {model_info['model_name']}")
        print(f"Dimension: {model_info['dimension']}")

    # Temporary service is cleaned up, GPU memory freed


async def compare_models_example() -> None:
    """Compare embeddings from different models."""
    test_text = "Compare embeddings from different models"

    # Load first model
    async with temporary_embedding_service("BAAI/bge-base-en-v1.5") as bge_service:
        await bge_service.embed_single(test_text)
        bge_dim = bge_service.get_dimension()

    # Load second model (first is already cleaned up)
    async with temporary_embedding_service("sentence-transformers/all-MiniLM-L6-v2") as minilm_service:
        await minilm_service.embed_single(test_text)
        minilm_dim = minilm_service.get_dimension()

    print(f"BGE dimension: {bge_dim}")
    print(f"MiniLM dimension: {minilm_dim}")


async def exception_handling_example() -> None:
    """Context managers handle exceptions properly."""
    try:
        async with embedding_service_context() as service:
            await service.initialize("BAAI/bge-base-en-v1.5")

            # Simulate an error during processing
            await service.embed_texts(["test"])
            raise ValueError("Something went wrong!")

    except ValueError:
        print("Exception caught, but service was still cleaned up")

    # The service cleanup happened despite the exception


async def fastapi_endpoint_example() -> None:
    """Example of using context manager in a FastAPI endpoint."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI()

    class EmbeddingRequest(BaseModel):
        texts: list[str]
        model_name: str = "BAAI/bge-base-en-v1.5"

    @app.post("/embed")
    async def create_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        """Create embeddings with automatic cleanup."""
        try:
            # Use a temporary service for this request
            async with temporary_embedding_service(request.model_name) as service:
                embeddings = await service.embed_texts(request.texts)

                return {
                    "embeddings": embeddings.tolist(),
                    "model": request.model_name,
                    "dimension": service.get_dimension(),
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


async def managed_service_example() -> None:
    """Using ManagedEmbeddingService for more control."""
    # Create a managed service instance
    managed = ManagedEmbeddingService(mock_mode=False)

    # Use it as a context manager
    async with managed as service:
        await service.initialize("BAAI/bge-base-en-v1.5")

        # Can be passed around to other functions
        await process_documents(service, ["doc1", "doc2"])

    # Automatically cleaned up


async def process_documents(service: Any, documents: list[str]) -> Any:
    """Helper function that uses an embedding service."""
    return await service.embed_texts(documents)


async def concurrent_services_example() -> None:
    """Using multiple services concurrently."""

    async def process_with_model(model_name: str, texts: list[str]) -> tuple[str, int]:
        async with temporary_embedding_service(model_name) as service:
            embeddings = await service.embed_texts(texts)
            return model_name, len(embeddings)

    # Process with multiple models concurrently
    tasks = [
        process_with_model("BAAI/bge-base-en-v1.5", ["text1", "text2"]),
        process_with_model("sentence-transformers/all-MiniLM-L6-v2", ["text3", "text4"]),
        process_with_model("BAAI/bge-small-en-v1.5", ["text5", "text6"]),
    ]

    results = await asyncio.gather(*tasks)

    for model_name, num_embeddings in results:
        print(f"{model_name}: Generated {num_embeddings} embeddings")


if __name__ == "__main__":
    # Run examples
    asyncio.run(basic_context_manager_example())
    print("\n" + "=" * 50 + "\n")

    asyncio.run(temporary_model_example())
    print("\n" + "=" * 50 + "\n")

    asyncio.run(exception_handling_example())
    print("\n" + "=" * 50 + "\n")

    asyncio.run(concurrent_services_example())
