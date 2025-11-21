"""
Example usage of the service layer DTOs.
This demonstrates how the service layer can use these DTOs to encapsulate business logic.
"""

from webui.services.dtos.chunking_dtos import ServiceChunkPreview, ServicePreviewResponse, ServiceStrategyInfo


def example_usage() -> None:
    """Example of how the service layer would use these DTOs."""

    # Example 1: Creating a strategy info DTO
    strategy_info = ServiceStrategyInfo(
        id="fixed_size",
        name="Fixed Size Chunking",
        description="Splits text into chunks of a fixed size",
        best_for=["txt", "log", "csv"],
        pros=["Predictable chunk sizes", "Fast processing"],
        cons=["May split sentences", "No semantic boundaries"],
        default_config={"chunk_size": 512, "chunk_overlap": 50},
    )

    # Convert to API model (this would happen in the router)
    api_strategy = strategy_info.to_api_model()
    print(f"Strategy API model: {api_strategy.id}")

    # Example 2: Creating chunk previews with automatic token count estimation
    chunk1 = ServiceChunkPreview(
        index=0,
        content="This is the first chunk of text.",
        # char_count and token_count will be calculated automatically
    )

    chunk2 = ServiceChunkPreview(
        index=1,
        text="Second chunk using 'text' field instead of 'content'",
        token_count=10,  # Explicitly provided
    )

    # Example 3: Creating a preview response
    preview = ServicePreviewResponse(
        preview_id="preview_123",
        strategy="fixed_size",
        config={"chunk_size": 512, "chunk_overlap": 50},
        chunks=[chunk1, chunk2],
        total_chunks=2,
        processing_time_ms=150,
    )

    # Convert to API model (this would happen in the router)
    api_preview = preview.to_api_model()
    print(f"Preview has {len(api_preview.chunks)} chunks")
    print(f"First chunk token count: {api_preview.chunks[0].token_count}")
    print(f"First chunk char count: {api_preview.chunks[0].char_count}")

    # Example 4: Handling dict chunks from service layer
    dict_chunk = {
        "index": 2,
        "content": "This is a dict chunk from the service",
        "quality_score": 0.9,
    }

    preview_with_dict = ServicePreviewResponse(
        preview_id="preview_456",
        strategy="semantic",
        config={"strategy": "semantic", "similarity_threshold": 0.7},
        chunks=[chunk1, dict_chunk],  # Mix of ServiceChunkPreview and dict
        total_chunks=2,
        processing_time_ms=200,
    )

    # Conversion handles both types seamlessly
    api_preview2 = preview_with_dict.to_api_model()
    print(f"Mixed preview has {len(api_preview2.chunks)} chunks")


if __name__ == "__main__":
    example_usage()
