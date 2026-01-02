"""Mock API responses for testing."""

MOCK_MODELS_RESPONSE = {
    "data": [
        {
            "id": "google/veo-3.1-fast",
            "name": "Veo 3.1 Fast",
            "provider": "Gemini",
            "output": ["video"],
            "pricing": {
                "type": "calculated",
                "range": {"min": 0.60, "average": 0.90, "max": 1.20},
            },
            "seconds": [4, 6, 8],
            "sizes": ["1280x720", "1920x1080"],
            "supported_params": {"edit": False},
        },
        {
            "id": "kwaivgi/kling-2.1-standard",
            "name": "Kling 2.1 Standard",
            "provider": "Runware",
            "output": ["video"],
            "pricing": {
                "type": "post_generation",
                "range": {"min": 0.18, "average": 0.27, "max": 0.37},
            },
            "seconds": [5, 10],
            "sizes": ["1280x720"],
            "supported_params": {"edit": True},
        },
        {
            "id": "openai/gpt-image-1",
            "name": "GPT Image 1",
            "provider": "OpenAI",
            "output": ["image"],
            "pricing": {
                "type": "calculated",
                "range": {"min": 0.01, "average": 0.15, "max": 0.30},
            },
            "sizes": ["1024x1024", "512x512"],
            "supported_params": {"edit": True},
        },
        {
            "id": "ir/test-video",
            "name": "Test Video",
            "provider": "Test",
            "output": ["video"],
            "pricing": {"type": "fixed", "value": 0.00},
            "seconds": [5],
            "sizes": ["1280x720"],
            "supported_params": {"edit": False},
        },
        {
            "id": "openai/gpt-image-1.5:free",
            "name": "GPT Image 1.5 Free",
            "provider": "OpenAI",
            "output": ["image"],
            "pricing": {"type": "fixed", "value": 0.00},
            "sizes": ["1024x1024"],
            "supported_params": {"edit": True},
        },
    ]
}

MOCK_CREDITS_RESPONSE = {
    "remaining_credits": 50.00,
    "credit_usage": 25.50,
    "total_deposits": 75.50,
}

MOCK_VIDEO_GENERATION_RESPONSE = {
    "created": 1735689600,
    "data": [
        {
            "url": "https://storage.imagerouter.io/videos/test123.mp4",
            "revised_prompt": "A cat playing piano in a jazz club",
        }
    ],
}

MOCK_IMAGE_GENERATION_RESPONSE = {
    "created": 1735689600,
    "data": [
        {
            "url": "https://storage.imagerouter.io/images/test456.png",
            "revised_prompt": "A futuristic cityscape at night with neon lights",
        }
    ],
}

MOCK_AUTH_TEST_RESPONSE = {"status": "ok"}


def get_mock_model_by_id(model_id: str) -> dict | None:
    """Get a mock model by ID from the mock response."""
    for model in MOCK_MODELS_RESPONSE["data"]:
        if model["id"] == model_id:
            return model
    return None
