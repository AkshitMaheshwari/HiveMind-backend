"""
Image Generation Tool
=====================
Generates images using OpenAI DALL-E 3 (key configured in .env).
Registered with tag "design" in registry_bootstrap.py.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def generate_image(prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
    """
    Generate an image using OpenAI DALL-E 3.

    Parameters:
        prompt: A detailed description of the image to generate.
        size: Image size - "1024x1024", "1792x1024", or "1024x1792".
        quality: "standard" or "hd".

    Returns:
        The URL of the generated image, or an error message string.

    Tags: ["design"]
    """
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
        if not api_key:
            return "Image generation unavailable: OPENAI_API_KEY not set."

        base_url = os.getenv("AZURE_OPENAI_BASE_URL", "https://models.inference.ai.azure.com")
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        url = response.data[0].url
        logger.info("image_generation: generated image url=%s", url[:80])
        return url

    except Exception as exc:
        logger.error("image_generation: failed: %s", exc)
        return f"Image generation failed: {exc}"