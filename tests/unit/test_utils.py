"""Tests for utility functions."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from imagerouter.utils import (
    validate_image_path,
    get_mime_type,
    validate_prompt,
    infer_output_extension,
    ensure_output_path,
    SUPPORTED_IMAGE_FORMATS,
)
from imagerouter.exceptions import ValidationError


class TestValidateImagePath:
    """Tests for image path validation."""

    def test_valid_image(self, tmp_path):
        """Test validation of valid image file."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake image data")

        result = validate_image_path(str(img_file))
        assert result == img_file.resolve()

    def test_file_not_found(self):
        """Test validation of non-existent file."""
        with pytest.raises(ValidationError) as exc:
            validate_image_path("/nonexistent/image.jpg")
        assert "not found" in str(exc.value)

    def test_unsupported_format(self, tmp_path):
        """Test validation of unsupported format."""
        bad_file = tmp_path / "test.bmp"
        bad_file.write_bytes(b"fake data")

        with pytest.raises(ValidationError) as exc:
            validate_image_path(str(bad_file))
        assert "Unsupported image format" in str(exc.value)

    def test_directory_not_file(self, tmp_path):
        """Test validation rejects directories."""
        with pytest.raises(ValidationError) as exc:
            validate_image_path(str(tmp_path))
        assert "not a file" in str(exc.value)


class TestGetMimeType:
    """Tests for MIME type detection."""

    def test_jpeg(self):
        """Test JPEG MIME type."""
        assert get_mime_type("test.jpg") == "image/jpeg"
        assert get_mime_type("test.jpeg") == "image/jpeg"

    def test_png(self):
        """Test PNG MIME type."""
        assert get_mime_type("test.png") == "image/png"

    def test_webp(self):
        """Test WebP MIME type."""
        assert get_mime_type("test.webp") == "image/webp"

    def test_mp4(self):
        """Test MP4 MIME type."""
        assert get_mime_type("video.mp4") == "video/mp4"

    def test_unknown(self):
        """Test unknown format fallback."""
        result = get_mime_type("file.unknownext123")
        assert result == "application/octet-stream"


class TestValidatePrompt:
    """Tests for prompt validation."""

    def test_valid_prompt(self):
        """Test valid prompt."""
        result = validate_prompt("A beautiful sunset")
        assert result == "A beautiful sunset"

    def test_prompt_with_whitespace(self):
        """Test prompt with leading/trailing whitespace."""
        result = validate_prompt("  A sunset  ")
        assert result == "A sunset"

    def test_empty_prompt(self):
        """Test empty prompt."""
        with pytest.raises(ValidationError) as exc:
            validate_prompt("")
        assert "cannot be empty" in str(exc.value)

    def test_whitespace_only_prompt(self):
        """Test whitespace-only prompt."""
        with pytest.raises(ValidationError) as exc:
            validate_prompt("   ")
        assert "cannot be empty" in str(exc.value)

    def test_prompt_too_long(self):
        """Test prompt exceeding max length."""
        long_prompt = "a" * 10001
        with pytest.raises(ValidationError) as exc:
            validate_prompt(long_prompt)
        assert "too long" in str(exc.value)

    def test_prompt_custom_max_length(self):
        """Test prompt with custom max length."""
        prompt = "a" * 100
        with pytest.raises(ValidationError):
            validate_prompt(prompt, max_length=50)


class TestInferOutputExtension:
    """Tests for output extension inference."""

    def test_video_default(self):
        """Test default video extension."""
        assert infer_output_extension("video") == ".mp4"

    def test_image_default(self):
        """Test default image extension."""
        assert infer_output_extension("image") == ".png"

    def test_preserve_existing_extension(self):
        """Test preserving existing extension."""
        assert infer_output_extension("video", "output.webm") == ".webm"
        assert infer_output_extension("image", "output.jpg") == ".jpg"


class TestEnsureOutputPath:
    """Tests for output path handling."""

    def test_provided_path_with_extension(self):
        """Test provided path with extension."""
        result = ensure_output_path("output.mp4", "video", "test/model")
        assert result == Path("output.mp4")

    def test_provided_path_without_extension(self):
        """Test provided path without extension gets one added."""
        result = ensure_output_path("output", "video", "test/model")
        assert result.suffix == ".mp4"

    def test_auto_generated_path(self):
        """Test auto-generated path."""
        result = ensure_output_path(None, "video", "google/veo-3.1-fast")
        assert result.suffix == ".mp4"
        assert "google_veo-3.1-fast" in result.stem
