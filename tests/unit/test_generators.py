"""Tests for video and image generators."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from imagerouter.generators.video import VideoGenerator
from imagerouter.generators.image import ImageGenerator
from imagerouter.exceptions import ValidationError

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from fixtures.mock_responses import (
    MOCK_VIDEO_GENERATION_RESPONSE,
    MOCK_IMAGE_GENERATION_RESPONSE,
)


class TestVideoGenerator:
    """Tests for VideoGenerator class."""

    def test_text_to_video(self):
        """Test text-to-video generation."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_VIDEO_GENERATION_RESPONSE

        generator = VideoGenerator(mock_client)
        result = generator.text_to_video(
            prompt="A cat playing piano",
            model="google/veo-3.1-fast",
            seconds=4,
        )

        assert result == MOCK_VIDEO_GENERATION_RESPONSE
        mock_client.post_json.assert_called_once()

        call_args = mock_client.post_json.call_args
        assert call_args[0][0] == "/v1/openai/videos/generations"
        payload = call_args[0][1]
        assert payload["prompt"] == "A cat playing piano"
        assert payload["model"] == "google/veo-3.1-fast"
        assert payload["seconds"] == 4

    def test_text_to_video_auto_settings(self):
        """Test text-to-video with auto settings."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_VIDEO_GENERATION_RESPONSE

        generator = VideoGenerator(mock_client)
        generator.text_to_video(
            prompt="Test",
            model="test/model",
            seconds="auto",
            size="auto",
        )

        payload = mock_client.post_json.call_args[0][1]
        assert "seconds" not in payload
        assert "size" not in payload

    def test_text_to_video_empty_prompt(self):
        """Test text-to-video with empty prompt."""
        mock_client = MagicMock()
        generator = VideoGenerator(mock_client)

        with pytest.raises(ValidationError) as exc:
            generator.text_to_video(
                prompt="",
                model="test/model",
            )
        assert "cannot be empty" in str(exc.value)

    @patch("imagerouter.generators.video.prepare_multiple_images")
    @patch("imagerouter.generators.video.close_file_handles")
    def test_image_to_video(self, mock_close, mock_prepare):
        """Test image-to-video generation."""
        mock_client = MagicMock()
        mock_client.post_multipart.return_value = MOCK_VIDEO_GENERATION_RESPONSE

        # Mock file preparation
        mock_file = MagicMock()
        mock_prepare.return_value = [("test.jpg", mock_file, "image/jpeg")]

        generator = VideoGenerator(mock_client)
        result = generator.image_to_video(
            image_path="test.jpg",
            prompt="Animate this image",
            model="kwaivgi/kling-2.1-standard",
            seconds=5,
        )

        assert result == MOCK_VIDEO_GENERATION_RESPONSE
        mock_client.post_multipart.assert_called_once()
        mock_close.assert_called_once()

    @patch("imagerouter.generators.video.download_file")
    def test_save_output_url(self, mock_download):
        """Test saving output from URL response."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_VIDEO_GENERATION_RESPONSE
        mock_download.return_value = Path("/tmp/output.mp4")

        generator = VideoGenerator(mock_client)
        generator.text_to_video(
            prompt="Test",
            model="test/model",
            output_path="/tmp/output.mp4",
        )

        mock_download.assert_called_once()


class TestImageGenerator:
    """Tests for ImageGenerator class."""

    def test_text_to_image(self):
        """Test text-to-image generation."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_IMAGE_GENERATION_RESPONSE

        generator = ImageGenerator(mock_client)
        result = generator.text_to_image(
            prompt="A futuristic city",
            model="openai/gpt-image-1",
            quality="high",
            size="1024x1024",
        )

        assert result == MOCK_IMAGE_GENERATION_RESPONSE
        mock_client.post_json.assert_called_once()

        call_args = mock_client.post_json.call_args
        assert call_args[0][0] == "/v1/openai/images/generations"
        payload = call_args[0][1]
        assert payload["prompt"] == "A futuristic city"
        assert payload["model"] == "openai/gpt-image-1"
        assert payload["quality"] == "high"
        assert payload["size"] == "1024x1024"

    def test_text_to_image_auto_settings(self):
        """Test text-to-image with auto settings."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_IMAGE_GENERATION_RESPONSE

        generator = ImageGenerator(mock_client)
        generator.text_to_image(
            prompt="Test",
            model="test/model",
            quality="auto",
            size="auto",
        )

        payload = mock_client.post_json.call_args[0][1]
        assert "quality" not in payload
        assert "size" not in payload

    @patch("imagerouter.generators.image.prepare_multiple_images")
    @patch("imagerouter.generators.image.close_file_handles")
    def test_image_to_image(self, mock_close, mock_prepare):
        """Test image-to-image generation."""
        mock_client = MagicMock()
        mock_client.post_multipart.return_value = MOCK_IMAGE_GENERATION_RESPONSE

        mock_file = MagicMock()
        mock_prepare.return_value = [("test.jpg", mock_file, "image/jpeg")]

        generator = ImageGenerator(mock_client)
        result = generator.image_to_image(
            image_path="test.jpg",
            prompt="Add a sunset",
            model="openai/gpt-image-1",
        )

        assert result == MOCK_IMAGE_GENERATION_RESPONSE
        mock_client.post_multipart.assert_called_once()

        call_args = mock_client.post_multipart.call_args
        assert call_args[0][0] == "/v1/openai/images/edits"

    @patch("imagerouter.generators.image.prepare_multiple_images")
    @patch("imagerouter.generators.image.close_file_handles")
    def test_image_to_image_with_mask(self, mock_close, mock_prepare):
        """Test image-to-image with mask."""
        mock_client = MagicMock()
        mock_client.post_multipart.return_value = MOCK_IMAGE_GENERATION_RESPONSE

        mock_file = MagicMock()
        mock_prepare.return_value = [("test.jpg", mock_file, "image/jpeg")]

        generator = ImageGenerator(mock_client)
        generator.image_to_image(
            image_path="test.jpg",
            prompt="Edit the sky",
            model="openai/gpt-image-1",
            mask_path="mask.png",
        )

        # Should prepare both image and mask
        assert mock_prepare.call_count == 2

    def test_empty_prompt(self):
        """Test generation with empty prompt."""
        mock_client = MagicMock()
        generator = ImageGenerator(mock_client)

        with pytest.raises(ValidationError):
            generator.text_to_image(
                prompt="   ",
                model="test/model",
            )

    @patch("imagerouter.generators.image.download_file")
    def test_save_output(self, mock_download):
        """Test saving output to file."""
        mock_client = MagicMock()
        mock_client.post_json.return_value = MOCK_IMAGE_GENERATION_RESPONSE
        mock_download.return_value = Path("/tmp/output.png")

        generator = ImageGenerator(mock_client)
        generator.text_to_image(
            prompt="Test",
            model="test/model",
            output_path="/tmp/output.png",
        )

        mock_download.assert_called_once_with(
            MOCK_IMAGE_GENERATION_RESPONSE["data"][0]["url"],
            "/tmp/output.png",
        )
