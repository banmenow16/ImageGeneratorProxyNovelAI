"""Backend-neutral video generation (image-to-video)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class VideoRequest:
	character: str
	source_image: Path
	prompt: str = ""
	negative_prompt: str = ""
	duration_seconds: int = 5
	width: int = 832
	height: int = 1216
	source_digest: str = ""


@dataclass
class GeneratedVideo:
	name: str
	data: bytes
	content_type: str = "video/mp4"


class VideoGenerator(Protocol):
	"""Animate a source image into a video.

	Implementations return None when the backend stays busy/fails after
	retries; raise on hard errors. NSFW tolerance is a backend property —
	only NSFW-capable backends should be registered.
	"""

	backend_name: str
	display_name: str

	async def generate_video(self, request: VideoRequest) -> Optional[GeneratedVideo]: ...
