"""Image generation backends. ImageGenerator is the backend-neutral
interface; add new backends (Civitai, local model servers) as sibling
modules and expose them here."""

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator
from imagine_generator_proxy.generation.video_base import GeneratedVideo, VideoGenerator, VideoRequest
from imagine_generator_proxy.generation.video_civitai import Ltx2Video, Wan22Video, Wan26Video
from imagine_generator_proxy.generation.civitai import (
	CivitaiGenerator,
	IllustriousGenerator,
	NoobAIGenerator,
	PonyV6Generator,
)
from imagine_generator_proxy.generation.novelai import ImageGenerationType, NovelAIGenerator, build_preset

__all__ = [
	"GeneratedImage", "ImageGenerator", "NovelAIGenerator", "build_preset",
	"ImageGenerationType", "CivitaiGenerator", "IllustriousGenerator",
	"PonyV6Generator", "NoobAIGenerator",
	"GeneratedVideo", "VideoGenerator", "VideoRequest", "Wan22Video", "Wan26Video", "Ltx2Video",
]
