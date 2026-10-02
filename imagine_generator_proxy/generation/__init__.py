"""Image generation backends. ImageGenerator is the backend-neutral
interface; add new backends (Civitai, local model servers) as sibling
modules and expose them here."""

from imagine_generator_proxy.generation.base import GeneratedImage, ImageGenerator
from imagine_generator_proxy.generation.novelai import ImageGenerationType, NovelAIGenerator, build_preset

__all__ = ["GeneratedImage", "ImageGenerator", "NovelAIGenerator", "build_preset", "ImageGenerationType"]
