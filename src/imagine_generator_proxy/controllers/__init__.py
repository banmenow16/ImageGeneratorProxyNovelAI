"""HTTP controllers: one class per concern, wired through register_controllers.

Endpoint groups:
- GenerationController — POST /api/generate (direct, no review chain)
- VisionController    — POST /api/vision/analyze (vision model on stored images)
- CharacterController — GET /api/characters[/{name}] (Mirror's verified output)
- AdviceToolsController — /api/advice/* iteration tools
"""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from imagine_generator_proxy.generation.base import Backend
from imagine_generator_proxy.controllers.advice_tools import AdviceToolsController
from imagine_generator_proxy.controllers.base import Controller
from imagine_generator_proxy.controllers.character_controller import CharacterController
from imagine_generator_proxy.controllers.generation_controller import GenerationController
from imagine_generator_proxy.controllers.vision_controller import VisionController

__all__ = [
	"Controller",
	"AdviceToolsController",
	"CharacterController",
	"GenerationController",
	"VisionController",
	"register_controllers",
]


def register_controllers(app: web.Application) -> None:
	"""Construct every controller with app state and register its routes."""
	config: dict = app.get("config") or {}
	backends: dict[str, Backend] = app.get("backends") or {}
	data_dir = Path(app["videos_dir"]).parent
	controllers: list[Controller] = [
		GenerationController(backends,
			str(app.get("generation_backend", "novelai")), data_dir),
		VisionController(config, data_dir),
		CharacterController(Path(app["db_path"])),
		AdviceToolsController(config, backends,
			str(app.get("generation_backend", "novelai")),
			Path(app["db_path"]), data_dir),
	]
	for controller in controllers:
		controller.add_routes(app.router)
