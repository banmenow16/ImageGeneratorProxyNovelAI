"""HTTP application layer: routes, request parsing, startup wiring.

Domain logic lives in GenerationService (service.py) and the prompts/
generation/review packages; handlers here stay thin adapters."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from imagine_generator_proxy.generation.base import ImageGenerator
from imagine_generator_proxy.prompts.massager import PromptMassager

from aiohttp import ClientSession, web

from novelai_api.NovelAI_API import NovelAIAPI

from imagine_generator_proxy.generation import (  # noqa: F401
	Ltx2Video,
	Wan26VideoOR,
	Wan27Video,
	Wan30Video,
	Wan22Video,
	Wan26Video,
	IllustriousGenerator,
	ImageGenerationType,
	NoobAIGenerator,
	NovelAIGenerator,
	PonyV6Generator,
)
from imagine_generator_proxy.prompts import PromptMassager
from imagine_generator_proxy.review import NoopReviewer, OpenRouterReviewer, Reviewer
from imagine_generator_proxy.server import advice, buzz, experiment, gallery, video
from imagine_generator_proxy.server.service import (
	GenerationRequest,
	GenerationService,
	PromptBlockedError,
	UpstreamBusyError,
	ImageRejectedError,
	GenerationError,
	sanitize_character,
)
from imagine_generator_proxy.server.storage import Database

try:
	import yaml
except Exception:
	yaml = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def _default_data_dir() -> Path:
	"""Runtime data location: the repo's data/ directory for source
	checkouts, or IGP_DATA_DIR for pip-installed deployments."""
	env = os.environ.get("IGP_DATA_DIR")
	if env:
		return Path(env)
	return Path(__file__).parents[3] / "data"


# Runtime data (config.yaml, cache/, images.db, logs/) lives OUTSIDE the
# package in data/, keeping src/ pure code.
DATA_DIR = _default_data_dir()


def configure_logging() -> None:
	if logging.getLogger().handlers:
		return

	log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
	log_dir = DATA_DIR / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(formatter)
	file_handler = RotatingFileHandler(
		log_dir / "server.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
	)
	file_handler.setFormatter(formatter)

	root_logger = logging.getLogger()
	root_logger.setLevel(getattr(logging, log_level, logging.INFO))
	root_logger.addHandler(console_handler)
	root_logger.addHandler(file_handler)


@web.middleware
async def request_logging_middleware(request: web.Request, handler) -> web.Response:
	started = time.perf_counter()
	try:
		response = await handler(request)
	except web.HTTPException:
		# router 404/405 etc. — expected traffic (external clients poll
		# unknown routes); log without a traceback
		LOGGER.info("HTTP %s %s -> %s", request.method, request.path, "not found")
		raise
	except Exception:
		elapsed = time.perf_counter() - started
		LOGGER.exception(
			"Unhandled exception while serving %s %s (%.3fs)",
			request.method, request.path, elapsed,
		)
		raise

	elapsed = time.perf_counter() - started
	if response.status >= 400:
		LOGGER.warning(
			"HTTP %s response for %s %s from %s (%.3fs)",
			response.status, request.method, request.path,
			request.remote or "unknown client", elapsed,
		)
	else:
		LOGGER.info(
			"HTTP %s %s -> %s (%.3fs)",
			request.method, request.path, response.status, elapsed,
		)
	return response


def _parse_generation_request(request: web.Request) -> Optional[GenerationRequest]:
	"""Build a GenerationRequest from query parameters, or None if invalid."""
	raw_prompt = request.query.get("prompt")
	if not raw_prompt:
		return None

	massager: PromptMassager = request.app["massager"]
	use_massage = massager.enabled and request.query.get("massage") not in ("0", "false")
	return GenerationRequest(
		raw_prompt=raw_prompt,
		character=sanitize_character(request.query.get("character") or "default"),
		q_negative=request.query.get("negative_prompt") or "",
		want_raw=(request.query.get("raw") in ("1", "true")
			or "image/" in request.headers.get("Accept", "")),
		force=request.query.get("force") in ("1", "true"),
		use_massage=use_massage,
	)


async def generate_handler(request: web.Request) -> web.Response:
	parsed = _parse_generation_request(request)
	if parsed is None:
		LOGGER.warning("Bad request: /generate is missing the 'prompt' query parameter")
		return web.json_response({"error": "missing 'prompt' query parameter"}, status=400)

	service: GenerationService = request.app["service"]
	try:
		return await service.generate(parsed)
	except PromptBlockedError as e:
		return web.json_response({"error": "prompt blocked by safety filter",
			"rule": e.rule, "reason": e.reason}, status=400)
	except ImageRejectedError:
		return web.json_response({"error": "image rejected by age-safety review"}, status=422)
	except UpstreamBusyError:
		return web.json_response({"error": "novelai busy, try again later"}, status=503)
	except GenerationError as e:
		return web.json_response({"error": str(e)}, status=500)


def load_config() -> dict:
	"""Load config.yaml at the package root (PyYAML, with a flat key:value
	fallback parser when PyYAML is unavailable)."""
	cfg_path = DATA_DIR / "config.yaml"
	if not cfg_path.exists():
		return {}

	try:
		text = cfg_path.read_text(encoding="utf-8")
	except Exception:
		LOGGER.exception("Failed to read config.yaml")
		return {}

	if yaml:
		try:
			return yaml.safe_load(text) or {}
		except Exception:
			LOGGER.exception("Failed to parse config.yaml with PyYAML")
			return {}

	# Fallback simple parser for key: value pairs when PyYAML is not installed
	cfg: dict = {}
	for line in text.splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		if ":" not in line:
			continue
		k, v = line.split(":", 1)
		k = k.strip()
		v = v.strip()
		if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
			v = v[1:-1]
		cfg[k] = v
	LOGGER.warning("PyYAML not installed — using simple config parser")
	return cfg


def _quiet_loop_exception_handler(loop, context) -> None:
	"""Downgrade Windows proactor ConnectionResetError noise (clients
	dropping keep-alive sockets) to DEBUG; everything else is real."""
	exc = context.get("exception")
	if isinstance(exc, ConnectionResetError):
		LOGGER.debug("connection reset (proactor noise): %s", exc)
		return
	loop.default_exception_handler(context)


async def on_startup(app: web.Application) -> None:
	asyncio.get_event_loop().set_exception_handler(_quiet_loop_exception_handler)
	session = ClientSession()
	app["session"] = session

	api = NovelAIAPI(session)
	app["novelai_api"] = api

	# Global lock: one generation at a time per upstream
	app["generate_lock"] = asyncio.Lock()

	cache_dir = DATA_DIR / "cache"
	try:
		cache_dir.mkdir(parents=True, exist_ok=True)
	except Exception:
		LOGGER.exception("Failed to create cache directory")
	app["cache_dir"] = cache_dir

	database = Database(DATA_DIR / "images.db")
	database.initialize()
	app["db_path"] = database.path

	config = load_config()
	app["config"] = config

	# --- backend registry: each generator carries its own rule overrides ---
	novelai_generator = NovelAIGenerator(api, config, lock=app["generate_lock"])
	backends: dict = {
		"novelai": Backend(
			name="novelai", display="NovelAI v4.5", generator=novelai_generator,
			massager=PromptMassager.from_config(config, DATA_DIR,
				backend_overrides=novelai_generator.rules_overrides())),
	}
	civitai_key = config.get("civitaiToken") or config.get("civitai_token")
	if civitai_key:
		for cls in (IllustriousGenerator, PonyV6Generator, NoobAIGenerator):
			gen = cls(api_key=str(civitai_key), session=session,
				urn_override=config.get(f"civitai_model_{cls.backend_name}"),
				lora_overrides=config.get(f"civitai_loras_{cls.backend_name}"))
			backends[cls.backend_name] = Backend(
				name=cls.backend_name, display=cls.display_name, generator=gen,
				massager=PromptMassager.from_config(config, DATA_DIR,
					backend_overrides=gen.rules_overrides()))
		LOGGER.info("Civitai backends registered: %s",
			", ".join(b for b in backends if b != "novelai"))
	else:
		LOGGER.info("No civitaiToken in config — Civitai backends disabled")
	app["backends"] = backends

	# Video backends (image-to-video): same registry pattern; Civitai-powered
	civitai_key_v = civitai_key or config.get("civitaiToken") or config.get("civitai_token")
	video_backends: dict = {}
	if civitai_key_v:
		for vcls in (Wan22Video, Wan26Video, Ltx2Video):
			video_backends[vcls.backend_name] = vcls(api_key=str(civitai_key_v), session=session)
	# separate loop so mypy keeps distinct class-tuple types
	or_key = config.get("openRouterToken") or config.get("openrouter_api_key")
	if or_key:
		for vcls in (Wan27Video, Wan26VideoOR, Wan30Video):  # type: ignore[assignment]
			video_backends[vcls.backend_name] = vcls(api_key=str(or_key), session=session)
	app["video_backends"] = video_backends
	app["video_backend"] = str(config.get("video_backend", "wan22"))
	app["videos_dir"] = DATA_DIR / "videos"
	(DATA_DIR / "videos").mkdir(parents=True, exist_ok=True)
	conn = database.connect()
	try:
		video.ensure_video_schema(conn)
	finally:
		conn.close()

	app["massager"] = backends["novelai"].massager
	app["generator"] = novelai_generator

	# Production /generate uses the configured backend
	backend_name = str(config.get("generation_backend", "novelai"))
	if backend_name not in backends:
		LOGGER.warning("Configured generation_backend %r not available — using novelai", backend_name)
		backend_name = "novelai"
	app["generation_backend"] = backend_name
	production = backends[backend_name]

	# Post-generation reviewer: two-tier OpenRouter vision review when enabled
	# in config (images are sent to OpenRouter); noop otherwise.
	reviewer: Reviewer = NoopReviewer()
	if config.get("vision_review"):
		reviewer = OpenRouterReviewer.from_config(config, db_path=database.path) or NoopReviewer()
	app["reviewer"] = reviewer

	# The pipeline service owns the /generate flow end to end
	app["service"] = GenerationService(production.massager, production.generator,
		reviewer, config, cache_dir, database)

	# Experiment workbench reviewer: runs whenever an OpenRouter key exists,
	# independently of the production vision_review flag.
	app["experiment_reviewer"] = experiment.build_experiment_reviewer(config, database.path)

	token = config.get("token") or os.environ.get("NOVELAI_TOKEN")
	email = os.environ.get("NOVELAI_EMAIL")
	password = os.environ.get("NOVELAI_PASSWORD")

	if token:
		try:
			await api.high_level.login_with_token(token)
		except Exception:
			LOGGER.exception("Failed to login with token")
	elif email and password:
		try:
			await api.high_level.login(email, password)
		except Exception:
			LOGGER.exception("Failed to login with email/password")


@dataclass
class Backend:
	"""A selectable generation backend: generator + its ruleset-matched massager."""
	name: str
	display: str
	generator: ImageGenerator
	massager: PromptMassager


async def on_cleanup(app: web.Application) -> None:
	session: Optional[ClientSession] = app.get("session")
	if session:
		await session.close()


def create_app() -> web.Application:
	app = web.Application(middlewares=[request_logging_middleware])
	app.router.add_get("/generate", generate_handler)
	app.router.add_get("/", gallery.gallery_handler)
	app.router.add_get("/folder/{character}", gallery.folder_handler)
	app.router.add_static("/cache", str(DATA_DIR / "cache"))
	(DATA_DIR / "videos").mkdir(parents=True, exist_ok=True)
	app.router.add_static("/video-files", str(DATA_DIR / "videos"))
	(DATA_DIR / "advice").mkdir(parents=True, exist_ok=True)
	app.router.add_static("/advice-files", str(DATA_DIR / "advice"))
	video.add_routes(app)
	buzz.add_routes(app)
	advice.add_routes(app)
	experiment.add_routes(app)
	app.on_startup.append(on_startup)
	app.on_cleanup.append(on_cleanup)
	return app


if __name__ == "__main__":
	from imagine_generator_proxy.__main__ import main
	main()
