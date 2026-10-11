"""Character data: the Mirror's verified output, readable over HTTP.

GET /api/characters            → all characters' canonical tags + advice
GET /api/characters/{name}     → one character incl. full self-description

External applications use these to render a character consistently: her
`canonical_tags` are the verified identity block to repeat verbatim in
every image prompt; `advice` is her written instruction block for the bot.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from aiohttp import web

from imagine_generator_proxy.controllers.base import not_found
from imagine_generator_proxy.server.storage import Database

LIST_COLUMNS = ("name", "canonical_tags", "advice", "updated_at")
DETAIL_COLUMNS = LIST_COLUMNS + ("description",)


class CharacterController:
	"""Read endpoints over the characters table."""

	def __init__(self, db_path: Path) -> None:
		self._db_path = db_path

	def add_routes(self, router: web.UrlDispatcher) -> None:
		router.add_get("/api/characters", self.list_characters)
		router.add_get("/api/characters/{name}", self.get_character)

	async def list_characters(self, request: web.Request) -> web.Response:
		with self._connect() as conn:
			rows = conn.execute(
				f"SELECT {', '.join(LIST_COLUMNS)} FROM characters ORDER BY name"
			).fetchall()
		return web.json_response({"characters": [dict(r) for r in rows]})

	async def get_character(self, request: web.Request) -> web.Response:
		name = request.match_info["name"]
		with self._connect() as conn:
			row = conn.execute(
				f"SELECT {', '.join(DETAIL_COLUMNS)} FROM characters WHERE name = ?",
				(name,)).fetchone()
		if row is None:
			return not_found(f"character {name!r} not found")
		return web.json_response({"character": dict(row)})

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(str(self._db_path))
		conn.row_factory = sqlite3.Row
		return conn
