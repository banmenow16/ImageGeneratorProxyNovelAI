# Imagine Generator Proxy

Lightweight HTTP proxy that generates images via NovelAI, massages prompts to
reduce bad generations, optionally reviews results with a vision model, and
caches everything on disk + SQLite.

## Install

```
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

## Run

```
python -m imagine_generator_proxy
```

Optional flags: `--port 8099` and `--host 127.0.0.1` (port also read from the
`PORT` environment variable, default 8080). Or use the provided start scripts.

## Endpoints

| Route | What it does |
|---|---|
| `GET /generate?character=X&prompt=...` | Generate (or serve cached) image. Extra params: `negative_prompt`, `raw=1`, `force=1`, `massage=0` (bypass the massager, legacy cache keys). JSON by default; send `Accept: image/*` for raw PNG. |
| `GET /` | Cache gallery — folders per character. |
| `GET /folder/{character}` | Image grid for one character with prompts. |
| `GET /experiment` | **Pipeline workbench page**: run a prompt through the full pipeline (raw baseline → massaged → review/fix/regenerate chain) with every image and verdict visible. |
| `POST /experiment/run` | JSON API behind the workbench. |
| `GET /experiment/runs`, `/experiment/run/{id}` | Experiment history. |

## Configuration (`imagine_generator_proxy/config.yaml`, gitignored)

Copy `config.example.yaml` to start. Keys: `token` (NovelAI), generation
settings (`generation_retries`, `generation_backoff`, `generation_min_interval`),
massager toggles (`prompt_massage`, `safety_filter`, `negative_prompt`,
`rules_file`), and vision review (`openRouterToken`, `vision_review`,
`reviewer_model`, `reviewer_escalation_model`, `vision_review_retries`).

**Vision review is off by default.** Enabling it sends every generated image to
OpenRouter for quality/age review (~$0.001–0.01 per generation) and can
regenerate rejected images.

## Architecture

| Module | Role |
|---|---|
| `main.py` | App factory, `/generate` handler, startup/config. |
| `generation.py` | NovelAI calls: lock, pacing, 429 retry/backoff. |
| `prompt_massager.py` | Rule pipeline: safety filter → cleanup → enhance. |
| `prompt_rules.py` + `rule_data.json` | Versioned rules + tag vocabularies. |
| `reviewer.py` / `openrouter_reviewer.py` | Two-tier vision QC (cheap first pass + age escalation), FIX proposals, retrieval hints from review history. |
| `experiment.py` + `experiment.html` | The pipeline workbench. |
| `storage.py` | Digests, DB schema/migrations, review logging. |
| `gallery.py` | Cache gallery pages. |

Safety filter: prompts with minor-coded content are refused (HTTP 400) before
any generation; age-flagged *images* are refused (HTTP 422) after two-model
agreement. Cache digests are namespaced by rule version, so rule changes
regenerate cleanly while unchanged rules keep serving cached images.

## Tests

```
python -m pytest tests/
```
