# Imagine Generator Proxy

Lightweight HTTP proxy to generate images via NovelAI and cache results.

Installation
```
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

Run
```
python -m imagine_generator_proxy
```

Optional flags: `--port 8099` and `--host 127.0.0.1` (port also read from the
`PORT` environment variable, default 8080).

Or use the provided start scripts for convenience.
start_server.ps1
start_server.sh
start_server.bat

Configuration
- Edit `imagine_generator_proxy/config.yaml` to enter token or set environment variables `NOVELAI_TOKEN`
