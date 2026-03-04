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

Or use the provided start scripts for convenience.
start_server.ps1
start_server.sh
start_server.bat

Configuration
- Edit `imagine_generator_proxy/config.yaml` to enter token or set environment variables `NOVELAI_TOKEN`
