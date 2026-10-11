#!/usr/bin/env bash
# Start the ImagineGeneratorProxy server (POSIX)
# Usage:
#   ./start_server.sh [port]
# Auth comes from imagine_generator_proxy/config.yaml (token) or the
# NOVELAI_TOKEN / NOVELAI_EMAIL + NOVELAI_PASSWORD environment variables.

PORT="${1:-8080}"
export PORT="$PORT"

echo "Starting server on port $PORT..."
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$DIR/.venv/bin/python" ]; then
  "$DIR/.venv/bin/python" -m imagine_generator_proxy
elif [ -x "$DIR/.venv/Scripts/python.exe" ]; then
  "$DIR/.venv/Scripts/python.exe" -m imagine_generator_proxy
else
  python -m imagine_generator_proxy
fi
