#!/usr/bin/env bash
# Start the ImagineGeneratorProxy server (POSIX)
# Usage:
#   ./start_server.sh <token> <port>
#   ./start_server.sh "my_token" 8080
# If no token supplied, uses NOVELAI_ACCESS_TOKEN env var if present.

TOKEN="$1"
PORT="${2:-8080}"

if [ -n "$TOKEN" ]; then
  export NOVELAI_ACCESS_TOKEN="$TOKEN"
fi

export PORT="$PORT"

echo "Starting server on port $PORT..."
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$DIR/imagine_generator_proxy/bin/python" ]; then
  "$DIR/imagine_generator_proxy/bin/python" -m imagine_generator_proxy.main
elif [ -x "$DIR/imagine_generator_proxy/Scripts/python.exe" ]; then
  "$DIR/imagine_generator_proxy/Scripts/python.exe" -m imagine_generator_proxy.main
else
  python -m imagine_generator_proxy.main
fi
