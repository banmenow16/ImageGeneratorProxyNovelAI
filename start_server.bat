@echo off
REM Start the ImagineGeneratorProxy server (Windows CMD)
REM Usage:
REM   start_server.bat [port]
REM Auth comes from imagine_generator_proxy\config.yaml (token) or the
REM NOVELAI_TOKEN / NOVELAI_EMAIL + NOVELAI_PASSWORD environment variables.

IF "%1"=="" (
  SET PORT=8080
) ELSE (
  SET PORT=%1
)

echo Starting server on port %PORT%...
IF EXIST "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m imagine_generator_proxy
) ELSE (
  python -m imagine_generator_proxy
)
