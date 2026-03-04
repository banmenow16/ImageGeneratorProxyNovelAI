@echo off
REM Start the ImagineGeneratorProxy server (Windows CMD)
REM Usage:
REM   start_server.bat <token> [port]
REM If no token is supplied, it will use NOVELAI_ACCESS_TOKEN from environment.

SET TOKEN=%1
IF "%2"=="" (
  SET PORT=8080
) ELSE (
  SET PORT=%2
)

IF NOT "%TOKEN%"=="" SET NOVELAI_ACCESS_TOKEN=%TOKEN%

SET PORT=%PORT%
echo Starting server on port %PORT%...
IF EXIST "%~dp0imagine_generator_proxy\Scripts\python.exe" (
  "%~dp0imagine_generator_proxy\Scripts\python.exe" -m imagine_generator_proxy.main
) ELSE (
  python -m imagine_generator_proxy.main
)
