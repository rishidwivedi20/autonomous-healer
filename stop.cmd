@echo off
rem Stops the dashboard safely (undoes any fault first). Option: -Sandbox also stops the app containers.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
