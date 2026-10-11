@echo off
rem One command that brings everything up (sandbox app, dashboard, browser). Options: -Fresh  -NoBrowser
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
