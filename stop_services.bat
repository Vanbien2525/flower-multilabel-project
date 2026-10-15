@echo off
chcp 65001 > nul
echo ============================================================
echo   dung
echo ============================================================
echo.

echo 1. Backend Python (AI_Backend)...
nssm stop AI_Backend > nul 2>&1

echo 2. tat Backend...
sc config AI_Backend start= demand > nul 2>&1

echo 3.  Cloudflare Tunnel...
sc stop Cloudflared > nul 2>&1
sc config Cloudflared start= demand > nul 2>&1

echo.
echo ============================================================
echo   dung thanh cong
echo ============================================================
pause
