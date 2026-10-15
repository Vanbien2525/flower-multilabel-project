@echo off
chcp 65001 > nul
echo ============================================================
echo    CLOUD & AI BACKEND
echo ============================================================
echo.

echo 1.  Cloudflare Tunnel..
sc config Cloudflared start= auto > nul 2>&1
sc start Cloudflared > nul 2>&1

echo 2. Backend Python AI_Backend...
sc config AI_Backend start= auto > nul 2>&1
nssm start AI_Backend > nul 2>&1

echo.
echo ============================================================
echo    !
echo   web: https://ai.vbfloweriris.id.vn
echo ============================================================
pause
