@echo off
setlocal

:: Kiem tra quyen Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0"

echo ============================================================
echo   DANG KHOI DONG CLOUDFLARE TUNNEL VA AI BACKEND
echo ============================================================
echo.

echo 1. Khoi dong Cloudflare Tunnel...
sc config Cloudflared start= auto
net start Cloudflared

echo.
echo 2. Khoi dong AI Backend...
sc config AI_Backend start= auto
nssm restart AI_Backend

echo.
echo ============================================================
echo   HE THONG DA BAT THANH CONG!
echo   Truy cap web tai: https://ai.vbfloweriris.id.vn
echo ============================================================
echo.
pause
