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
sc config Cloudflared start= auto > nul 2>&1
net start Cloudflared > nul 2>&1

echo.
echo 2. Khoi dong AI Backend Python...
sc config AI_Backend start= auto > nul 2>&1
nssm stop AI_Backend > nul 2>&1
timeout /t 2 /nobreak > nul
nssm start AI_Backend

echo.
echo ============================================================
echo   HE THONG DA BAT THANH CONG!
echo   Truy cap web tai: https://ai.vbfloweriris.id.vn
echo ============================================================
echo.
pause
