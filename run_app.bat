@echo off
chcp 65001 > nul
echo ============================================================
echo   🌸 VBxIris: KHỞI ĐỘNG HỆ THỐNG NHẬN DIỆN HOA & CHĂM SÓC
echo ============================================================
echo.
echo Đang kiểm tra môi trường Python...

set "ROOT=%~dp0"
cd /d "%ROOT%"

if exist "backend\venv\Scripts\python.exe" (
    set "PY_CMD=%ROOT%backend\venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

echo Đang khởi động Flask Server tại http://localhost:5000 ...
echo Trình duyệt Chrome sẽ tự động mở trang web sau 3 giây...
echo.

start "" http://localhost:5000

cd /d "%ROOT%backend"
"%PY_CMD%" app.py
pause
