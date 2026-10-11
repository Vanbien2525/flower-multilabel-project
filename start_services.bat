@echo off
chcp 65001 > nul
echo ============================================================
echo   🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG CLOUD & AI BACKEND
echo ============================================================
echo.

echo 1. Đang bật dịch vụ Cloudflare Tunnel...
sc config Cloudflared start= auto > nul 2>&1
sc start Cloudflared > nul 2>&1

echo 2. Đang bật Backend Python AI_Backend...
sc config AI_Backend start= auto > nul 2>&1
nssm start AI_Backend > nul 2>&1

echo.
echo ============================================================
echo   🌸 HỆ THỐNG ĐÃ SẴN SÀNG!
echo   👉 Truy cập web: https://ai.vbfloweriris.id.vn
echo ============================================================
pause
