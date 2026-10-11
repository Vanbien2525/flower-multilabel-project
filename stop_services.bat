@echo off
chcp 65001 > nul
echo ============================================================
echo   🛑 ĐANG TẠM DỪNG CÁC DỊCH VỤ & GIẢI PHÓNG RAM
echo ============================================================
echo.

echo 1. Đang dừng Backend Python (AI_Backend)...
nssm stop AI_Backend > nul 2>&1

echo 2. Đang tắt chế độ tự khởi động cùng Windows của Backend...
sc config AI_Backend start= demand > nul 2>&1

echo 3. Đang dừng dịch vụ Cloudflare Tunnel...
sc stop Cloudflared > nul 2>&1
sc config Cloudflared start= demand > nul 2>&1

echo.
echo ============================================================
echo   ✅ ĐÃ DỪNG THÀNH CÔNG! TOÀN BỘ RAM ĐÃ ĐƯỢC GIẢI PHÓNG.
echo   (Hệ thống sẽ KHÔNG tự khởi động lại khi bạn mở máy).
echo ============================================================
pause
