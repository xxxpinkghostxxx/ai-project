@echo off
echo ============================================
echo   AI Project - Environment Setup
echo ============================================
echo.

echo Installing PyTorch (CUDA 12.1)...
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing Taichi (GPU compute)...
pip install taichi

echo.
echo Installing core dependencies...
pip install numpy opencv-python PyQt6 Pillow mss sounddevice typing-extensions

echo.
echo ============================================
echo   Setup complete!
echo ============================================
pause
