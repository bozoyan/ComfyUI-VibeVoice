@echo off
chcp 65001 > nul
title VibeVoice DirectML Unicode Error Fix

echo =====================================
echo VibeVoice DirectML Unicode 错误修复
echo =====================================
echo.

echo 设置环境变量...
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8
set TORCH_DIRECTML_DEBUG=0
set TORCH_DIRECTML_DISABLE_OPTIMIZATION=1
set OMP_NUM_THREADS=1

echo 环境变量设置完成
echo.

echo 运行修复脚本...
python fix_directml_unicode.py

echo.
echo 修复完成！
echo 请重启 ComfyUI 以应用更改。
echo.

pause