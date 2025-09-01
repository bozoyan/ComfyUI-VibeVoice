#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectML安装脚本
为VibeVoice-ComfyUI插件安装DirectML支持
"""

import subprocess
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='[DirectML安装] %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """运行命令并检查结果"""
    logger.info(f"正在{description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"{description}成功")
            return True
        else:
            logger.error(f"{description}失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"{description}出错: {e}")
        return False

def main():
    """主安装流程"""
    logger.info("开始安装DirectML支持...")
    
    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    
    # 安装torch-directml
    logger.info("正在安装torch-directml...")
    cmd = [sys.executable, "-m", "pip", "install", "torch-directml"]
    if not run_command(cmd, "安装torch-directml"):
        logger.error("torch-directml安装失败")
        return False
    
    # 验证安装
    logger.info("验证DirectML安装...")
    try:
        import torch
        import torch_directml
        
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        if torch_directml.is_available():
            device_count = torch_directml.device_count()
            logger.info(f"DirectML可用，检测到 {device_count} 个设备")
            
            # 测试设备
            for i in range(device_count):
                device = torch_directml.device(i)
                logger.info(f"设备 {i}: {device}")
            
            # 简单测试
            device = torch_directml.device()
            test_tensor = torch.randn(3, 3).to(device)
            logger.info("DirectML设备测试成功")
            
            return True
        else:
            logger.error("DirectML不可用")
            return False
            
    except ImportError as e:
        logger.error(f"导入错误: {e}")
        return False
    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("DirectML安装完成！现在可以使用AMD显卡加速VibeVoice")
        logger.info("请重启ComfyUI以使用新的DirectML支持")
    else:
        logger.error("DirectML安装失败，请检查错误信息")
    
    input("按回车键继续...")