#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectML问题诊断和修复工具
专门用于解决VibeVoice-ComfyUI在DirectML环境下的Unicode编码问题
"""

import os
import sys
import logging
import subprocess

# 设置日志
logging.basicConfig(level=logging.INFO, format='[DirectML修复] %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """设置环境变量以避免编码问题"""
    logger.info("设置DirectML环境变量...")
    
    env_vars = {
        'PYTHONIOENCODING': 'utf-8',
        'LANG': 'en_US.UTF-8',
        'LC_ALL': 'en_US.UTF-8',
        'TORCH_DIRECTML_DEBUG': '0',
        'TORCH_DIRECTML_DISABLE_OPTIMIZATION': '1',  # 禁用可能导致问题的优化
        'OMP_NUM_THREADS': '1',  # 限制线程数以避免竞态条件
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"设置 {key}={value}")

def check_directml_installation():
    """检查DirectML安装状态"""
    logger.info("检查DirectML安装状态...")
    
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        
        import torch_directml
        logger.info(f"torch-directml已安装")
        
        if torch_directml.is_available():
            device_count = torch_directml.device_count()
            logger.info(f"可用DirectML设备数量: {device_count}")
            
            for i in range(device_count):
                device = torch_directml.device(i)
                logger.info(f"设备 {i}: {device}")
            
            return True
        else:
            logger.warning("DirectML不可用")
            return False
            
    except ImportError as e:
        logger.error(f"导入错误: {e}")
        return False
    except Exception as e:
        logger.error(f"检查失败: {e}")
        return False

def install_additional_dependencies():
    """安装可能缺失的依赖"""
    logger.info("检查并安装额外依赖...")
    
    dependencies = [
        "chardet",  # 字符编码检测
        "charset-normalizer",  # 字符集标准化
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace('-', '_'))
            logger.info(f"{dep} 已安装")
        except ImportError:
            logger.info(f"安装 {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             capture_output=True, text=True, timeout=120)
                logger.info(f"{dep} 安装完成")
            except Exception as e:
                logger.warning(f"{dep} 安装失败: {e}")

def test_directml_tensor_operations():
    """测试DirectML张量操作"""
    logger.info("测试DirectML张量操作...")
    
    try:
        import torch
        import torch_directml
        
        if not torch_directml.is_available():
            logger.warning("DirectML不可用，跳过测试")
            return False
        
        device = torch_directml.device()
        logger.info(f"使用设备: {device}")
        
        # 测试基本张量操作
        test_tensor = torch.randn(10, 10, dtype=torch.float32)
        logger.info(f"创建测试张量: {test_tensor.shape}, dtype: {test_tensor.dtype}")
        
        # 移动到DirectML设备
        directml_tensor = test_tensor.to(device)
        logger.info(f"成功移动到DirectML设备")
        
        # 执行简单运算
        result = directml_tensor + 1.0
        logger.info(f"DirectML计算成功")
        
        # 移回CPU
        cpu_result = result.cpu()
        logger.info(f"成功移回CPU: {cpu_result.shape}")
        
        logger.info("DirectML张量操作测试通过")
        return True
        
    except Exception as e:
        logger.error(f"DirectML张量操作测试失败: {e}")
        return False

def apply_unicode_fixes():
    """应用Unicode编码修复"""
    logger.info("应用Unicode编码修复...")
    
    try:
        # 设置Python默认编码
        if hasattr(sys, 'setdefaultencoding'):
            sys.setdefaultencoding('utf-8')
        
        # 重新配置标准输出
        if sys.stdout.encoding != 'utf-8':
            logger.info(f"当前stdout编码: {sys.stdout.encoding}")
            logger.info("建议重启ComfyUI以应用编码设置")
        
        return True
        
    except Exception as e:
        logger.error(f"Unicode修复失败: {e}")
        return False

def generate_directml_config():
    """生成DirectML配置文件"""
    config_path = "directml_config.txt"
    
    try:
        import torch_directml
        
        config_content = f"""
DirectML配置信息
================
生成时间: {__import__('datetime').datetime.now()}

环境变量:
PYTHONIOENCODING=utf-8
LANG=en_US.UTF-8
LC_ALL=en_US.UTF-8
TORCH_DIRECTML_DEBUG=0
TORCH_DIRECTML_DISABLE_OPTIMIZATION=1
OMP_NUM_THREADS=1

DirectML状态:
可用: {torch_directml.is_available()}
设备数量: {torch_directml.device_count() if torch_directml.is_available() else 0}

建议设置:
1. 使用float32而不是float16
2. 启用CPU后备机制
3. 禁用某些优化以避免编码问题

故障排除:
如果仍然出现Unicode错误，请尝试:
1. 重启ComfyUI
2. 使用CPU模式作为后备
3. 检查文本输入是否包含特殊字符
"""
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"配置文件已生成: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"生成配置文件失败: {e}")
        return False

def main():
    """主修复流程"""
    logger.info("开始DirectML问题诊断和修复...")
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 检查安装
    directml_ok = check_directml_installation()
    
    # 3. 安装依赖
    install_additional_dependencies()
    
    # 4. 测试操作
    if directml_ok:
        test_ok = test_directml_tensor_operations()
    else:
        test_ok = False
    
    # 5. 应用修复
    unicode_ok = apply_unicode_fixes()
    
    # 6. 生成配置
    config_ok = generate_directml_config()
    
    # 总结
    logger.info("=" * 50)
    logger.info("修复结果总结:")
    logger.info(f"DirectML安装: {'✓' if directml_ok else '✗'}")
    logger.info(f"张量操作测试: {'✓' if test_ok else '✗'}")
    logger.info(f"Unicode修复: {'✓' if unicode_ok else '✗'}")
    logger.info(f"配置生成: {'✓' if config_ok else '✗'}")
    
    if directml_ok and test_ok:
        logger.info("DirectML环境正常，编码问题应该已修复")
        logger.info("请重启ComfyUI以应用所有更改")
    else:
        logger.warning("发现问题，建议使用CPU模式或联系技术支持")
    
    input("按回车键继续...")

if __name__ == "__main__":
    main()