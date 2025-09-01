# Created by Fabio Sarracino
__version__ = "1.0.4"
__author__ = "Fabio Sarracino"
__title__ = "VibeVoice ComfyUI"

import logging
import os
import sys
import subprocess

# Setup logging
logger = logging.getLogger("VibeVoice")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[VibeVoice] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def apply_timm_compatibility_patches():
    """Apply compatibility patches for timm package conflicts"""
    try:
        import timm.data
        
        # Patch missing functions that cause import errors
        patches = {
            'ImageNetInfo': lambda: type('ImageNetInfo', (), {'__init__': lambda self: None})(),
            'infer_imagenet_subset': lambda class_to_idx: 'imagenet',
            'get_imagenet_subset_labels': lambda *args, **kwargs: [],
            'get_imagenet_subset_info': lambda *args, **kwargs: {},
            'resolve_data_config': lambda *args, **kwargs: {}
        }
        
        for attr_name, patch_func in patches.items():
            if not hasattr(timm.data, attr_name):
                if attr_name == 'ImageNetInfo':
                    setattr(timm.data, attr_name, type('ImageNetInfo', (), {'__init__': lambda self: None}))
                else:
                    setattr(timm.data, attr_name, patch_func)
        
        return True
    except Exception as e:
        return False

def check_directml_support():
    """Check if DirectML is available for AMD GPU acceleration"""
    try:
        import torch_directml
        if torch_directml.is_available():
            device_count = torch_directml.device_count()
            logger.info(f"DirectML detected with {device_count} AMD GPU device(s)")
            return True
        else:
            logger.info("DirectML 可用，但未检测到兼容的 AMD GPU")
            return False
    except ImportError:
        logger.info("未安装 DirectML。对于 AMD GPU 加速，请安装 torch-directml")
        return False
    except Exception as e:
        logger.warning(f"DirectML 检查失败: {e}")
        return False

def check_vibevoice_available():
    """Check if VibeVoice is available for import"""
    try:
        # Apply timm patches first
        apply_timm_compatibility_patches()
        
        import vibevoice
        return True
    except ImportError:
        return False

def install_vibevoice():
    """Install VibeVoice if not already installed"""
    if check_vibevoice_available():
        return True
        
    try:
        # Install VibeVoice with specific transformers version to avoid LossKwargs issue
        logger.info("安装具有兼容依赖项的 VibeVoice...")
        
        # First install compatible transformers version
        transformers_cmd = [sys.executable, "-m", "pip", "install", "transformers>=4.44.0"]
        result = subprocess.run(transformers_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.warning(f"Transformers 安装警告: {result.stderr}")
        
        # Then install VibeVoice
        cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/microsoft/VibeVoice.git"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("VibeVoice 安装完成")
            return check_vibevoice_available()  # Verify installation
        else:
            logger.error(f"安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"安装错误: {e}")
        return False

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register text loading node (always available)
try:
    from .nodes.load_text_node import LoadTextFromFileNode
    NODE_CLASS_MAPPINGS["LoadTextFromFileNode"] = LoadTextFromFileNode
    NODE_DISPLAY_NAME_MAPPINGS["LoadTextFromFileNode"] = "VibeVoice 从文件加载文本 🎯BOZO "
except Exception as e:
    logger.error(f"无法注册 LoadTextFromFile 节点: {e}")

# Register VibeVoice nodes (requires VibeVoice installation)
if install_vibevoice():
    # Check for DirectML support
    directml_available = check_directml_support()
    if directml_available:
        logger.info("AMD GPU 使用 DirectML 加速已准备就绪")
    
    try:
        from .nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
        from .nodes.multi_speaker_node import VibeVoiceMultipleSpeakersNode
        from .nodes.free_memory_node import VibeVoiceFreeMemoryNode
        
        # Single speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceSingleSpeakerNode"] = VibeVoiceSingleSpeakerNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceSingleSpeakerNode"] = "VibeVoice 单扬声器 🎯BOZO "
        
        # Multi speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceMultipleSpeakersNode"] = VibeVoiceMultipleSpeakersNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceMultipleSpeakersNode"] = "VibeVoice 多扬声器 🎯BOZO "
        
        # Free memory node
        NODE_CLASS_MAPPINGS["VibeVoiceFreeMemoryNode"] = VibeVoiceFreeMemoryNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceFreeMemoryNode"] = "VibeVoice 释放内存 🎯BOZO  "
        
    except Exception as e:
        logger.error(f"无法注册 VibeVoice 节点: {e}")
        logger.info("VibeVoice 可能需要重新启动 ComfyUI 才能完成安装")
else:
    logger.warning("VibeVoice nodes 不可用 - 安装失败")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']