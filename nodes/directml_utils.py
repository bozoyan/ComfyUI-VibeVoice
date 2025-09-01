# DirectML工具和兼容性处理
# Created for VibeVoice-ComfyUI DirectML支持

import torch
import logging
import os

logger = logging.getLogger("VibeVoice")

class DirectMLUtils:
    """DirectML相关的工具函数"""
    
    @staticmethod
    def ensure_directml_compatibility():
        """确保DirectML环境的兼容性"""
        try:
            import torch_directml
            
            # 设置环境变量来避免编码问题
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['LANG'] = 'en_US.UTF-8'
            
            # 设置DirectML的调试选项
            os.environ['TORCH_DIRECTML_DEBUG'] = '0'  # 关闭调试输出
            
            logger.info("DirectML compatibility settings applied")
            return True
            
        except ImportError:
            logger.warning("torch-directml not available for compatibility setup")
            return False
        except Exception as e:
            logger.warning(f"DirectML compatibility setup failed: {e}")
            return False
    
    @staticmethod
    def safe_tensor_to_device(tensor, device, is_directml=False):
        """安全地将张量移动到设备，特别处理DirectML"""
        try:
            if is_directml:
                # DirectML需要特殊处理
                if tensor.dtype == torch.float16:
                    tensor = tensor.float()  # 转换为float32
                
                # 确保张量是连续的
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
            
            return tensor.to(device)
            
        except Exception as e:
            logger.warning(f"Failed to move tensor to device {device}: {e}")
            # 尝试CPU作为后备
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            return tensor.cpu()
    
    @staticmethod
    def prepare_directml_model(model, device):
        """为DirectML设备准备模型"""
        try:
            # 确保模型使用float32
            if hasattr(model, 'float'):
                model = model.float()
            
            # 移动到DirectML设备
            model = model.to(device)
            
            logger.info(f"Model prepared for DirectML device: {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to prepare model for DirectML: {e}")
            # 后备到CPU
            logger.info("Falling back to CPU")
            return model.cpu().float()
    
    @staticmethod
    def handle_directml_encoding_error(func, *args, **kwargs):
        """处理DirectML可能出现的编码错误"""
        try:
            return func(*args, **kwargs)
        except UnicodeDecodeError as e:
            logger.warning(f"DirectML encoding error: {e}")
            logger.info("Attempting CPU fallback for this operation")
            
            # 这里应该由调用者实现CPU后备逻辑
            raise e
    
    @staticmethod
    def get_directml_info():
        """获取DirectML设备信息"""
        try:
            import torch_directml
            
            if not torch_directml.is_available():
                return "DirectML not available"
            
            device_count = torch_directml.device_count()
            device_info = []
            
            for i in range(device_count):
                device = torch_directml.device(i)
                device_info.append(f"Device {i}: {device}")
            
            return f"DirectML devices: {device_count}\n" + "\n".join(device_info)
            
        except ImportError:
            return "torch-directml not installed"
        except Exception as e:
            return f"DirectML info error: {e}"

# 在模块加载时自动设置兼容性
DirectMLUtils.ensure_directml_compatibility()