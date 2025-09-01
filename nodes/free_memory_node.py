# Created by Fabio Sarracino
# Node to free VibeVoice model memory

import logging
import torch
import gc
from typing import Any

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceFreeMemoryNode:
    """Node to explicitly free VibeVoice model memory"""
    
    # Class variables to store node instances
    _single_speaker_instances = []
    _multi_speaker_instances = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input that triggers memory cleanup and gets passed through"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "free_vibevoice_memory"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Free all loaded VibeVoice models from memory when audio passes through"
    
    @classmethod
    def register_single_speaker(cls, node_instance):
        """Register a single speaker node instance"""
        if node_instance not in cls._single_speaker_instances:
            cls._single_speaker_instances.append(node_instance)
    
    @classmethod
    def register_multi_speaker(cls, node_instance):
        """Register a multi speaker node instance"""
        if node_instance not in cls._multi_speaker_instances:
            cls._multi_speaker_instances.append(node_instance)
    
    def free_vibevoice_memory(self, audio):
        """Free memory from all VibeVoice nodes and pass through the audio"""
        
        try:
            freed_count = 0
            
            # Try to access and free memory from globally cached instances
            # ComfyUI might cache node instances
            try:
                import sys
                from .base_vibevoice import BaseVibeVoiceNode
                
                # Search in all modules for BaseVibeVoiceNode instances
                for module_name, module in sys.modules.items():
                    if module and 'vibevoice' in module_name.lower():
                        for attr_name in dir(module):
                            if not attr_name.startswith('_'):
                                try:
                                    attr = getattr(module, attr_name)
                                    if isinstance(attr, type) and issubclass(attr, BaseVibeVoiceNode):
                                        # Check if the class has any cached instances
                                        for instance_attr in dir(attr):
                                            instance = getattr(attr, instance_attr)
                                            if isinstance(instance, BaseVibeVoiceNode) and hasattr(instance, 'free_memory'):
                                                instance.free_memory()
                                                freed_count += 1
                                except:
                                    pass
            except:
                pass
            
            # Free from registered single speaker instances
            for node in self._single_speaker_instances:
                if hasattr(node, 'free_memory'):
                    node.free_memory()
                    freed_count += 1
            
            # Free from registered multi speaker instances  
            for node in self._multi_speaker_instances:
                if hasattr(node, 'free_memory'):
                    node.free_memory()
                    freed_count += 1
            
            # Force garbage collection
            gc.collect()
            
            # Clear cache based on available hardware
            cache_cleared = False
            
            # Try DirectML cache clearing first
            try:
                import torch_directml
                if torch_directml.is_available():
                    # DirectML doesn't have explicit cache clearing like CUDA
                    # but we can try if the method exists
                    if hasattr(torch_directml, 'empty_cache'):
                        torch_directml.empty_cache()
                    cache_cleared = True
                    logger.info(f"Freed VibeVoice memory from {freed_count} nodes and cleared DirectML cache")
            except (ImportError, Exception):
                pass
            
            # Clear CUDA cache if available and DirectML wasn't used
            if not cache_cleared and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info(f"Freed VibeVoice memory from {freed_count} nodes and cleared CUDA cache")
            elif not cache_cleared:
                logger.info(f"Freed VibeVoice memory from {freed_count} nodes")
            
            # Pass through the audio unchanged
            return (audio,)
                
        except Exception as e:
            logger.error(f"Error freeing VibeVoice memory: {str(e)}")
            # Still pass through audio even if error occurs
            return (audio,)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always execute this node"""
        return float("nan")  # Forces re-execution every time