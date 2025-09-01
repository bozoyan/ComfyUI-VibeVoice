# Created by Fabio Sarracino

import logging
import os
import tempfile
import torch
import numpy as np
import re
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceSingleSpeakerNode(BaseVibeVoiceNode):
    def __init__(self):
        super().__init__()
        # Register this instance for memory management
        try:
            from .free_memory_node import VibeVoiceFreeMemoryNode
            VibeVoiceFreeMemoryNode.register_single_speaker(self)
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the VibeVoice text-to-speech system.", 
                    "tooltip": "Text to convert to speech. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (["VibeVoice-1.5B", "VibeVoice-7B-Preview"], {
                    "default": "VibeVoice-1.5B", 
                    "tooltip": "Model to use. 1.5B is faster, 7B has better quality"
                }),
                "attention_type": (["auto", "eager", "sdpa", "flash_attention_2"], {
                    "default": "auto",
                    "tooltip": "Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU"
                }),
                "free_memory_after_generate": ("BOOLEAN", {"default": True, "tooltip": "Free model from memory after generation to save VRAM/RAM. Disable to keep model loaded for faster subsequent generations"}),
                "diffusion_steps": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1, "tooltip": "Number of denoising steps. More steps = better quality but slower. Default: 20"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "voice_to_clone": ("AUDIO", {"tooltip": "Optional: Reference voice to clone. If not provided, synthetic voice will be used."}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate speech from text using Microsoft VibeVoice with optional voice cloning"

    def _prepare_voice_samples(self, speakers: list, voice_to_clone) -> List[np.ndarray]:
        """Prepare voice samples from input audio or create synthetic ones"""
        
        if voice_to_clone is not None:
            # Use the base class method to prepare audio
            audio_np = self._prepare_audio_from_comfyui(voice_to_clone)
            if audio_np is not None:
                return [audio_np]
        
        # Create synthetic voice samples for speakers
        voice_samples = []
        for i, speaker in enumerate(speakers):
            voice_sample = self._create_synthetic_voice_sample(i)
            voice_samples.append(voice_sample)
            
        return voice_samples
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-1.5B", 
                       attention_type: str = "auto", free_memory_after_generate: bool = True,
                       diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                       use_sampling: bool = False, voice_to_clone=None,
                       temperature: float = 0.95, top_p: float = 0.95):
        """Generate speech from text using VibeVoice"""
        
        try:
            # Use text directly (it now serves as both manual input and connection input)
            if text and text.strip():
                final_text = text
            else:
                raise Exception("No text provided. Please enter text or connect from LoadTextFromFile node.")
            
            # Get model mapping and load model with attention type
            model_mapping = self._get_model_mapping()
            model_path = model_mapping.get(model, model)
            self.load_model(model_path, attention_type)
            
            # For single speaker, we just use ["Speaker 1"]
            speakers = ["Speaker 1"]
            
            # Format text for VibeVoice
            formatted_text = self._format_text_for_vibevoice(final_text, speakers)
            
            # Create or use voice samples
            voice_samples = self._prepare_voice_samples(speakers, voice_to_clone)
            
            # Generate audio using base class method
            audio_dict = self._generate_with_vibevoice(
                formatted_text, voice_samples, cfg_scale, seed, diffusion_steps, 
                use_sampling, temperature, top_p
            )
            
            # Free memory if requested
            if free_memory_after_generate:
                self.free_memory()
            
            return (audio_dict,)
                    
        except Exception as e:
            logger.error(f"Single speaker speech generation failed: {str(e)}")
            raise Exception(f"Error generating speech: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-1.5B", voice_to_clone=None, **kwargs):
        """Cache key for ComfyUI"""
        voice_hash = hash(str(voice_to_clone)) if voice_to_clone else 0
        return f"{hash(text)}_{model}_{voice_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"