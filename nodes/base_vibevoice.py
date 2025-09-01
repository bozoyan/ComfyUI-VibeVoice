# Created by Fabio Sarracino
# Base class for VibeVoice nodes with common functionality

import logging
import os
import tempfile
import torch
import numpy as np
import re
from typing import List, Optional, Tuple, Any

# Setup logging
logger = logging.getLogger("VibeVoice")

def detect_directml_device():
    """Detect if DirectML is available and return appropriate device"""
    try:
        # Check if torch-directml is available
        import torch_directml
        if torch_directml.is_available():
            device_count = torch_directml.device_count()
            if device_count > 0:
                logger.info(f"DirectML detected with {device_count} device(s)")
                return torch_directml.device(), True
    except ImportError:
        logger.info("torch-directml not available")
    except Exception as e:
        logger.warning(f"DirectML detection failed: {e}")
    
    # Fallback to CUDA if available
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda"), False
    
    # Fallback to CPU
    logger.info("Using CPU device")
    return torch.device("cpu"), False

class BaseVibeVoiceNode:
    """Base class for VibeVoice nodes containing common functionality"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None
        self.current_attention_type = None
        # Detect DirectML device
        self.device, self.is_directml = detect_directml_device()
        logger.info(f"Initialized with device: {self.device}, DirectML: {self.is_directml}")
    
    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            self.current_model_path = None
            
            # Force garbage collection and clear cache
            import gc
            gc.collect()
            
            # Clear appropriate cache based on device type
            if self.is_directml:
                try:
                    import torch_directml
                    # DirectML doesn't have explicit cache clearing, but force GC
                    torch_directml.empty_cache() if hasattr(torch_directml, 'empty_cache') else None
                except:
                    pass
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import vibevoice
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            return vibevoice, VibeVoiceForConditionalGenerationInference
            
        except ImportError as e:
            # Try to install and import again
            try:
                import subprocess
                import sys
                
                # First ensure compatible transformers version
                transformers_cmd = [sys.executable, "-m", "pip", "install", "transformers>=4.44.0"]
                subprocess.run(transformers_cmd, capture_output=True, text=True, timeout=300)
                
                cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/microsoft/VibeVoice.git"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Force reload of sys.modules to pick up new installation
                    import importlib
                    if 'vibevoice' in sys.modules:
                        importlib.reload(sys.modules['vibevoice'])
                    
                    # Try import again
                    import vibevoice
                    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                    return vibevoice, VibeVoiceForConditionalGenerationInference
                    
            except Exception as install_error:
                logger.error(f"Installation attempt failed: {install_error}")
            
            logger.error(f"VibeVoice import failed: {e}")
            raise Exception(
                "VibeVoice installation/import failed. Please restart ComfyUI completely, "
                "or install manually with: pip install transformers>=4.44.0 && pip install git+https://github.com/microsoft/VibeVoice.git"
            )
    
    def load_model(self, model_path: str, attention_type: str = "auto"):
        """Load VibeVoice model with specified attention implementation"""
        # Check if we need to reload model due to attention type change
        current_attention = getattr(self, 'current_attention_type', None)
        if (self.model is None or 
            getattr(self, 'current_model_path', None) != model_path or
            current_attention != attention_type):
            try:
                vibevoice, VibeVoiceInferenceModel = self._check_dependencies()
                
                # Set ComfyUI models directory
                import folder_paths
                models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                comfyui_models_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
                os.makedirs(comfyui_models_dir, exist_ok=True)
                
                # Force HuggingFace to use ComfyUI directory
                original_hf_home = os.environ.get('HF_HOME')
                original_hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
                
                os.environ['HF_HOME'] = comfyui_models_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = comfyui_models_dir
                
                # Import time for timing
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Check if model exists locally
                model_dir = os.path.join(comfyui_models_dir, f"models--{model_path.replace('/', '--')}")
                model_exists_in_comfyui = os.path.exists(model_dir)
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "cache_dir": comfyui_models_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                }
                
                # Set device mapping based on available hardware
                if self.is_directml:
                    # For DirectML, don't use device_map, we'll move manually
                    logger.info("Loading model for DirectML device")
                elif torch.cuda.is_available():
                    model_kwargs["device_map"] = "cuda"
                    logger.info("Loading model for CUDA device")
                else:
                    model_kwargs["device_map"] = "cpu"
                    logger.info("Loading model for CPU device")
                
                # Set attention implementation based on user selection
                if attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    # Auto mode - let transformers decide the best implementation
                    logger.info("Using auto attention implementation selection")
                
                # Try to load locally first
                try:
                    if model_exists_in_comfyui:
                        model_kwargs["local_files_only"] = True
                        self.model = VibeVoiceInferenceModel.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                    else:
                        raise FileNotFoundError("Model not found locally")
                except (FileNotFoundError, OSError) as e:
                    logger.info(f"Downloading {model_path}...")
                    
                    model_kwargs["local_files_only"] = False
                    self.model = VibeVoiceInferenceModel.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    elapsed = time.time() - start_time
                else:
                    elapsed = time.time() - start_time
                
                # Load processor with proper error handling
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
                try:
                    # First try with local files if model was loaded locally
                    if model_exists_in_comfyui:
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path, 
                            local_files_only=True,
                            trust_remote_code=True,
                            cache_dir=comfyui_models_dir
                        )
                    else:
                        # Download from HuggingFace
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            cache_dir=comfyui_models_dir
                        )
                except Exception as proc_error:
                    logger.warning(f"Failed to load processor from {model_path}: {proc_error}")
                    
                    # Check if error is about missing Qwen tokenizer
                    if "Qwen" in str(proc_error) and "tokenizer" in str(proc_error).lower():
                        logger.info("Downloading required Qwen tokenizer files...")
                        # The processor needs the Qwen tokenizer, ensure it's available
                        try:
                            from transformers import AutoTokenizer
                            # Pre-download the Qwen tokenizer that VibeVoice depends on
                            _ = AutoTokenizer.from_pretrained(
                                "Qwen/Qwen2.5-1.5B",
                                trust_remote_code=True,
                                cache_dir=comfyui_models_dir
                            )
                            logger.info("Qwen tokenizer downloaded, retrying processor load...")
                        except Exception as tokenizer_error:
                            logger.warning(f"Failed to download Qwen tokenizer: {tokenizer_error}")
                    
                    logger.info("Attempting to load processor with fallback method...")
                    
                    # Fallback: try loading without local_files_only constraint
                    try:
                        self.processor = VibeVoiceProcessor.from_pretrained(
                            model_path,
                            local_files_only=False,
                            trust_remote_code=True,
                            cache_dir=comfyui_models_dir
                        )
                    except Exception as fallback_error:
                        logger.error(f"Processor loading failed completely: {fallback_error}")
                        raise Exception(
                            f"Failed to load VibeVoice processor. Error: {fallback_error}\n"
                            f"This might be due to missing tokenizer files. Try:\n"
                            f"1. Ensure you have internet connection for first-time download\n"
                            f"2. Clear the ComfyUI/models/vibevoice folder and retry\n"
                            f"3. Install transformers: pip install transformers>=4.44.0\n"
                            f"4. Manually download Qwen tokenizer: from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')"
                        )
                
                # Restore environment variables
                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']
                    
                if original_hf_cache is not None:
                    os.environ['HUGGINGFACE_HUB_CACHE'] = original_hf_cache
                elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
                    del os.environ['HUGGINGFACE_HUB_CACHE']
                
                # Move to appropriate device
                if self.is_directml:
                    # For DirectML, move model to DirectML device
                    try:
                        self.model = self.model.to(self.device)
                        logger.info(f"Model moved to DirectML device: {self.device}")
                    except Exception as e:
                        logger.warning(f"Failed to move model to DirectML, using CPU: {e}")
                        self.device = torch.device("cpu")
                        self.is_directml = False
                        self.model = self.model.to(self.device)
                elif torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("Model moved to CUDA device")
                else:
                    logger.info("Model remains on CPU device")
                    
                self.current_model_path = model_path
                self.current_attention_type = attention_type
                
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)
    
    def _prepare_audio_from_comfyui(self, voice_audio, target_sample_rate: int = 24000) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format to numpy array"""
        if voice_audio is None:
            return None
            
        # Extract waveform from ComfyUI audio format
        if isinstance(voice_audio, dict) and "waveform" in voice_audio:
            waveform = voice_audio["waveform"]
            input_sample_rate = voice_audio.get("sample_rate", target_sample_rate)
            
            # Convert to numpy
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different audio shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0, 0, :]  # Take first batch, first channel
            elif audio_np.ndim == 2:  # (channels, samples)
                audio_np = audio_np[0, :]  # Take first channel
            # If 1D, leave as is
            
            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length), 
                                   np.arange(len(audio_np)), audio_np)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)  # Normalize
            
            return audio_np.astype(np.float32)
        
        return None
    
    def _get_model_mapping(self) -> dict:
        """Get model name mappings"""
        return {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-7B-Preview": "WestZhang/VibeVoice-Large-pt"
        }
    
    def _format_text_for_vibevoice(self, text: str, speakers: list) -> str:
        """Format text with speaker information for VibeVoice using correct format"""
        # Remove any newlines from the text to prevent parsing issues
        # The processor splits by newline and expects each line to have "Speaker N:" format
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # VibeVoice expects format: "Speaker 1: text" not "Name: text"
        if len(speakers) == 1:
            return f"Speaker 1: {text}"
        else:
            # Check if text already has proper Speaker N: format
            if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
                return text
            # If text has name format, convert to Speaker N format
            elif any(f"{speaker}:" in text for speaker in speakers):
                formatted_text = text
                for i, speaker in enumerate(speakers):
                    formatted_text = formatted_text.replace(f"{speaker}:", f"Speaker {i+1}:")
                return formatted_text
            else:
                # Plain text, assign to first speaker
                return f"Speaker 1: {text}"
    
    def _generate_with_vibevoice(self, formatted_text: str, voice_samples: List[np.ndarray], 
                                cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                                temperature: float = 0.95, top_p: float = 0.95) -> dict:
        """Generate audio using VibeVoice model"""
        try:
            # Ensure model and processor are loaded
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            
            # Set device-specific seeds
            if self.is_directml:
                try:
                    import torch_directml
                    torch_directml.manual_seed(seed)
                except:
                    pass
            elif torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # Also set numpy seed for any numpy operations
            np.random.seed(seed)
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            logger.info(f"Starting audio generation with {diffusion_steps} diffusion steps...")
            
            # Prepare inputs using processor
            inputs = self.processor(
                [formatted_text],  # Wrap text in list
                voice_samples=[voice_samples], # Provide voice samples for reference
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            device = self.device  # Use the detected device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Estimate tokens for user information (not used as limit)
            text_length = len(formatted_text.split())
            estimated_tokens = text_length * 2  # More accurate estimate for display
            
            # Log generation start with explanation
            logger.info(f"Generating audio with {diffusion_steps} diffusion steps...")
            logger.info(f"Note: Progress bar shows max possible tokens, not actual needed (~{estimated_tokens} estimated)")
            logger.info("The generation will stop automatically when audio is complete")
            
            # Generate with official parameters
            with torch.no_grad():
                if use_sampling:
                    # Use sampling mode (less stable but more varied)
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    # Use deterministic mode like official examples
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=None,
                        do_sample=False,  # More deterministic generation
                    )
                
                # Check if we got actual audio output
                if hasattr(output, 'speech_outputs') and output.speech_outputs:
                    speech_tensors = output.speech_outputs
                    
                    if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                        audio_tensor = torch.cat(speech_tensors, dim=-1)
                    else:
                        audio_tensor = speech_tensors
                    
                    # Ensure proper format (1, 1, samples)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.dim() == 2:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    return {
                        "waveform": audio_tensor.cpu(),
                        "sample_rate": 24000
                    }
                    
                elif hasattr(output, 'sequences'):
                    logger.error("VibeVoice returned only text tokens, no audio generated")
                    raise Exception("VibeVoice failed to generate audio - only text tokens returned")
                    
                else:
                    logger.error(f"Unexpected output format from VibeVoice: {type(output)}")
                    raise Exception(f"VibeVoice returned unexpected output format: {type(output)}")
                
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            raise Exception(f"VibeVoice generation failed: {str(e)}")