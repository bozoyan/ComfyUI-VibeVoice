# Created by Fabio Sarracino

import os
import logging
import hashlib
import folder_paths

# Setup logging
logger = logging.getLogger("VibeVoice")

class LoadTextFromFileNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Get all text files from all directories
        all_files = []
        
        # Add files from each directory with prefix
        for dir_name in ["input", "output", "temp"]:
            files = cls.get_files_for_directory(dir_name)
            for f in files:
                if f != "No text files found":
                    all_files.append(f"{dir_name}/{f}")
        
        if not all_files:
            all_files = ["No text files found in any directory"]
        
        return {
            "required": {
                "file": (sorted(all_files), {
                    "tooltip": "Select a text file to load (format: directory/filename)"
                }),
            }
        }
    
    @classmethod
    def get_files_for_directory(cls, source_dir):
        """Get list of text files for the selected directory"""
        # Get the appropriate directory path
        if source_dir == "input":
            dir_path = folder_paths.get_input_directory()
        elif source_dir == "output":
            dir_path = folder_paths.get_output_directory()
        elif source_dir == "temp":
            dir_path = folder_paths.get_temp_directory()
        else:
            return []
        
        files = []
        try:
            for f in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, f)):
                    # Check for text file extensions
                    if f.lower().endswith(('.txt')):
                        files.append(f)
        except Exception as e:
            logger.warning(f"Error listing files in {source_dir}: {e}")
            
        return files

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_text"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Load text content from a .txt file"

    def load_text(self, file: str):
        """Load text content from file"""
        
        try:
            # Check if no file selected
            if not file or file == "No text files found in any directory":
                raise Exception("Please select a valid text file.")
            
            # Parse directory and filename from the combined string
            if "/" not in file:
                raise Exception(f"Invalid file format: {file}")
            
            source_dir, filename = file.split("/", 1)
            
            # Get the appropriate directory path
            if source_dir == "input":
                dir_path = folder_paths.get_input_directory()
            elif source_dir == "output":
                dir_path = folder_paths.get_output_directory()
            elif source_dir == "temp":
                dir_path = folder_paths.get_temp_directory()
            else:
                raise Exception(f"Invalid source directory: {source_dir}")
            
            # Build full file path
            file_path = os.path.join(dir_path, filename)
            
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")
            
            # Read file with UTF-8 encoding (most common)
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                raise Exception("File is empty or contains only whitespace")
            
            return (text_content,)
            
        except UnicodeDecodeError as e:
            raise Exception(f"Encoding error reading file: {str(e)}. File may not be UTF-8 encoded.")
        except Exception as e:
            logger.error(f"Failed to load text file: {str(e)}")
            raise Exception(f"Error loading text file: {str(e)}")

    @classmethod
    def IS_CHANGED(cls, file):
        """Cache key for ComfyUI"""
        if not file or file == "No text files found in any directory":
            return "no_file"
        
        # Parse directory and filename
        if "/" not in file:
            return f"{file}_invalid"
        
        source_dir, filename = file.split("/", 1)
        
        # Get the appropriate directory path
        if source_dir == "input":
            dir_path = folder_paths.get_input_directory()
        elif source_dir == "output":
            dir_path = folder_paths.get_output_directory()
        elif source_dir == "temp":
            dir_path = folder_paths.get_temp_directory()
        else:
            return f"{file}_invalid_dir"
        
        file_path = os.path.join(dir_path, filename)
        
        if not os.path.exists(file_path):
            return f"{file}_not_found"
        
        # Use file hash for cache invalidation
        try:
            m = hashlib.sha256()
            with open(file_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except:
            return f"{file}_error"
    
    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
        """Validate that the file exists"""
        if not file or file == "No text files found in any directory":
            return "No valid text file selected"
        
        # Parse directory and filename
        if "/" not in file:
            return f"Invalid file format: {file}"
        
        source_dir, filename = file.split("/", 1)
        
        # Get the appropriate directory path
        if source_dir == "input":
            dir_path = folder_paths.get_input_directory()
        elif source_dir == "output":
            dir_path = folder_paths.get_output_directory()
        elif source_dir == "temp":
            dir_path = folder_paths.get_temp_directory()
        else:
            return f"Invalid source directory: {source_dir}"
        
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            return f"File not found: {filename} in {source_dir}"
        
        return True