import os
import tempfile
from typing import Dict, Any, Optional

# Whisper imports with comprehensive error handling
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper loaded successfully")
except ImportError as e:
    whisper = None
    WHISPER_AVAILABLE = False
    print(f"❌ Whisper import error: {e}")
except OSError as e:
    # Handle PyTorch/Torch DLL loading issues on Windows
    whisper = None
    WHISPER_AVAILABLE = False
    print(f"❌ Whisper/PyTorch DLL loading error: {e}")
    print("💡 This is likely a PyTorch installation issue. Try: pip uninstall torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
except Exception as e:
    whisper = None
    WHISPER_AVAILABLE = False
    print(f"❌ Unexpected Whisper loading error: {e}")


class WhisperService:
    """
    Service for audio transcription using OpenAI Whisper with medical terminology optimization.
    Provides lazy loading and error handling for voice note processing.
    """
    
    def __init__(self, model_size: str = "base") -> None:
        """
        Initialize WhisperService with specified model size.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self._model = None
        self._model_loaded = False
        
    def _load_model(self) -> bool:
        """
        Lazy load the Whisper model to avoid startup delays.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True
            
        if not WHISPER_AVAILABLE:
            return False
            
        try:
            print(f"🔄 Loading Whisper model: {self.model_size}")
            self._model = whisper.load_model(self.model_size)
            self._model_loaded = True
            print(f"✅ Whisper model loaded successfully: {self.model_size}")
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def transcribe_with_medical_context(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Transcribe audio with medical terminology optimization.
        
        Args:
            audio_bytes: Raw audio data in bytes
            
        Returns:
            Dict containing transcription results with success status
        """
        if not self._load_model():
            return {
                "success": False,
                "error": "Whisper model not available. Please install openai-whisper: pip install openai-whisper",
                "text": "",
                "language": "unknown"
            }
        
        try:
            # Create temporary file for audio processing with Windows-safe handling
            temp_file_path = None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_file_path = temp_file.name
                print(f"🎵 Created temporary audio file: {temp_file_path}")
            
            # Verify file exists before processing
            if not os.path.exists(temp_file_path):
                return {
                    "success": False,
                    "error": f"Temporary audio file not found: {temp_file_path}",
                    "text": "",
                    "language": "unknown"
                }
            
            print(f"📁 File size: {os.path.getsize(temp_file_path)} bytes")
            
            # Ensure file is closed before processing (Windows requirement)
            try:
                # Transcribe with medical-optimized parameters
                print(f"🎯 Starting Whisper transcription of: {temp_file_path}")
                result = self._model.transcribe(
                    temp_file_path,
                    language="en",  # Optimize for English medical terminology
                    task="transcribe",
                    temperature=0.0,  # Deterministic output for medical accuracy
                    best_of=1,
                    beam_size=1,
                    patience=1.0,
                    length_penalty=1.0,
                    suppress_tokens=[-1],  # Don't suppress any tokens
                    initial_prompt="Medical consultation recording with clinical terminology, laboratory results, medications, and diagnostic findings."
                )
                
                # Extract and clean transcription
                transcription = result.get("text", "").strip()
                language = result.get("language", "en")
                
                # Post-process for medical terminology
                transcription = self._enhance_medical_terminology(transcription)
                
                return {
                    "success": True,
                    "text": transcription,
                    "language": language,
                    "error": None
                }
                
            finally:
                # Clean up temp file with Windows-safe deletion
                try:
                    if temp_file_path and os.path.exists(temp_file_path):
                        # Give Windows time to release file handles
                        import time
                        time.sleep(0.1)
                        os.unlink(temp_file_path)
                except (OSError, PermissionError) as cleanup_error:
                    print(f"⚠️ Could not delete temp audio file {temp_file_path}: {cleanup_error}")
                    # File will be cleaned up by system temp cleanup eventually
                        
        except Exception as e:
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}",
                "text": "",
                "language": "unknown"
            }
    
    def _enhance_medical_terminology(self, text: str) -> str:
        """
        Enhance transcription by correcting common medical terminology errors.
        
        Args:
            text: Raw transcription text
            
        Returns:
            str: Enhanced text with corrected medical terms
        """
        # Common medical term corrections
        medical_corrections = {
            # Laboratory terms
            "hemoglobin": "hemoglobin",
            "hematocrit": "hematocrit", 
            "white blood cell": "white blood cell",
            "wbc": "WBC",
            "rbc": "RBC",
            "platelet": "platelet",
            "glucose": "glucose",
            "creatinine": "creatinine",
            "bun": "BUN",
            "cholesterol": "cholesterol",
            "triglyceride": "triglyceride",
            "hdl": "HDL",
            "ldl": "LDL",
            
            # Vital signs
            "blood pressure": "blood pressure",
            "heart rate": "heart rate",
            "temperature": "temperature",
            "respiratory rate": "respiratory rate",
            "oxygen saturation": "oxygen saturation",
            
            # Common medications
            "metformin": "metformin",
            "lisinopril": "lisinopril",
            "atorvastatin": "atorvastatin",
            "amlodipine": "amlodipine",
            "omeprazole": "omeprazole",
            
            # Medical conditions
            "diabetes": "diabetes",
            "hypertension": "hypertension",
            "hyperlipidemia": "hyperlipidemia",
            "anemia": "anemia",
            "kidney disease": "kidney disease",
            
            # Units
            "milligrams per deciliter": "mg/dL",
            "grams per deciliter": "g/dL",
            "millimeters of mercury": "mmHg",
            "beats per minute": "bpm",
            "degrees fahrenheit": "°F",
            "degrees celsius": "°C"
        }
        
        # Apply corrections (case-insensitive)
        enhanced_text = text
        for incorrect, correct in medical_corrections.items():
            enhanced_text = enhanced_text.replace(incorrect.lower(), correct)
            enhanced_text = enhanced_text.replace(incorrect.title(), correct)
            enhanced_text = enhanced_text.replace(incorrect.upper(), correct)
        
        return enhanced_text
    
    def is_available(self) -> bool:
        """
        Check if Whisper service is available.
        
        Returns:
            bool: True if Whisper is available and can be loaded
        """
        return WHISPER_AVAILABLE and self._load_model()
