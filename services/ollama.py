import json
import requests
import os
from typing import Any, Dict, Optional
from cachetools import TTLCache

class OllamaReasoner:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.1:8b", base_url: Optional[str] = None) -> None:
        self.model = model
        # Allow configuration via environment variable or parameter
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache(maxsize=128, ttl=600)

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        if prompt in self.cache:
            return self.cache[prompt]["text"]

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120  # Increased timeout for model loading
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "")
            self.cache[prompt] = {"text": text}
            return text
        except requests.exceptions.Timeout:
            raise Exception("Ollama model is loading. Please wait a moment and try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve'")
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")

    def generate_json(self, prompt: str, schema_description: str) -> Dict[str, Any]:
        """Ask Ollama to return strict JSON using its JSON mode."""
        json_system_instructions = (
            "You are a JSON generator. Always return a single valid JSON object that conforms to the schema description. "
            "Do not include markdown fences, explanations, or extra text."
        )
        full_prompt = (
            f"{json_system_instructions}\n\nSchema: {schema_description}\n\nTask:\n{prompt}"
        )
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "format": "json",  # Enforce JSON output
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            out = resp.json().get("response", "").strip()
            return json.loads(out)
        except requests.exceptions.Timeout:
            return {"error": "timeout", "message": "Model timed out generating JSON. Try again shortly."}
        except requests.exceptions.ConnectionError:
            return {"error": "connection", "message": "Cannot connect to Ollama. Ensure 'ollama serve' is running."}
        except Exception as e:
            return {"error": "invalid_json", "raw": str(e)}
