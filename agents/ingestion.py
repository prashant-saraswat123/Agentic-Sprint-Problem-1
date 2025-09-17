from typing import Dict, Any


class IngestionAgent:
	def run(self, raw_input: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
		structured = {
			"patient": {
				"age": raw_input.get("demographics", {}).get("age"),
				"sex": raw_input.get("demographics", {}).get("sex"),
			},
			"symptom_notes": raw_input.get("free_text", ""),
			"vitals": raw_input.get("vitals", ""),
			"labs": raw_input.get("labs", ""),  # Explicitly include labs field
			"medications": raw_input.get("medications", ""),
			"imaging": raw_input.get("imaging", ""),
		}
		session.setdefault("events", []).append({
			"agent": "Ingestion",
			"action": "normalize",
			"details": "Structured patient record created",
		})
		return structured
