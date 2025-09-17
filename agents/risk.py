from typing import Dict, Any, List

from services.ollama import OllamaReasoner


class RiskAgent:
	def __init__(self, reasoner: OllamaReasoner) -> None:
		self.reasoner = reasoner

	def run(self, record: Dict[str, Any], diagnoses: List[Dict[str, Any]], session: Dict[str, Any]) -> List[Dict[str, Any]]:
		prompt = self._build_prompt(record, diagnoses)
		response = self.reasoner.generate_json(
			prompt,
			schema_description="{""flags"": [{""name"": str, ""urgency"": str, ""rationale"": str}]}",
		)
		flags: List[Dict[str, Any]] = response.get("flags", [])
		session.setdefault("events", []).append({
			"agent": "Risk",
			"action": "detect_flags",
			"details": f"Detected {len(flags)} red flags",
		})
		return flags

	def _build_prompt(self, record: Dict[str, Any], diagnoses: List[Dict[str, Any]]) -> str:
		return (
			"You are a clinical triage AI. Your sole task is to identify and flag clinically urgent conditions based on the provided patient data.\n\n"
			"CRITICAL INSTRUCTIONS:\n"
			"1. NO FABRICATION: If no data points meet the criteria for a risk flag, you MUST return an empty 'flags' list. Do not invent risks.\n"
			"2. STRICT LOGIC: Base flags on values that are CLEARLY AND SIGNIFICANTLY outside the normal range. Do not flag values that are at or near the normal limit.\n"
			"3. EVIDENCE-BASED RATIONALE: Your rationale MUST cite the specific data point (e.g., 'Fasting Glucose of 141 mg/dL') from the record.\n"
			"4. USE DEFINED URGENCY LEVELS:\n"
			"   - 'High': Conditions requiring urgent attention (e.g., severe hyperglycemia, SBP > 180).\n"
			"   - 'Medium': Serious conditions that need prompt attention (e.g., uncontrolled chronic diseases).\n"
			"   - 'Low': Important but not urgent findings (e.g., vitamin deficiencies).\n\n"
			"Return ONLY a valid JSON object with the key 'flags'.\n\n"
			"---BEGIN DATA---\n"
			f"Record: {record}\n"
			f"Diagnoses: {diagnoses}\n"
			"---END DATA---\n"
		)