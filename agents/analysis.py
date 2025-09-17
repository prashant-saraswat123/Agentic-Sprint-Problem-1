from typing import Dict, Any, List

from services.ollama import OllamaReasoner


class AnalysisAgent:
	def __init__(self, reasoner: OllamaReasoner) -> None:
		self.reasoner = reasoner

	def run(self, record: Dict[str, Any], session: Dict[str, Any]) -> List[Dict[str, Any]]:
		prompt = self._build_prompt(record)
		response = self.reasoner.generate_json(
			prompt,
			schema_description="{""diagnoses"": [{""name"": str, ""confidence"": float, ""evidence"": str}]}",
		)
		diagnoses: List[Dict[str, Any]] = response.get("diagnoses", [])
		session.setdefault("events", []).append({
			"agent": "Analysis",
			"action": "rank_diagnoses",
			"details": f"Returned {len(diagnoses)} candidates",
		})
		return diagnoses

	def _build_prompt(self, record: Dict[str, Any]) -> str:
		return (
			"You are an expert clinical pathologist. Your task is to analyze the provided patient's laboratory report, identify all abnormal values, and generate a list of potential diagnoses based exclusively on that data.\n\n"
			"CRITICAL REQUIREMENTS:\n"
			"1. STRICT DATA ADHERENCE: You MUST ONLY use the data provided in the 'Patient Record' section below. Do NOT use any outside knowledge or examples from this prompt.\n"
			"2. EVIDENCE-BASED DIAGNOSIS: For each diagnosis, you MUST cite the specific abnormal lab value, its units, and the provided normal range from the report as evidence.\n"
			"3. STRICT COMPARISON LOGIC: A value is ONLY abnormal if it is strictly greater than the upper limit or strictly less than the lower limit of the normal range. Values that are equal to the limits of the normal range are considered NORMAL.\n"
			"4. NO HALLUCINATION: Do NOT invent or hallucinate any values. If a value is not in the record, you cannot use it.\n"
			"5. CONFIDENCE SCORE: Assign a confidence score (from 0.0 to 1.0) based on how strongly the lab data supports the diagnosis.\n\n"
			"OUTPUT FORMAT:\n"
			"Return a single valid JSON object with the key 'diagnoses'.\n\n"
			"---BEGIN PATIENT RECORD---\n"
			f"Demographics: {record.get('patient')}\n"
			f"Labs: {record.get('labs')}\n"
			f"Vitals: {record.get('vitals')}\n"
			f"Symptoms/Notes: {record.get('symptom_notes')}\n"
			"---END PATIENT RECORD---\n"
		)