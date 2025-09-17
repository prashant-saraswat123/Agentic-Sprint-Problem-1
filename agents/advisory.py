from typing import Dict, Any, List

from services.ollama import OllamaReasoner


class AdvisoryAgent:
	def __init__(self, reasoner: OllamaReasoner) -> None:
		self.reasoner = reasoner

	def run(
		self,
		record: Dict[str, Any],
		diagnoses: List[Dict[str, Any]],
		flags: List[Dict[str, Any]],
		session: Dict[str, Any],
	) -> Dict[str, Any]:
		prompt = self._build_prompt(record, diagnoses, flags)
		response = self.reasoner.generate_json(
			prompt,
			schema_description=(
				"{""summary"": str, ""recommendations"": [str], ""next_steps"": [str], ""trace"": {""data_links"": [str]} }"
			),
		)
		session.setdefault("events", []).append({
			"agent": "Advisory",
			"action": "advise",
			"details": "Generated explainable report",
		})
		return response

	def _build_prompt(self, record: Dict[str, Any], diagnoses: List[Dict[str, Any]], flags: List[Dict[str, Any]]) -> str:
		return (
			"Create a comprehensive, structured clinical advisory for healthcare professionals. "
			"Provide detailed, actionable recommendations with clear rationale and evidence-based reasoning.\n\n"
			"CLINICAL ADVISORY STRUCTURE:\n"
			"1. SUMMARY: Concise clinical overview integrating patient presentation, key findings, and diagnostic conclusions\n"
			"2. RECOMMENDATIONS: Detailed, prioritized clinical recommendations with specific rationale for each\n"
			"3. NEXT STEPS: Actionable immediate and follow-up steps including:\n"
			"   - Additional diagnostic tests needed\n"
			"   - Treatment considerations\n"
			"   - Monitoring requirements\n"
			"   - Specialist referrals if indicated\n"
			"   - Patient education points\n"
			"4. DATA TRACEABILITY: Link each recommendation to specific evidence (lab values, symptoms, risk factors)\n\n"
			"Make recommendations comprehensive, specific, and clinically relevant. Include timeframes where appropriate.\n"
			"Return structured JSON with keys: summary, recommendations, next_steps, trace.data_links\n\n"
			f"Patient Record: {record}\n"
			f"Diagnosed Conditions: {diagnoses}\n"
			f"Risk Flags: {flags}\n"
		)