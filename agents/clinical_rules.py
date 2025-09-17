from typing import Dict, Any, List, Optional, Tuple
import re


class ClinicalRuleEngineAgent:
    """
    Clinical Rule Engine Agent - Validates LLM diagnoses against hard-coded clinical criteria.
    
    This agent acts as a guardrail to prevent AI hallucinations by applying deterministic
    clinical rules to validate diagnoses against actual patient data.
    """
    
    def __init__(self) -> None:
        self.clinical_rules = self._initialize_clinical_rules()
    
    def validate_diagnoses(self, patient_data: Dict[str, Any], diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate LLM diagnoses against clinical rules and patient data.
        
        Args:
            patient_data: Structured patient data including vitals and labs
            diagnoses: List of diagnoses from AnalysisAgent
            
        Returns:
            List of validated diagnoses with corrected confidence scores
        """
        validated_diagnoses = []
        
        for diagnosis in diagnoses:
            diagnosis_name = diagnosis.get('name', '').lower()
            confidence = diagnosis.get('confidence', 0)
            evidence = diagnosis.get('evidence', '')
            
            # Apply clinical rules validation
            validation_result = self._validate_single_diagnosis(diagnosis_name, patient_data, evidence)
            
            if validation_result['is_valid']:
                # Keep diagnosis but potentially adjust confidence
                validated_diagnoses.append({
                    'name': diagnosis['name'],
                    'confidence': min(confidence, validation_result['max_confidence']),
                    'evidence': evidence,
                    'validation_status': 'VALIDATED',
                    'rule_engine_notes': validation_result['notes']
                })
            else:
                # Flag as invalid or remove
                if validation_result['severity'] == 'CRITICAL':
                    # Critical error - completely remove diagnosis
                    continue
                else:
                    # Warning - keep but flag and reduce confidence
                    validated_diagnoses.append({
                        'name': diagnosis['name'],
                        'confidence': max(0.1, confidence * 0.3),  # Drastically reduce confidence
                        'evidence': evidence,
                        'validation_status': 'FLAGGED',
                        'rule_engine_notes': f"⚠️ WARNING: {validation_result['reason']}"
                    })
        
        return validated_diagnoses
    
    def _validate_single_diagnosis(self, diagnosis_name: str, patient_data: Dict[str, Any], evidence: str) -> Dict[str, Any]:
        """Validate a single diagnosis against clinical rules."""
        
        # Extract vital signs and lab values from patient data
        vitals = self._extract_vitals(patient_data)
        labs = self._extract_labs(patient_data)
        
        # Check specific diagnosis rules
        if 'hypertension' in diagnosis_name:
            return self._validate_hypertension(vitals, evidence)
        elif 'hypotension' in diagnosis_name:
            return self._validate_hypotension(vitals, evidence)
        elif 'tachycardia' in diagnosis_name:
            return self._validate_tachycardia(vitals, evidence)
        elif 'bradycardia' in diagnosis_name:
            return self._validate_bradycardia(vitals, evidence)
        elif 'diabetes' in diagnosis_name or 'diabetic' in diagnosis_name:
            return self._validate_diabetes(labs, evidence)
        elif 'anemia' in diagnosis_name:
            return self._validate_anemia(labs, evidence)
        elif 'kidney' in diagnosis_name or 'renal' in diagnosis_name:
            return self._validate_kidney_disease(labs, evidence)
        else:
            # No specific rule - allow but note
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': 'No specific validation rule available',
                'notes': 'Diagnosis not validated by clinical rules engine'
            }
    
    def _validate_hypertension(self, vitals: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate hypertension diagnosis against BP values."""
        systolic = vitals.get('systolic_bp')
        diastolic = vitals.get('diastolic_bp')
        
        if systolic is None or diastolic is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Blood pressure values not available for hypertension diagnosis',
                'notes': 'Cannot validate hypertension without BP measurements'
            }
        
        # Hypertension criteria: SBP ≥ 140 OR DBP ≥ 90
        if systolic >= 140 or diastolic >= 90:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'BP {systolic}/{diastolic} meets hypertension criteria',
                'notes': f'Validated: BP {systolic}/{diastolic} ≥ 140/90'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'BP {systolic}/{diastolic} is NORMAL (not hypertensive). Normal range: <140/90',
                'notes': f'INVALID: BP {systolic}/{diastolic} does not meet hypertension criteria (≥140/90)'
            }
    
    def _validate_hypotension(self, vitals: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate hypotension diagnosis."""
        systolic = vitals.get('systolic_bp')
        
        if systolic is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Blood pressure values not available',
                'notes': 'Cannot validate hypotension without BP measurements'
            }
        
        # Hypotension criteria: SBP < 90
        if systolic < 90:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'SBP {systolic} meets hypotension criteria',
                'notes': f'Validated: SBP {systolic} < 90'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'SBP {systolic} is NORMAL (not hypotensive). Hypotension: <90 mmHg',
                'notes': f'INVALID: SBP {systolic} does not meet hypotension criteria (<90)'
            }
    
    def _validate_tachycardia(self, vitals: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate tachycardia diagnosis."""
        heart_rate = vitals.get('heart_rate')
        
        if heart_rate is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Heart rate not available',
                'notes': 'Cannot validate tachycardia without heart rate'
            }
        
        # Tachycardia criteria: HR > 100 bpm
        if heart_rate > 100:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'HR {heart_rate} bpm meets tachycardia criteria',
                'notes': f'Validated: HR {heart_rate} > 100 bpm'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'HR {heart_rate} bpm is NORMAL (not tachycardic). Tachycardia: >100 bpm',
                'notes': f'INVALID: HR {heart_rate} does not meet tachycardia criteria (>100)'
            }
    
    def _validate_bradycardia(self, vitals: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate bradycardia diagnosis."""
        heart_rate = vitals.get('heart_rate')
        
        if heart_rate is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Heart rate not available',
                'notes': 'Cannot validate bradycardia without heart rate'
            }
        
        # Bradycardia criteria: HR < 60 bpm
        if heart_rate < 60:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'HR {heart_rate} bpm meets bradycardia criteria',
                'notes': f'Validated: HR {heart_rate} < 60 bpm'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'HR {heart_rate} bpm is NORMAL (not bradycardic). Bradycardia: <60 bpm',
                'notes': f'INVALID: HR {heart_rate} does not meet bradycardia criteria (<60)'
            }
    
    def _validate_diabetes(self, labs: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate diabetes diagnosis."""
        glucose = labs.get('glucose')
        hba1c = labs.get('hba1c')
        
        if glucose is None and hba1c is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'No glucose or HbA1c values available',
                'notes': 'Cannot validate diabetes without glucose or HbA1c'
            }
        
        # Diabetes criteria: Glucose ≥ 126 mg/dL (fasting) or HbA1c ≥ 6.5%
        if glucose and glucose >= 126:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'Glucose {glucose} mg/dL meets diabetes criteria',
                'notes': f'Validated: Glucose {glucose} ≥ 126 mg/dL'
            }
        elif hba1c and hba1c >= 6.5:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'HbA1c {hba1c}% meets diabetes criteria',
                'notes': f'Validated: HbA1c {hba1c} ≥ 6.5%'
            }
        else:
            glucose_text = f"Glucose: {glucose} mg/dL" if glucose else "Glucose: N/A"
            hba1c_text = f"HbA1c: {hba1c}%" if hba1c else "HbA1c: N/A"
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'Lab values do not meet diabetes criteria. {glucose_text}, {hba1c_text}',
                'notes': f'INVALID: Neither glucose (≥126) nor HbA1c (≥6.5%) criteria met'
            }
    
    def _validate_anemia(self, labs: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate anemia diagnosis."""
        hemoglobin = labs.get('hemoglobin')
        
        if hemoglobin is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Hemoglobin value not available',
                'notes': 'Cannot validate anemia without hemoglobin'
            }
        
        # Anemia criteria: Hemoglobin < 12 g/dL (women) or < 13 g/dL (men)
        # Using conservative threshold of < 12 g/dL
        if hemoglobin < 12:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'Hemoglobin {hemoglobin} g/dL meets anemia criteria',
                'notes': f'Validated: Hemoglobin {hemoglobin} < 12 g/dL'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'Hemoglobin {hemoglobin} g/dL is NORMAL (not anemic). Anemia: <12 g/dL',
                'notes': f'INVALID: Hemoglobin {hemoglobin} does not meet anemia criteria (<12)'
            }
    
    def _validate_kidney_disease(self, labs: Dict[str, float], evidence: str) -> Dict[str, Any]:
        """Validate kidney disease diagnosis."""
        creatinine = labs.get('creatinine')
        
        if creatinine is None:
            return {
                'is_valid': False,
                'max_confidence': 0.5,
                'severity': 'WARNING',
                'reason': 'Creatinine value not available',
                'notes': 'Cannot validate kidney disease without creatinine'
            }
        
        # Kidney disease criteria: Creatinine > 1.3 mg/dL
        if creatinine > 1.3:
            return {
                'is_valid': True,
                'max_confidence': 1.0,
                'severity': 'INFO',
                'reason': f'Creatinine {creatinine} mg/dL suggests kidney dysfunction',
                'notes': f'Validated: Creatinine {creatinine} > 1.3 mg/dL'
            }
        else:
            return {
                'is_valid': False,
                'max_confidence': 0.0,
                'severity': 'CRITICAL',
                'reason': f'Creatinine {creatinine} mg/dL is NORMAL. Kidney disease: >1.3 mg/dL',
                'notes': f'INVALID: Creatinine {creatinine} does not suggest kidney disease (>1.3)'
            }
    
    def _extract_vitals(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract vital signs from patient data."""
        vitals = {}
        
        # Try to get vitals from structured data
        if patient_data.get('patient'):
            patient = patient_data['patient']
            vitals.update({
                'systolic_bp': patient.get('systolic_bp'),
                'diastolic_bp': patient.get('diastolic_bp'),
                'heart_rate': patient.get('heart_rate'),
                'respiratory_rate': patient.get('resp_rate'),
                'temperature': patient.get('temperature'),
                'spo2': patient.get('spo2')
            })
        
        # Also try to parse from vitals string
        vitals_text = patient_data.get('vitals', '')
        if vitals_text:
            parsed_vitals = self._parse_vitals_from_text(vitals_text)
            vitals.update(parsed_vitals)
        
        return {k: v for k, v in vitals.items() if v is not None}
    
    def _extract_labs(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract lab values from patient data."""
        labs = {}
        
        labs_text = patient_data.get('labs', '')
        if labs_text:
            labs = self._parse_labs_from_text(labs_text)
        
        return labs
    
    def _parse_vitals_from_text(self, vitals_text: str) -> Dict[str, float]:
        """Parse vital signs from text using regex."""
        vitals = {}
        
        # Blood pressure patterns
        bp_match = re.search(r'BP:?\s*(\d+)/(\d+)', vitals_text, re.IGNORECASE)
        if bp_match:
            vitals['systolic_bp'] = float(bp_match.group(1))
            vitals['diastolic_bp'] = float(bp_match.group(2))
        
        # Heart rate patterns
        hr_match = re.search(r'HR:?\s*(\d+)', vitals_text, re.IGNORECASE)
        if hr_match:
            vitals['heart_rate'] = float(hr_match.group(1))
        
        # Temperature patterns
        temp_match = re.search(r'Temp:?\s*(\d+\.?\d*)', vitals_text, re.IGNORECASE)
        if temp_match:
            vitals['temperature'] = float(temp_match.group(1))
        
        # SpO2 patterns
        spo2_match = re.search(r'SpO2:?\s*(\d+)', vitals_text, re.IGNORECASE)
        if spo2_match:
            vitals['spo2'] = float(spo2_match.group(1))
        
        return vitals
    
    def _parse_labs_from_text(self, labs_text: str) -> Dict[str, float]:
        """Parse lab values from text using regex."""
        labs = {}
        
        # Glucose patterns
        glucose_match = re.search(r'Glucose:?\s*(\d+\.?\d*)', labs_text, re.IGNORECASE)
        if glucose_match:
            labs['glucose'] = float(glucose_match.group(1))
        
        # Hemoglobin patterns
        hgb_match = re.search(r'Hemoglobin:?\s*(\d+\.?\d*)', labs_text, re.IGNORECASE)
        if hgb_match:
            labs['hemoglobin'] = float(hgb_match.group(1))
        
        # Creatinine patterns
        creat_match = re.search(r'Creatinine:?\s*(\d+\.?\d*)', labs_text, re.IGNORECASE)
        if creat_match:
            labs['creatinine'] = float(creat_match.group(1))
        
        # HbA1c patterns
        hba1c_match = re.search(r'HbA1c:?\s*(\d+\.?\d*)', labs_text, re.IGNORECASE)
        if hba1c_match:
            labs['hba1c'] = float(hba1c_match.group(1))
        
        return labs
    
    def _initialize_clinical_rules(self) -> Dict[str, Any]:
        """Initialize clinical decision rules."""
        return {
            'hypertension': {
                'criteria': 'SBP ≥ 140 OR DBP ≥ 90',
                'thresholds': {'systolic': 140, 'diastolic': 90}
            },
            'hypotension': {
                'criteria': 'SBP < 90',
                'thresholds': {'systolic': 90}
            },
            'tachycardia': {
                'criteria': 'HR > 100 bpm',
                'thresholds': {'heart_rate': 100}
            },
            'bradycardia': {
                'criteria': 'HR < 60 bpm',
                'thresholds': {'heart_rate': 60}
            },
            'diabetes': {
                'criteria': 'Glucose ≥ 126 mg/dL OR HbA1c ≥ 6.5%',
                'thresholds': {'glucose': 126, 'hba1c': 6.5}
            },
            'anemia': {
                'criteria': 'Hemoglobin < 12 g/dL',
                'thresholds': {'hemoglobin': 12}
            },
            'kidney_disease': {
                'criteria': 'Creatinine > 1.3 mg/dL',
                'thresholds': {'creatinine': 1.3}
            }
        }
