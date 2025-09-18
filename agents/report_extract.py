from typing import Dict, Any, Optional, List, Tuple
import io
import tempfile
import os

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Enhanced PDF processing imports
try:
    import pdfplumber
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_AVAILABLE = True
    ENHANCED_PDF_AVAILABLE = True
    print("✅ LangChain and pdfplumber dependencies loaded successfully")
except ImportError as e:
    pdfplumber = None
    PyPDFLoader = None
    LANGCHAIN_AVAILABLE = False
    ENHANCED_PDF_AVAILABLE = False
    print(f"❌ LangChain/pdfplumber import error: {e}")


from services.ollama import OllamaReasoner


class ReportExtractAgent:
	def __init__(self, reasoner: OllamaReasoner) -> None:
		self.reasoner = reasoner

	def extract_from_pdf(self, pdf_bytes: bytes, progress_callback=None) -> Dict[str, Any]:
		"""Robust multi-pass PDF extraction with parallel processing and result merging."""
		print(f"🔍 Starting robust PDF extraction. LANGCHAIN_AVAILABLE: {LANGCHAIN_AVAILABLE}, ENHANCED_PDF_AVAILABLE: {ENHANCED_PDF_AVAILABLE}")
		
		try:
			# Create temporary file for processing
			temp_file_path = None
			with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
				temp_file.write(pdf_bytes)
				temp_file.flush()
				temp_file_path = temp_file.name
				print(f"📄 Created temporary PDF file: {temp_file_path}")
			
			try:
				# Multi-pass extraction approach
				extraction_results = []
				
				# Pass 1: LangChain PyPDFLoader (structured text)
				if LANGCHAIN_AVAILABLE and PyPDFLoader:
					print("🚀 Pass 1: LangChain PyPDFLoader extraction...")
					if progress_callback:
						progress_callback("pass1", "running", "LangChain PyPDFLoader extraction in progress...")
					try:
						loader = PyPDFLoader(temp_file_path)
						documents = loader.load()
						total_text = "\n".join([doc.page_content for doc in documents if doc.page_content])
						if len(total_text.strip()) > 50:  # Lower threshold for partial success
							result = self._process_langchain_documents(documents)
							if result and not result.get("error"):
								extraction_results.append(("LangChain", result))
								print(f" LangChain extraction successful: {len(total_text)} characters")
								if progress_callback:
									progress_callback("pass1", "completed", f"LangChain extraction successful: {len(total_text)} characters")
							else:
								print(" LangChain processing failed")
								if progress_callback:
									progress_callback("pass1", "failed", "LangChain processing failed")
						else:
							print(" LangChain extraction yielded minimal text")
							if progress_callback:
								progress_callback("pass1", "failed", "LangChain extraction yielded minimal text")
					except Exception as e:
						print(f" LangChain extraction failed: {e}")
						if progress_callback:
							progress_callback("pass1", "failed", f"LangChain extraction failed: {e}")
				
				# Pass 2: pdfplumber (enhanced text + tables)
				if ENHANCED_PDF_AVAILABLE and pdfplumber:
					print(" Pass 2: pdfplumber extraction...")
					if progress_callback:
						progress_callback("pass2", "running", "pdfplumber extraction in progress...")
					try:
						result = self._extract_with_pdfplumber(temp_file_path)
						if result and not result.get("error"):
							extraction_results.append(("pdfplumber", result))
							print(" pdfplumber extraction successful")
							if progress_callback:
								progress_callback("pass2", "completed", "pdfplumber extraction successful")
						else:
							print(" pdfplumber extraction failed")
							if progress_callback:
								progress_callback("pass2", "failed", "pdfplumber extraction failed")
					except Exception as e:
						print(f" pdfplumber extraction failed: {e}")
						if progress_callback:
							progress_callback("pass2", "failed", f"pdfplumber extraction failed: {e}")
				
				# Pass 3: Basic PyPDF (fallback)
				print(" Pass 3: Basic PyPDF extraction...")
				if progress_callback:
					progress_callback("pass3", "running", "PyPDF extraction in progress...")
				try:
					with open(temp_file_path, 'rb') as f:
						reader = PdfReader(f)
						text_parts = []
						for page in reader.pages:
							try:
								text_parts.append(page.extract_text() or "")
							except Exception:
								continue
						full_text = "\n".join(text_parts)
						
						if len(full_text.strip()) > 50:
							result = self._structure_text(full_text)
							if result and not result.get("error"):
								extraction_results.append(("PyPDF", result))
								print("✅ PyPDF extraction successful")
								if progress_callback:
									progress_callback("pass3", "completed", "PyPDF extraction successful")
							else:
								print("⚠️ PyPDF processing failed")
								if progress_callback:
									progress_callback("pass3", "failed", "PyPDF processing failed")
						else:
							print("⚠️ PyPDF extraction yielded minimal text")
							if progress_callback:
								progress_callback("pass3", "failed", "PyPDF extraction yielded minimal text")
				except Exception as e:
					print(f"❌ PyPDF extraction failed: {e}")
					if progress_callback:
						progress_callback("pass3", "failed", f"PyPDF extraction failed: {e}")
				
				# Merge all successful extractions
				if extraction_results:
					print(f"🔄 Merging {len(extraction_results)} successful extractions...")
					merged_result = self._merge_multiple_extractions(extraction_results)
					print("✅ Multi-pass extraction completed successfully")
					return merged_result
				else:
					print("❌ All extraction methods failed")
					return self._create_basic_structure("All PDF extraction methods failed")
				
			finally:
				# Clean up temporary file
				if temp_file_path and os.path.exists(temp_file_path):
					try:
						os.unlink(temp_file_path)
						print(f"🗑️ Cleaned up temporary file: {temp_file_path}")
					except Exception as cleanup_error:
						print(f"⚠️ Could not clean up temporary file: {cleanup_error}")
		
		except Exception as e:
			print(f"❌ Multi-pass PDF extraction failed: {e}")
			# Ultimate fallback: Basic PyPDF extraction from bytes
			try:
				reader = PdfReader(io.BytesIO(pdf_bytes))
				text_parts = []
				for page in reader.pages:
					try:
						text_parts.append(page.extract_text() or "")
					except Exception:
						continue
				full_text = "\n".join(text_parts)
				result = self._structure_text(full_text)
				# Add warning to result
				if isinstance(result, dict):
					result["extraction_warning"] = f"Emergency fallback extraction used due to error: {str(e)}"
				return result
			except Exception as fallback_error:
				return {"error": f"PDF extraction failed completely: {str(fallback_error)}"}

	def _process_langchain_documents(self, documents: list) -> Dict[str, Any]:
		"""
		Process LangChain documents with intelligent chunking for large documents.
		"""
		if not documents:
			return {"error": "No content could be extracted from the PDF."}

		# Combine the page content from all documents into a single string.
		full_text = "\n\n".join([doc.page_content for doc in documents if doc.page_content])

		if not full_text.strip():
			return {"error": "Extracted text from the PDF is empty."}

		# For very large documents, use intelligent chunking
		if len(full_text) > 20000:
			return self._process_document_with_chunking(full_text)
		else:
			# For smaller documents, process directly
			return self._structure_text(full_text)


	def extract_from_url(self, url: str) -> Dict[str, Any]:
		resp = httpx.get(url, timeout=30)
		resp.raise_for_status()
		soup = BeautifulSoup(resp.text, "html.parser")
		text = soup.get_text(" ")
		return self._structure_text(text)

	def _process_document_with_chunking(self, text: str) -> Dict[str, Any]:
		"""Process document with simple chunking for large documents."""
		if len(text) <= 20000:
			# For smaller documents, use direct processing
			return self._structure_text(text)
		
		# Simple chunking for large documents
		chunk_size = 15000
		chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
		
		# Process each chunk and combine results
		combined_data = {
			"demographics": {"age": None, "sex": None},
			"free_text": "",
			"vitals": "",
			"labs": "",
			"medications": "",
			"imaging": ""
		}
		
		for chunk in chunks:
			chunk_data = self._structure_text(chunk)
			
			# Merge demographics (take first non-null values)
			if chunk_data.get("demographics"):
				if not combined_data["demographics"]["age"] and chunk_data["demographics"].get("age"):
					combined_data["demographics"]["age"] = chunk_data["demographics"]["age"]
				if not combined_data["demographics"]["sex"] and chunk_data["demographics"].get("sex"):
					combined_data["demographics"]["sex"] = chunk_data["demographics"]["sex"]
			
			# Concatenate text fields
			for field in ["free_text", "vitals", "labs", "medications", "imaging"]:
				field_value = chunk_data.get(field)
				if field_value and isinstance(field_value, str) and field_value.strip():
					if combined_data[field]:
						combined_data[field] += f"\n{field_value}"
					else:
						combined_data[field] = field_value
		
		return combined_data


	def _structure_text(self, text: str) -> Dict[str, Any]:
		"""Enhanced text structuring with forgiving prompts and robust JSON extraction."""
		
		prompt = (
			"You are an expert medical data extraction specialist. Analyze this medical report/document and extract ALL clinical information.\n\n"
			"IMPORTANT: Put your JSON response inside ```json code blocks like this:\n"
			"```json\n"
			"{\n"
			"  \"demographics\": {\"age\": 45, \"sex\": \"Male\"},\n"
			"  \"free_text\": \"Patient presents with chest pain...\",\n"
			"  \"vitals\": \"BP: 120/80 mmHg, HR: 72 bpm, Temp: 98.6°F\",\n"
			"  \"labs\": \"Glucose: 95 mg/dL (70-100), Creatinine: 1.0 mg/dL (0.7-1.3)\",\n"
			"  \"medications\": \"Lisinopril 10mg daily, Metformin 500mg twice daily\",\n"
			"  \"imaging\": \"Chest X-ray: Normal heart size, clear lungs\",\n"
			"  \"diagnoses\": [\"Hypertension\", \"Type 2 Diabetes\"]\n"
			"}\n"
			"```\n\n"
			"You may add explanatory text outside the code blocks if needed, but the JSON must be inside ```json blocks.\n\n"
			"EXTRACTION REQUIREMENTS:\n"
			"- **DEMOGRAPHICS (CRITICAL)**: Look carefully for patient age and sex/gender. Search for:\n"
			"  * Age: numbers followed by 'years old', 'y/o', 'yr old', 'age', or similar\n"
			"  * Sex: 'Male', 'Female', 'M', 'F', 'man', 'woman', 'gender', or similar\n"
			"  * Common locations: patient header, demographics section, admission notes\n"
			"- Extract ALL laboratory values with exact numbers, units, and reference ranges\n"
			"- Include ALL vital signs (BP, HR, temp, respiratory rate, SpO2)\n"
			"- List ALL medications with dosages and frequencies\n"
			"- Include ALL imaging results and interpretations\n"
			"- List ALL diagnoses and medical conditions mentioned\n\n"
			"If you cannot find data for a field, use:\n"
			"- Empty string \"\" for text fields\n"
			"- Empty array [] for diagnoses\n"
			"- null for missing age/sex\n\n"
			"DEMOGRAPHICS EXAMPLES TO LOOK FOR:\n"
			"- '45-year-old male' → age: 45, sex: 'Male'\n"
			"- 'Age: 67, Gender: Female' → age: 67, sex: 'Female'\n"
			"- 'Patient is a 32 y/o woman' → age: 32, sex: 'Female'\n"
			"- 'DOB: 1980 (43 years old), Sex: M' → age: 43, sex: 'Male'\n\n"
			f"MEDICAL DOCUMENT CONTENT:\n{text}"
		)
		
		# Get structured response with markdown parsing
		try:
			# Use regular text generation instead of JSON-specific method
			raw_response = self.reasoner.generate_text(prompt)
			
			# Parse JSON from markdown code blocks
			parsed_response = self._parse_json_from_markdown(raw_response)
			
			if parsed_response:
				return parsed_response
			else:
				print("⚠️ Could not parse JSON from response, returning basic structure")
				return self._create_basic_structure(text)
				
		except Exception as e:
			print(f"❌ Markdown JSON extraction failed: {e}")
			return self._create_basic_structure(text)
	
	
	def _create_basic_structure(self, text: str) -> Dict[str, Any]:
		"""Create a basic structure when all extraction methods fail."""
		return {
			"demographics": {"age": None, "sex": None},
			"free_text": text[:2000] + "..." if len(text) > 2000 else text,  # Truncate very long text
			"vitals": "",
			"labs": "",
			"medications": "",
			"imaging": "",
			"diagnoses": [],
			"extraction_warning": "Automated extraction failed. Raw text provided in free_text field."
		}
	
	
	def _parse_json_from_markdown(self, text: str) -> Dict[str, Any]:
		"""Parse JSON from markdown code blocks in LLM response."""
		import json
		import re
		
		# Look for JSON code blocks
		json_pattern = r'```json\s*(.*?)\s*```'
		matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
		
		if not matches:
			# Try without language specifier
			json_pattern = r'```\s*(.*?)\s*```'
			matches = re.findall(json_pattern, text, re.DOTALL)
		
		if not matches:
			# Try to find JSON-like structure without code blocks
			json_pattern = r'\{.*\}'
			matches = re.findall(json_pattern, text, re.DOTALL)
		
		for match in matches:
			try:
				# Clean up the JSON string
				json_str = match.strip()
				
				# Try to parse as JSON
				parsed = json.loads(json_str)
				
				# Validate it has the expected structure
				if isinstance(parsed, dict) and any(key in parsed for key in ['demographics', 'free_text', 'vitals', 'labs']):
					return parsed
					
			except json.JSONDecodeError:
				continue
		
		# If no valid JSON found, return None
		return None
	
	def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
		"""Extract text and tables using pdfplumber."""
		if not ENHANCED_PDF_AVAILABLE or not pdfplumber:
			return {"error": "pdfplumber not available"}
		
		try:
			full_text = ""
			tables_text = ""
			
			with pdfplumber.open(pdf_path) as pdf:
				for page in pdf.pages:
					page_text = page.extract_text()
					if page_text:
						full_text += page_text + "\n"
					
					# Extract tables for lab results
					tables = page.extract_tables()
					for table in tables:
						if table:
							table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
							tables_text += f"\nTable:\n{table_text}\n"
			
			combined_text = full_text + tables_text
			
			if len(combined_text.strip()) > 50:
				return self._structure_text(combined_text)
			else:
				return {"error": "pdfplumber extraction yielded minimal text"}
				
		except Exception as e:
			return {"error": f"pdfplumber extraction failed: {str(e)}"}
	
	def _merge_multiple_extractions(self, extraction_results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
		"""Merge results from multiple extraction methods, prioritizing quality and completeness."""
		if not extraction_results:
			return self._create_basic_structure("No extraction results to merge")
		
		# Start with empty base structure
		merged_result = {
			"demographics": {"age": None, "sex": None},
			"free_text": "",
			"vitals": "",
			"labs": "",
			"medications": "",
			"imaging": "",
			"diagnoses": [],
			"extraction_methods": []
		}
		
		# Track which methods contributed
		methods_used = []
		
		for method_name, result in extraction_results:
			if not result or result.get("error"):
				continue
			
			methods_used.append(method_name)
			
			# Merge demographics (take first non-null values)
			if result.get("demographics"):
				if not merged_result["demographics"]["age"] and result["demographics"].get("age"):
					merged_result["demographics"]["age"] = result["demographics"]["age"]
				if not merged_result["demographics"]["sex"] and result["demographics"].get("sex"):
					merged_result["demographics"]["sex"] = result["demographics"]["sex"]
			
			# Merge text fields (concatenate with source labels)
			text_fields = ["free_text", "vitals", "labs", "medications", "imaging"]
			for field in text_fields:
				field_value = result.get(field)
				if field_value and isinstance(field_value, str) and field_value.strip():
					if merged_result[field]:
						merged_result[field] += f"\n\n--- FROM {method_name.upper()} ---\n{field_value}"
					else:
						merged_result[field] = f"--- FROM {method_name.upper()} ---\n{field_value}"
			
			# Merge diagnoses (combine unique entries)
			if result.get("diagnoses") and isinstance(result["diagnoses"], list):
				for diagnosis in result["diagnoses"]:
					if diagnosis and diagnosis not in merged_result["diagnoses"]:
						merged_result["diagnoses"].append(diagnosis)
		
		# Add extraction metadata
		merged_result["extraction_methods"] = methods_used
		merged_result["extraction_note"] = f"Data merged from {len(methods_used)} extraction methods: {', '.join(methods_used)}"
		
		print(f"✅ Successfully merged data from: {', '.join(methods_used)}")
		return merged_result
	
