from typing import Dict, Any, List
import time

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from services.ollama import OllamaReasoner
# Whisper service import with error handling
try:
    from services.whisper_service import WhisperService
    WHISPER_SERVICE_AVAILABLE = True
except Exception as e:
    WhisperService = None
    WHISPER_SERVICE_AVAILABLE = False
    print(f"⚠️ Whisper service not available: {e}")
    print("💡 Voice note functionality will be disabled. To fix: reinstall PyTorch with CPU support")
from agents.ingestion import IngestionAgent
from agents.analysis import AnalysisAgent
from agents.clinical_rules import ClinicalRuleEngineAgent
from agents.risk import RiskAgent
from agents.advisory import AdvisoryAgent
from agents.report_extract import ReportExtractAgent

st.set_page_config(
    page_title="AI Diagnostic & Triage Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏥"
)

# Custom CSS for medical theme with doctor background
st.markdown("""
<style>
    /* Main app background with medical theme */
    .stApp {
        background: linear-gradient(135deg, rgba(144, 238, 144, 0.3) 0%, rgba(152, 251, 152, 0.2) 25%, rgba(173, 255, 173, 0.25) 50%, rgba(144, 238, 144, 0.2) 75%, rgba(152, 251, 152, 0.3) 100%), 
                   url('./medical_bg.png');
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    /* Medical pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(0, 123, 255, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(40, 167, 69, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 60%, rgba(220, 53, 69, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Header with enhanced medical styling */
    .main-header {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 25%, #2196f3 50%, #03a9f4 75%, #00bcd4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 12px 24px rgba(21, 101, 192, 0.3), 0 0 40px rgba(33, 150, 243, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    /* Medical cross pattern in header */
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="85" cy="15" r="8" fill="white" opacity="0.1"/><path d="M80 15 Q85 10 90 15 Q85 20 80 15" stroke="white" stroke-width="1" fill="none" opacity="0.15"/><circle cx="15" cy="85" r="6" fill="white" opacity="0.08"/><path d="M10 85 Q15 80 20 85" stroke="white" stroke-width="1" fill="none" opacity="0.1"/></svg>');
        background-repeat: repeat;
        background-size: 100px 100px;
        pointer-events: none;
        z-index: 0;
    }
    
    .main-header::after {
        content: '🩺';
        position: absolute;
        top: 15px;
        right: 25px;
        font-size: 2.5rem;
        opacity: 0.4;
        z-index: 1;
    }
    
    /* Sidebar with medical theme */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 50%, #f0f8ff 100%);
        border-right: 3px solid #007bff;
    }
    
    /* Enhanced agent cards with medical styling */
    .agent-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 0.8rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    /* Medical icons for different risk levels */
    .risk-high {
        border-left-color: #dc3545;
        background: rgba(255, 245, 245, 0.95);
    }
    
    .risk-high::before {
        content: '🚨';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.2rem;
    }
    
    .risk-medium {
        border-left-color: #ffc107;
        background: rgba(255, 251, 240, 0.95);
    }
    
    .risk-medium::before {
        content: '⚠️';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.2rem;
    }
    
    .risk-low {
        border-left-color: #28a745;
        background: rgba(240, 255, 244, 0.95);
    }
    
    .risk-low::before {
        content: '✅';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.2rem;
    }
    
    /* Enhanced diagnosis cards */
    .diagnosis-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.2rem;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        margin: 0.8rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(5px);
        position: relative;
    }
    
    .diagnosis-card::before {
        content: '🔬';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.2rem;
        opacity: 0.6;
    }
    
    /* Enhanced confidence bar */
    .confidence-bar {
        height: 10px;
        background: #e9ecef;
        border-radius: 5px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997, #17a2b8);
        transition: width 0.5s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        backdrop-filter: blur(5px);
        position: relative;
    }
    
    .metric-card::before {
        content: '📊';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 1.1rem;
        opacity: 0.5;
    }
    
    /* Medical-themed buttons */
    .stButton > button {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 123, 255, 0.4);
    }
    
    /* Medical-themed progress bars */
    .stProgress > div > div > div {
        background-color: white;
    }
    
    /* Enhanced text areas and inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e9ecef;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
    }
    
    /* White spinner text */
    .stSpinner > div > div {
        font-weight: bold;
    }
    
    /* Alternative spinner text selectors */
    div[data-testid="stSpinner"] > div {
        font-weight: bold;
    }
    
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_reasoner() -> OllamaReasoner:
	return OllamaReasoner()


@st.cache_resource(show_spinner=False)
def get_whisper_service():
	"""Get Whisper service with error handling for PyTorch issues."""
	if not WHISPER_SERVICE_AVAILABLE or WhisperService is None:
		return None
	try:
		return WhisperService(model_size="base")
	except Exception as e:
		print(f"❌ Failed to initialize Whisper service: {e}")
		return None


@st.cache_resource(show_spinner=False)
def get_agents(_reasoner: OllamaReasoner):
	ingest = IngestionAgent()
	analysis = AnalysisAgent(_reasoner)
	clinical_rules = ClinicalRuleEngineAgent()
	risk = RiskAgent(_reasoner)
	advisory = AdvisoryAgent(_reasoner)
	extractor = ReportExtractAgent(_reasoner)
	return ingest, analysis, clinical_rules, risk, advisory, extractor


def render_header():
	st.markdown("""
	<div class="main-header">
		<h1 style="position: relative; z-index: 2; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🏥 AI-Powered Diagnostic & Triage Support System</h1>
		<p style="margin: 0; font-size: 1.2em; opacity: 0.95; position: relative; z-index: 2; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">
			Advanced Multi-Agent Clinical Reasoning with Explainable AI
		</p>
	</div>
	""", unsafe_allow_html=True)


def sidebar_inputs() -> Dict[str, Any]:
	with st.sidebar:
		st.markdown("### 📋 Patient Data Input")
		
		# File upload section
		st.markdown("#### 📄 Document Upload")
	
		# Dropdown for upload method selection
		upload_method = st.selectbox(
			"Select input method:",
			options=["PDF Upload", "URL Link"],
			help="Choose how you want to provide the medical report"
		)
	
		pdf_file = None
		url = None
		voice_transcription = None
	
		if upload_method == "PDF Upload":
			pdf_file = st.file_uploader(
				"Upload Medical Report PDF", 
				type=["pdf"], 
				key="pdf",
				help="Upload a medical report in PDF format for automatic extraction"
			)
		else:  # URL Link
			url = st.text_input(
				"Paste report URL", 
				placeholder="https://example.com/medical-report",
				help="Paste a URL to a medical report webpage"
			)
		
		st.markdown("---")
		
		# Voice Note section (separate and optional)
		st.markdown("#### 🎤 Voice Note (Optional)")
		st.info("📝 Hold the microphone button to record your voice describing patient symptoms, history, and clinical observations.")
		
		# WhatsApp-style voice recorder
		audio_bytes = mic_recorder(
			start_prompt="🎤 Hold to Record",
			stop_prompt="⏹️ Recording...",
			just_once=False,
			use_container_width=True,
			callback=None,
			args=(),
			kwargs={},
			key="voice_recorder"
		)
		
		if audio_bytes:
			st.audio(audio_bytes['bytes'], format="audio/wav")
			
			# Process button
			if st.button("📤 Process Voice Note", key="process_voice", help="Process and integrate voice note into clinical analysis"):
				# Transcribe audio using Whisper
				whisper_service = get_whisper_service()
				
				if whisper_service is None:
					st.error("❌ Voice processing unavailable: Whisper service could not be initialized due to PyTorch issues")
					st.info("💡 To fix: Try reinstalling PyTorch with CPU support: `pip uninstall torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`")
				else:
					# Debug information
					st.info(f"🔍 Audio data size: {len(audio_bytes['bytes'])} bytes")
					
					with st.spinner("🎯 Processing voice note through Whisper → LLaMA → LangChain..."):
						try:
							transcription_result = whisper_service.transcribe_with_medical_context(audio_bytes['bytes'])
						except Exception as e:
							st.error(f"❌ Voice processing failed: {str(e)}")
							transcription_result = {"success": False, "error": str(e)}
				
				if transcription_result["success"]:
					voice_transcription = transcription_result["text"]
					st.success("✅ Voice note transcribed successfully! You can now fill in additional manual entries below.")
					
					# Store in session state for integration with main analysis
					if 'voice_data' not in st.session_state:
						st.session_state.voice_data = {}
					
					st.session_state.voice_data = {
						'transcription': voice_transcription,
						'original_audio': audio_bytes['bytes'],
						'language': transcription_result.get("language", "unknown"),
						'confidence': transcription_result.get("confidence", 0),
						'processed': True
					}
					
					# Show transcription with edit capability
					st.markdown("**📝 Transcribed Voice Note:**")
					voice_transcription = st.text_area(
						"Edit transcription if needed:",
						value=voice_transcription,
						height=100,
						key="voice_transcription_edit",
						help="Review and edit the transcribed text before analysis"
					)
					
					# Update session state with edited transcription
					st.session_state.voice_data['transcription'] = voice_transcription
					
					# Show language info (confidence not available from Whisper)
					col1, col2 = st.columns(2)
					with col1:
						st.metric("Language", transcription_result.get("language", "unknown").upper())
					with col2:
						st.metric("Status", "Transcribed")
					
					# Inform user about next steps
					st.info("👇 **Next Step:** Fill in the manual entries below, then click 'Run AI Analysis' to start the diagnostic process.")
						
				else:
					st.error(f"❌ Voice processing failed: {transcription_result.get('error', 'Unknown error')}")
		
		# Show voice data status if available
		if 'voice_data' in st.session_state and st.session_state.voice_data.get('processed'):
			st.success("🎤 Voice note ready! Transcription will be included in analysis.")
			voice_transcription = st.session_state.voice_data['transcription']
		else:
			voice_transcription = ""
		
		# Optional file upload as alternative
		st.markdown("**Or upload an audio file:**")
		audio_file = st.file_uploader(
			"Upload Audio File", 
			type=["wav", "mp3", "m4a", "flac", "ogg"], 
			key="audio_file_upload",
			help="Upload an audio recording in WAV, MP3, M4A, FLAC, or OGG format"
		)
		
		if audio_file is not None:
			st.audio(audio_file, format="audio/wav")
			
			# Transcribe uploaded audio using Whisper
			whisper_service = get_whisper_service()
			
			if whisper_service is None:
				st.error("❌ Audio transcription unavailable: Whisper service could not be initialized due to PyTorch issues")
				st.info("💡 To fix: Try reinstalling PyTorch with CPU support: `pip uninstall torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`")
				transcription_result = {"success": False, "error": "Whisper service unavailable"}
			else:
				with st.spinner("🎯 Transcribing uploaded audio with Whisper..."):
					audio_bytes_uploaded = audio_file.getvalue()
					transcription_result = whisper_service.transcribe_with_medical_context(audio_bytes_uploaded)
			
			if transcription_result["success"]:
				voice_transcription = transcription_result["text"]
				st.success("✅ Audio transcribed successfully!")
				
				# Show transcription with edit capability
				st.markdown("**Transcribed Text:**")
				voice_transcription = st.text_area(
					"Edit transcription if needed:",
					value=voice_transcription,
					height=100,
					key="voice_transcription_edit_upload",
					help="Review and edit the transcribed text before analysis"
				)
				
				# Show language info (confidence not available from Whisper)
				col1, col2 = st.columns(2)
				with col1:
					st.metric("Language", transcription_result.get("language", "unknown").upper())
				with col2:
					st.metric("Status", "Transcribed")
					
		st.markdown("---")
		
		# Manual input section
		st.markdown("#### ✍️ Manual Entry")
		text = st.text_area(
			"Clinical Notes / HPI / Summary *", 
			height=150,
			placeholder="Enter patient history, symptoms, and clinical observations...",
			help="Primary complaint, history of present illness, and clinical notes (Required)"
		)
		
		col1, col2 = st.columns(2)
		with col1:
			age = st.number_input("Age *", min_value=1, max_value=120, step=1, help="Patient age in years (Required)")
		with col2:
			sex = st.selectbox("Sex *", ["Select...", "Female", "Male", "Other"], index=0, help="Patient sex (Required)")
		
		# Vital Signs section with individual fields
		st.markdown("*Vital Signs *")
		
		# Use 2-column layout instead of 3 to give more width for +/- buttons
		vital_col1, vital_col2 = st.columns(2)
		with vital_col1:
			systolic_bp = st.number_input("Systolic BP *", min_value=50, max_value=300, step=1, help="Systolic blood pressure (mmHg)")
			resp_rate = st.number_input("Respiratory Rate *", min_value=5, max_value=60, step=1, help="Respiratory rate (breaths/min)")
			heart_rate = st.number_input("Heart Rate *", min_value=30, max_value=250,  step=1, help="Heart rate (bpm)")
		
		with vital_col2:
			diastolic_bp = st.number_input("Diastolic BP *", min_value=30, max_value=200, step=1, help="Diastolic blood pressure (mmHg)")
			temperature = st.number_input("Temp *", min_value=30.0, max_value=45.0, step=0.1, help="Body temperature (°C)")
			spo2 = st.number_input("SpO2 *", min_value=50, max_value=100, step=1, help="Oxygen saturation (%)")
		
		# Combine vitals into a formatted string
		vitals = ""
		if all(v is not None for v in [systolic_bp, diastolic_bp, heart_rate, resp_rate, temperature, spo2]):
			vitals = f"BP {systolic_bp}/{diastolic_bp}, HR {heart_rate}, Temp {temperature}°C, RR {resp_rate}, SpO2 {spo2}%"
		
		meds = st.text_area(
			"Current Medications *", 
			placeholder="metformin, lisinopril, aspirin (or 'None' if no medications)",
			help="List of current medications and dosages (Required - enter 'None' if no medications)"
		)
		
		imaging = st.text_area(
			"Imaging Studies *", 
			placeholder="CXR: bilateral infiltrates, CT: no acute findings (or 'None' if no imaging)",
			help="Radiology reports and imaging findings (Required - enter 'None' if no imaging studies)"
		)
		
		return {
			"pdf_file": pdf_file,
			"url": url.strip() if url else "",
			"voice_transcription": voice_transcription.strip() if voice_transcription else "",
			"text": text.strip() if text else "",
			"age": age,
			"sex": sex,
			"vitals": vitals.strip() if vitals else "",
			"systolic_bp": systolic_bp,
			"diastolic_bp": diastolic_bp,
			"heart_rate": heart_rate,
			"resp_rate": resp_rate,
			"temperature": temperature,
			"spo2": spo2,
			"meds": meds.strip() if meds else "",
			"imaging": imaging.strip() if imaging else "",
		}


def validate_mandatory_fields(inputs: Dict[str, Any]) -> tuple[bool, list[str]]:
	"""Validate all mandatory patient data fields."""
	missing_fields = []
	
	# Check document input (PDF or URL) - Voice Note is now optional
	if not ((inputs["pdf_file"] is not None) or bool(inputs["url"])):
		missing_fields.append("Document (PDF or URL)")
	
	# Check clinical notes (manual entry is required, voice note is supplementary)
	if not inputs["text"]:
		missing_fields.append("Clinical Notes / HPI / Summary")
	
	# Check age
	if inputs["age"] is None or inputs["age"] <= 0:
		missing_fields.append("Age")
	
	# Check sex
	if inputs["sex"] == "Select..." or not inputs["sex"]:
		missing_fields.append("Sex")
	
	# Check individual vital signs
	vital_fields = ["systolic_bp", "diastolic_bp", "heart_rate", "resp_rate", "temperature", "spo2"]
	vital_names = ["Systolic BP", "Diastolic BP", "Heart Rate", "Respiratory Rate", "Temperature", "SpO2"]
	
	for field, name in zip(vital_fields, vital_names):
		if inputs.get(field) is None:
			missing_fields.append(name)
	
	# Check medications
	if not inputs["meds"]:
		missing_fields.append("Current Medications")
	
	# Check imaging
	if not inputs["imaging"]:
		missing_fields.append("Imaging Studies")
	
	return len(missing_fields) == 0, missing_fields




def merge_data_sources(manual_data: Dict[str, Any], extracted_data: Dict[str, Any], source_label: str) -> Dict[str, Any]:
	"""Helper function to merge manual data with extracted data from various sources."""
	if not extracted_data:
		return manual_data
	
	# Merge demographics (extracted data takes priority if available)
	if extracted_data.get("demographics"):
		if extracted_data["demographics"].get("age"):
			manual_data["demographics"]["age"] = extracted_data["demographics"]["age"]
		if extracted_data["demographics"].get("sex"):
			manual_data["demographics"]["sex"] = extracted_data["demographics"]["sex"]
	
	# Merge text fields with source labels
	text_fields = ["free_text", "vitals", "medications", "imaging"]
	for field in text_fields:
		if extracted_data.get(field) and extracted_data[field].strip():
			if manual_data.get(field) and manual_data[field].strip():
				manual_data[field] = f"{manual_data[field]}\n\n--- FROM {source_label} ---\n{extracted_data[field]}"
			else:
				manual_data[field] = f"--- FROM {source_label} ---\n{extracted_data[field]}"
	
	# Handle labs field specially (always append)
	if extracted_data.get("labs") and extracted_data["labs"].strip():
		if manual_data.get("labs") and manual_data["labs"].strip():
			manual_data["labs"] = f"{manual_data['labs']}\n\n--- FROM {source_label} ---\n{extracted_data['labs']}"
		else:
			manual_data["labs"] = f"--- FROM {source_label} ---\n{extracted_data['labs']}"
	
	return manual_data


def aggregate_inputs(inputs: Dict[str, Any], extractor: ReportExtractAgent, session: Dict[str, Any]) -> Dict[str, Any]:
	# Start with manual input data
	data: Dict[str, Any] = {
		"demographics": {"age": inputs["age"], "sex": inputs["sex"]},
		"free_text": inputs["text"],
		"vitals": inputs["vitals"],
		"medications": inputs["meds"],
		"imaging": inputs["imaging"],
		"labs": ""  # Initialize labs field
	}
	
	# Process voice transcription
	if inputs["voice_transcription"]:
		voice_data = {"free_text": inputs["voice_transcription"]}
		data = merge_data_sources(data, voice_data, "VOICE TRANSCRIPTION")
		session.setdefault("events", []).append({
			"agent": "🎤 Voice Processing", 
			"action": "Voice Transcription Integration", 
			"details": "Voice transcription integrated into clinical data"
		})
	
	# Process PDF with detailed pass tracking
	if inputs["pdf_file"] is not None:
		# Create a progress container for Step 1 passes
		step1_progress_container = st.container()
		with step1_progress_container:
			st.markdown("#### 📄 Step 1 - PDF Extraction Progress")
			pass_cols = st.columns(3)
			
			# Initialize pass status displays
			with pass_cols[0]:
				pass1_status = st.empty()
				pass1_status.markdown("⏳ **Pass 1:** LangChain PyPDFLoader")
			with pass_cols[1]:
				pass2_status = st.empty()
				pass2_status.markdown("⏳ **Pass 2:** pdfplumber Tables")
			with pass_cols[2]:
				pass3_status = st.empty()
				pass3_status.markdown("⏳ **Pass 3:** PyPDF Fallback")
		
		# Define progress callback function
		def update_pass_status(pass_id, status, message):
			status_icons = {
				"running": "🚀",
				"completed": "✅", 
				"failed": "❌"
			}
			icon = status_icons.get(status, "⏳")
			
			if pass_id == "pass1":
				pass1_status.markdown(f"{icon} **Pass 1:** LangChain PyPDFLoader - {status.title()}")
			elif pass_id == "pass2":
				pass2_status.markdown(f"{icon} **Pass 2:** pdfplumber Tables - {status.title()}")
			elif pass_id == "pass3":
				pass3_status.markdown(f"{icon} **Pass 3:** PyPDF Fallback - {status.title()}")
		
		try:
			pdf_bytes = inputs["pdf_file"].getvalue()
			
			# Call extract method with progress callback
			extracted = extractor.extract_from_pdf(pdf_bytes, progress_callback=update_pass_status)
			
			data = merge_data_sources(data, extracted, "PDF")
			
			session.setdefault("events", []).append({
				"agent": "📄 Multi-Pass Extract", 
				"action": "PDF Processing (3 Passes)", 
				"details": "PDF processed through LangChain → pdfplumber → PyPDF pipeline"
			})
			st.success("✅ PDF processed through 3-pass extraction pipeline!")
			
		except Exception as e:
			# Update all passes to show failure if extraction completely fails
			pass1_status.markdown("❌ **Pass 1:** LangChain PyPDFLoader - Failed")
			pass2_status.markdown("❌ **Pass 2:** pdfplumber Tables - Failed") 
			pass3_status.markdown("❌ **Pass 3:** PyPDF Fallback - Failed")
			st.error(f"❌ PDF processing failed: {e}")
	
	# Process URL
	if inputs["url"]:
		with st.spinner("🌐 Processing URL..."):
			try:
				extracted = extractor.extract_from_url(inputs["url"])
				data = merge_data_sources(data, extracted, "URL")
				
				session.setdefault("events", []).append({
					"agent": "🌐 URL Extract", 
					"action": "URL Processing", 
					"details": "URL content processed and data extracted"
				})
				st.success("✅ URL processed successfully!")
				
			except Exception as e:
				st.error(f"❌ URL processing failed: {e}")
	
	return data




def render_diagnoses(diagnoses: List[Dict[str, Any]]):
	if not diagnoses:
		return
	
	st.markdown("### 🎯 Diagnostic Analysis")
	
	for i, dx in enumerate(diagnoses, 1):
		confidence = dx.get("confidence", 0)
		confidence_pct = int(confidence * 100)
		
		# Get validation status and notes
		validation_status = dx.get("validation_status", "UNKNOWN")
		rule_engine_notes = dx.get("rule_engine_notes", "")
		
		# Set styling based on validation status
		if validation_status == "VALIDATED":
			card_class = "diagnosis-card"
			status_icon = "✅"
			status_color = "#28a745"
		elif validation_status == "FLAGGED":
			card_class = "diagnosis-card risk-medium"
			status_icon = "⚠️"
			status_color = "#ffc107"
		else:
			card_class = "diagnosis-card"
			status_icon = "❓"
			status_color = "#6c757d"
		
		st.markdown(f"""
		<div class="{card_class}">
			<h4>#{i} {dx.get('name', 'Unknown Diagnosis')} {status_icon}</h4>
			<div class="confidence-bar">
				<div class="confidence-fill" style="width: {confidence_pct}%"></div>
			</div>
			<p><strong>Confidence:</strong> {confidence_pct}%</p>
			<p><strong>Evidence:</strong> {dx.get('evidence', 'No evidence provided')}</p>
			<p style="color: {status_color}; font-weight: bold;"><strong>Clinical Validation:</strong> {validation_status}</p>
		</div>
		""", unsafe_allow_html=True)


def render_risk_flags(flags: List[Dict[str, Any]]):
	if not flags:
		return
	
	st.markdown("### 🚨 Risk Assessment")
	
	for flag in flags:
		urgency = flag.get("urgency", "low").lower()
		risk_class = f"risk-{urgency}"
		
		st.markdown(f"""
		<div class="agent-card {risk_class}">
			<h4>⚠️ {flag.get('name', 'Risk Flag')}</h4>
			<p><strong>Urgency:</strong> {flag.get('urgency', 'Unknown').upper()}</p>
			<p><strong>Rationale:</strong> {flag.get('rationale', 'No rationale provided')}</p>
		</div>
		""", unsafe_allow_html=True)


def render_ingestion_data(structured_data: Dict[str, Any]):
	"""Render ingestion data in a structured format."""
	if not structured_data:
		st.warning("No ingestion data available")
		return
	
	# Patient Demographics and Vital Signs
	if structured_data.get("patient") or structured_data.get("vitals"):
		st.markdown("#### 👤 Age, Sex & Vital Signs")
		
		# Demographics row
		if structured_data.get("patient"):
			patient = structured_data["patient"]
			col1, col2 = st.columns(2)
			with col1:
				st.metric("Age", f"{patient.get('age', 'N/A')} years")
			with col2:
				st.metric("Sex", patient.get('sex', 'N/A'))
		
		# Vital Signs
		if structured_data.get("vitals"):
			st.info(f"**Vital Signs:** {structured_data['vitals']}")
		
		st.markdown("---")
	
	# Clinical Notes
	if structured_data.get("symptom_notes"):
		st.markdown("#### 📝 Clinical Notes")
		st.text_area(
			"Clinical Notes",
			structured_data["symptom_notes"],
			height=100,
			disabled=True,
			key="clinical_notes_display",
			label_visibility="collapsed"
		)
		st.markdown("---")
	
	# Laboratory Results
	if structured_data.get("labs"):
		st.markdown("#### 🧪 Laboratory Results")
		st.text_area(
			"Laboratory Results",
			structured_data["labs"],
			height=150,
			disabled=True,
			key="labs_display",
			label_visibility="collapsed"
		)
		st.markdown("---")
	
	# Medications
	if structured_data.get("medications"):
		st.markdown("#### 💊 Medications")
		st.text_area(
			"Medications",
			structured_data["medications"],
			height=100,
			disabled=True,
			key="medications_display",
			label_visibility="collapsed"
		)
		st.markdown("---")
	
	# Imaging
	if structured_data.get("imaging"):
		st.markdown("#### 📷 Imaging Studies")
		st.text_area(
			"Imaging Studies",
			structured_data["imaging"],
			height=100,
			disabled=True,
			key="imaging_display",
			label_visibility="collapsed"
		)


def render_clinical_advisory(advice: Dict[str, Any]):
	"""Render clinical advisory in a structured, readable format."""
	if not advice:
		st.warning("No clinical advisory available")
		return
	
	advisory_data = advice
	
	# Summary Section
	if advisory_data.get("summary"):
		st.markdown("#### 📋 Clinical Summary")
		st.markdown(f"**{advisory_data['summary']}**")
		st.markdown("---")
	
	# Recommendations Section
	if advisory_data.get("recommendations"):
		st.markdown("#### 💡 Clinical Recommendations")
		for i, rec in enumerate(advisory_data["recommendations"], 1):
			st.markdown(f"**{i}.** {rec}")
		st.markdown("---")
	
	# Next Steps Section
	if advisory_data.get("next_steps"):
		st.markdown("#### 🎯 Next Steps")
		for i, step in enumerate(advisory_data["next_steps"], 1):
			st.markdown(f"**{i}.** {step}")
		st.markdown("---")
	
	# Data Traceability Section
	if advisory_data.get("trace", {}).get("data_links"):
		st.markdown("#### 🔗 Data Traceability")
		st.markdown("*Evidence supporting this advisory:*")
		for link in advisory_data["trace"]["data_links"]:
			st.markdown(f"• {link}")


def render_agents_timeline(events: List[Dict[str, Any]], dx_ranked: List[Dict[str, Any]] = None, flags: List[Dict[str, Any]] = None, advice: Dict[str, Any] = None, structured_data: Dict[str, Any] = None):
	if not events:
		return
	
	for i, event in enumerate(events):
		agent_name = event.get('agent', 'Unknown Agent')
		action = event.get('action', 'Unknown')
		details = event.get('details', 'No details')
		
		# Create expandable sections for different agents
		if agent_name == "Ingestion" and structured_data:
			with st.expander(f"📥 {agent_name} - {action}", expanded=False):
				st.markdown(f"**Details:** {details}")
				st.markdown("---")
				render_ingestion_data(structured_data)
		elif agent_name == "Analysis" and dx_ranked:
			with st.expander(f"🧠 {agent_name} - {action}", expanded=False):
				st.markdown(f"**Details:** {details}")
				st.markdown("---")
				render_diagnoses(dx_ranked)
		elif agent_name == "Risk" and flags:
			with st.expander(f"⚠️ {agent_name} - {action}", expanded=False):
				st.markdown(f"**Details:** {details}")
				st.markdown("---")
				render_risk_flags(flags)
		elif agent_name == "Advisory" and advice:
			with st.expander(f"📋 {agent_name} - {action}", expanded=False):
				st.markdown(f"**Details:** {details}")
				st.markdown("---")
				render_clinical_advisory(advice)
		else:
			st.markdown(f"""
			<div class="agent-card">
				<h4>{agent_name}</h4>
				<p><strong>Action:</strong> {action}</p>
				<p><strong>Details:</strong> {details}</p>
			</div>
			""", unsafe_allow_html=True)


def main():
	render_header()
	inputs = sidebar_inputs()
	reasoner = get_reasoner()
	ingest, analysis, clinical_rules, risk, advisory, extractor = get_agents(reasoner)

	# Main content area
	col1, col2 = st.columns([0.6, 0.4])
	
	with col1:
		st.markdown('<h3>🔬 Clinical Analysis Pipeline</h3>', unsafe_allow_html=True)
		
		# Show status of voice note if processed
		if 'voice_data' in st.session_state and st.session_state.voice_data.get('processed'):
			st.info("🎤 Voice note transcribed and ready for analysis")
		
		if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, help="Start comprehensive AI diagnostic analysis using all provided data"):
			# Validate all mandatory fields
			is_valid, missing_fields = validate_mandatory_fields(inputs)
			if not is_valid:
				st.error("❌ Please fill in all required fields before running analysis:")
				for field in missing_fields:
					st.write(f"• {field}")
				st.info("💡 All fields marked with * are mandatory.")
				return
			
			session: Dict[str, Any] = {"events": []}
			
			# Progress tracking
			progress_container = st.container()
			progress_bar = st.progress(0)
			status_text = st.empty()
			
			# Step 1: Data Ingestion
			status_text.text("📥 Step 1/6: Data Ingestion & Extraction")
			patient = aggregate_inputs(inputs, extractor, session)
			progress_bar.progress(20)
			time.sleep(1)  # Allow UI to update
			
			# Step 2: Data Structuring
			status_text.text("🔧 Step 2/6: Data Structuring & Normalization")
			structured = ingest.run(patient, session)
			progress_bar.progress(40)
			time.sleep(1)  # Allow UI to update
			
			# Step 3: Diagnostic Analysis
			status_text.text("🧠 Step 3/6: AI Diagnostic Analysis")
			dx_ranked = analysis.run(structured, session)
			progress_bar.progress(50)
			time.sleep(1)  # Allow UI to update
			
			# Step 4: Clinical Rule Validation
			status_text.text("🩺 Step 4/6: Clinical Rule Engine Validation")
			dx_validated = clinical_rules.validate_diagnoses(structured, dx_ranked)
			progress_bar.progress(65)
			time.sleep(1)  # Allow UI to update
			
			# Step 5: Risk Assessment
			status_text.text("⚠️ Step 5/6: Risk Assessment & Red Flag Detection")
			flags = risk.run(structured, dx_validated, session)
			progress_bar.progress(80)
			time.sleep(1)  # Allow UI to update
			
			# Step 6: Advisory Generation
			status_text.text("📋 Step 6/6: Generating Clinical Advisory")
			advice = advisory.run(structured, dx_validated, flags, session)
			progress_bar.progress(100)
			time.sleep(1)  # Show completion
			
			# Keep progress indicators visible for reference
			status_text.text("✅ Analysis Complete - All 6 steps completed successfully")
			
			# Add a progress summary section
			st.markdown("### 📏 Processing Summary")
			progress_cols = st.columns(6)
			step_names = [
				"📥 Data Ingestion",
				"🔧 Data Structuring", 
				"🧠 AI Diagnosis",
				"🩺 Clinical Validation",
				"⚠️ Risk Assessment",
				"📋 Advisory Generation"
			]
			
			for i, (col, step_name) in enumerate(zip(progress_cols, step_names)):
				with col:
					st.markdown(f"""
					<div style="text-align: center; padding: 10px; background: rgba(40, 167, 69, 0.1); border-radius: 8px; border-left: 3px solid #28a745;">
						<div style="font-size: 1.2em; margin-bottom: 5px;">✅</div>
						<div style="font-size: 0.8em; font-weight: bold;">Step {i+1}</div>
						<div style="font-size: 0.7em;">{step_name.split(' ', 1)[1]}</div>
					</div>
					""", unsafe_allow_html=True)
			
			# Display results
			st.success("🎉 AI Analysis Complete! All 6 processing steps finished successfully. Results are displayed below.")
			
			# === MAIN RESULTS SECTION ===
			st.markdown("---")
			st.markdown("## 📋 **Clinical Results Summary**")
			
			# Display comprehensive results in single timeline view
			st.markdown("### 📊 Multi-Agent Processing Timeline")
			# Render detailed timeline with all information
			render_agents_timeline(session["events"], dx_validated, flags, advice, structured)
			
	
	with col2:
		st.markdown("### 📈 System Status")
		
		# Metrics
		col_a, col_b = st.columns(2)
		with col_a:
			st.metric("🤖 AI Agents", "6", "Active")
		with col_b:
			st.metric("⚡ Processing", "Ready", "Online")
		
		# System info
		st.markdown("""
		<div class="metric-card">
			<h4>🏥 Clinical AI System</h4>
			<p><strong>Status:</strong> Operational</p>
			<p><strong>Model:</strong> Llama 3.1 8B + Whisper</p>
			<p><strong>Capabilities:</strong></p>
			<ul>
				<li>📄 PDF/URL Extraction</li>
				<li>🎤 Voice Note Transcription</li>
				<li>🧠 Diagnostic Reasoning</li>
				<li>⚠️ Risk Assessment</li>
				<li>📋 Clinical Advisory</li>
			</ul>
		</div>
		""", unsafe_allow_html=True)
		
		# Disclaimer
		st.markdown("---")
		st.warning("""
		⚠️ **Important Disclaimer**
		
		This AI system is for educational and research purposes only. 
		It is not a medical device and should not be used for clinical decision-making.
		
		Always consult qualified healthcare professionals for medical advice.
		""")
		
		# Quick stats
		if 'session' in locals() and session.get("events"):
			st.markdown("### 📊 Session Statistics")
			st.metric("Total Agents", len(set(e.get("agent", "") for e in session["events"])))
			st.metric("Processing Steps", len(session["events"]))


if __name__ == "__main__":
	main()