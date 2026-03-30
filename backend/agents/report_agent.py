import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def run_report_agent(
    patient_name: str,
    scan_type: str,
    findings: list,
    risk_level: str,
    top_finding: str,
    rag_context: str,
    body_location: str = None
) -> str:
    """
    Agent 3 — Report Synthesis Agent.
    Persona: Senior Clinical Diagnostic Agent.
    Reconciles visual CV findings with Clinical Context (RAG) and Patient Data.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    findings_text = "\n".join([
        f"- {f['condition']}: {f['confidence']*100:.1f}% confidence"
        for f in findings
    ])

    scan_labels = {
        "chest_xray": "Chest X-Ray",
        "skin_lesion": "Skin Lesion",
        "retinal": "Retinal Scan"
    }
    scan_label = scan_labels.get(scan_type, scan_type)

    # Build scan-specific instructions
    if scan_type == "chest_xray":
        specific_instructions = """
IMPORTANT FOR CHEST X-RAY INTERPRETATION:
- Infiltration + Consolidation together = strong clinical indicator of pneumonia.
- Synthesize radiological findings into a single clinical picture rather than a list.
"""
    elif scan_type == "skin_lesion" or scan_type == "skin":
        specific_instructions = f"""
IMPORTANT FOR SKIN LESION INTERPRETATION:
- Body location is a CRITICAL clinical anchor.
- BCC and Actinic Keratosis (AK) are statistically dominant on sun-exposed areas (face, neck, scalp).
- If the CV Model suggests a benign 'Nevus' but the location is 'Face' or 'Lower Lip', you MUST cross-reference with RAG context to check if BCC or AK should be mentioned as a clinical suspicion.
- Highlight any contradictions between visual confidence and clinical probability.
{f'- Patient reported lesion on: {body_location}' if body_location else '- No location provided, note this limitation'}
"""
    else:
        specific_instructions = """
IMPORTANT FOR RETINAL SCAN INTERPRETATION:
- Grade diabetic retinopathy clearly (No DR / Mild / Moderate / Severe / Proliferative).
- Ensure referral urgency matches the DR grade.
"""

    # THE CORE REASONING PROMPT
    prompt = f"""You are a Senior Clinical Diagnostic Agent. Your task is to reconcile visual data with clinical context.

### INPUT DATA:
PATIENT: {patient_name}
SCAN TYPE: {scan_label}
BODY LOCATION: {body_location if body_location else 'Not Provided'}
PRIMARY CV FINDING: {top_finding}

### CV MODEL OUTPUTS:
{findings_text}

### CLINICAL KNOWLEDGE BASE (RAG):
{rag_context}

### RECONCILIATION RULES:
1. THE CV MODEL IS A SUGGESTER: It analyzes pixels and can be fooled by visual similarities (e.g., mistaking BCC for a Nevus).
2. RAG & LOCATION ARE FACTS: Use these to validate or challenge the CV model. 
3. MALIGNANCY OVERRIDE: If the CV model predicts a benign condition (Low Risk) but the RAG evidence suggests the location is High Risk for malignancy (e.g. Face/BCC), you MUST address the possibility of the malignant condition in the summary.

{specific_instructions}

Generate a structured clinical report with EXACTLY these sections:

## 📋 SCAN SUMMARY
Summarize the findings. If there is a contradiction between the CV model and the clinical context (e.g. location-based risk), clearly state: "While the visual model suggests [X], the clinical location of [Location] necessitates a differential diagnosis of [Y]."

## 🔍 DETAILED FINDINGS
Explain the visual findings in plain language. Group related findings.

## 📍 LOCATION ANALYSIS
(Include ONLY for skin scans)
Detail how the specific body location affects the risk. Explain why certain conditions (like BCC or AK) are more probable due to this location.

## ⚠️ RISK ASSESSMENT
Risk Level: {risk_level}
Explain the reasoning. If clinical context increases the risk beyond what the CV model calculated, state it clearly.

## 💊 CLINICAL RECOMMENDATIONS
Provide 3-5 actionable steps. Tailor these to the specific findings AND the location.

## 👨‍⚕️ REFERRAL SUGGESTION
Specify the specialist and urgency.

## ℹ️ DISCLAIMER
Standard medical AI disclaimer.

Keep the tone professional, objective, and cautious."""

    print(f"[Report Agent] Reconciling data for {patient_name}...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a clinical diagnostic agent that synthesizes CV findings with medical literature. You prioritize patient safety by highlighting potential malignant risks even if visual models are uncertain."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2, # Lower temperature for higher consistency
        max_tokens=2000
    )

    report = response.choices[0].message.content
    print(f"[Report Agent] Reasoning report generated successfully")
    return report