import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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
    Generates structured clinical report using CV findings,
    RAG context, and body location for differential diagnosis.
    """

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

    # Build location context for prompt
    location_context = ""
    if body_location and scan_type == "skin_lesion":
        location_context = f"\nBODY LOCATION: {body_location}"

    # Build scan-specific instructions
    if scan_type == "chest_xray":
        specific_instructions = """
IMPORTANT FOR CHEST X-RAY INTERPRETATION:
- Multiple radiological findings often describe the same underlying condition
- Infiltration + Consolidation together = strong indicator of pneumonia
- Always connect radiological terms to plain clinical diagnosis
- Example: "Infiltration (67%) and Consolidation (51%) together are consistent with bacterial pneumonia"
- Never just list findings — always synthesize them into a clinical picture
"""
    elif scan_type == "skin_lesion":
        specific_instructions = f"""
IMPORTANT FOR SKIN LESION INTERPRETATION:
- Body location is CRITICAL for differential diagnosis
- BCC occurs almost exclusively on sun-exposed areas (face, neck, scalp)
- Actinic Keratosis occurs on sun-exposed areas especially lower lip and hands
- Dermatofibroma is most common on lower legs
- If model confidence is split between visually similar conditions, use location to differentiate
- Always explain why location supports or changes the diagnosis
{f'- Patient reported lesion on: {body_location}' if body_location else '- No location provided, note this limitation'}
"""
    else:
        specific_instructions = """
IMPORTANT FOR RETINAL SCAN INTERPRETATION:
- Grade diabetic retinopathy clearly (No DR / Mild / Moderate / Severe / Proliferative)
- Always mention urgency of ophthalmology referral based on grade
- Connect findings to diabetes management recommendations
"""

    prompt = f"""You are an expert medical AI assistant generating a structured clinical triage report.

PATIENT: {patient_name}
SCAN TYPE: {scan_label}
RISK LEVEL: {risk_level}
PRIMARY FINDING: {top_finding}{location_context}

CV MODEL FINDINGS:
{findings_text}

RELEVANT MEDICAL LITERATURE:
{rag_context}

{specific_instructions}

Generate a structured clinical report with EXACTLY these sections:

## 📋 SCAN SUMMARY
2-3 sentences summarizing what was analyzed, the primary finding, and what it means clinically. For X-ray, connect radiological terms to clinical diagnosis. For skin, mention location significance if provided.

## 🔍 DETAILED FINDINGS
List each significant finding with its clinical meaning in plain language. Group related findings together. For X-ray, explain which findings together point to the same condition.

## 📍 LOCATION ANALYSIS
(Include ONLY for skin lesion scans)
Explain how the body location {'(' + body_location + ')' if body_location else '(not provided)'} affects the differential diagnosis. Which conditions are more or less likely given the location? If no location provided, explain why location would be important.

## ⚠️ RISK ASSESSMENT
Risk Level: {risk_level}
Explain why this risk level was assigned. For skin, mention if location increases or decreases risk.

## 💊 CLINICAL RECOMMENDATIONS
3-5 specific actionable recommendations. For skin with location, tailor recommendations to that location.

## 👨‍⚕️ REFERRAL SUGGESTION
Which specialist and with what urgency (routine / within 1 month / within 1 week / urgent).

## ℹ️ DISCLAIMER
This is an AI-assisted triage tool. All findings must be confirmed by a qualified medical professional. Do not use this as a substitute for professional medical diagnosis.

Keep tone professional but understandable to a non-medical reader."""

    print(f"[Report Agent] Generating report for {patient_name} via Groq...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a clinical AI assistant generating structured, accurate medical triage reports. Always synthesize multiple findings into clinical diagnoses. Always use body location to refine differential diagnosis for skin lesions. Include appropriate medical disclaimers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000
    )

    report = response.choices[0].message.content
    print(f"[Report Agent] Report generated successfully")
    return report