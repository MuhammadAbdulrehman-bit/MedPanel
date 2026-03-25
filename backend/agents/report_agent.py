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
    rag_context: str
) -> str:
    """
    Agent 3 — Report Synthesis Agent.
    Uses Groq (Llama 3) to generate a structured clinical report.

    Args:
        patient_name: name of the patient
        scan_type: type of scan performed
        findings: list of CV model findings
        risk_level: Low / Medium / High
        top_finding: primary detected condition
        rag_context: medical literature from RAG agent

    Returns:
        Structured clinical report as a string
    """

    # Format findings for the prompt
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

    prompt = f"""You are an expert medical AI assistant generating a structured clinical triage report.

PATIENT: {patient_name}
SCAN TYPE: {scan_label}
RISK LEVEL: {risk_level}
PRIMARY FINDING: {top_finding}

CV MODEL FINDINGS:
{findings_text}

RELEVANT MEDICAL LITERATURE:
{rag_context}

Generate a structured clinical report with EXACTLY these sections:

## 📋 SCAN SUMMARY
Brief 2-3 sentence summary of what was analyzed and the primary finding.

## 🔍 DETAILED FINDINGS
List each finding with its significance. Explain what each condition means in plain language.

## ⚠️ RISK ASSESSMENT
Risk Level: {risk_level}
Explain why this risk level was assigned and what it means for the patient.

## 💊 CLINICAL RECOMMENDATIONS
3-5 specific actionable recommendations based on the findings.

## 👨‍⚕️ REFERRAL SUGGESTION
Which specialist should the patient see, and with what urgency (routine / soon / urgent).

## ℹ️ DISCLAIMER
Standard medical disclaimer that this is an AI-assisted triage tool and not a substitute for professional medical diagnosis.

Keep the tone professional but understandable to a non-medical reader. Be concise and specific."""

    print(f"[Report Agent] Generating report for {patient_name} via Groq...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a clinical AI assistant that generates structured, accurate, and professional medical triage reports. Always include appropriate medical disclaimers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1500
    )

    report = response.choices[0].message.content
    print(f"[Report Agent] Report generated successfully")
    return report