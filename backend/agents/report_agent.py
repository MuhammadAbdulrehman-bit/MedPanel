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

    # ─── Scan-specific reasoning instructions ────────────────────────
    if scan_type == "chest_xray":
        specific_instructions = """
IMPORTANT FOR CHEST X-RAY INTERPRETATION:
- Infiltration + Consolidation together = strong clinical indicator of pneumonia.
- Infiltration alone could be pneumonia, pulmonary edema, or hemorrhage — 
  use RAG context to differentiate.
- Effusion + Cardiomegaly together suggests heart failure, not infection.
- Upper lobe findings: consider TB or atypical pneumonia.
- Bilateral symmetric findings: consider pulmonary edema over pneumonia.
- Synthesize ALL radiological findings into ONE clinical picture.
- Never just list findings — always explain what they mean together.
- If findings contradict each other clinically, say so and explain why.
- Always mention what conditions were ruled out and why.
"""
    elif scan_type in ["skin_lesion", "skin"]:
        specific_instructions = f"""
IMPORTANT FOR SKIN LESION INTERPRETATION:
- Body location is the MOST CRITICAL clinical anchor for skin diagnosis.
- BCC and Actinic Keratosis are statistically dominant on sun-exposed areas 
  (face, nose, ears, neck, scalp, lower lip, dorsal hands).
- Dermatofibroma is most common on lower legs.
- Melanoma is most common on back (men) and legs (women).
- If CV model suggests benign Nevus but location is face/lower lip/ear, 
  you MUST flag BCC or AK as clinical suspicion regardless of confidence.
- Always apply ABCDE criteria reasoning for pigmented lesions.
- Highlight any contradictions between visual confidence and location probability.
- Consider patient age and sun exposure history if provided.
{f'- Patient reported lesion location: {body_location}' if body_location else '- WARNING: No body location provided. Note this as a significant limitation — location is essential for accurate skin diagnosis.'}
"""
    else:
        specific_instructions = """
IMPORTANT FOR RETINAL SCAN INTERPRETATION:
- Grade diabetic retinopathy clearly: No DR / Mild / Moderate / Severe / Proliferative.
- Ensure referral urgency strictly matches DR grade:
  No DR / Mild → Annual screening
  Moderate → Ophthalmology within 3-6 months
  Severe → Ophthalmology within 1 month
  Proliferative → URGENT ophthalmology within 1 week
- Always connect retinal findings to diabetes management.
- Mention macular edema risk at Moderate+ grades.
- Emphasize that good glycemic control slows progression.
"""

    # ─── Core reasoning prompt ────────────────────────────────────────
    prompt = f"""You are a Senior Clinical Diagnostic Agent. Your task is to 
REASON about medical image findings — not just format them.

═══════════════════════════════════════════
PATIENT DATA:
═══════════════════════════════════════════
Patient Name: {patient_name}
Scan Type: {scan_label}
Body Location: {body_location if body_location else 'Not Provided'}
Primary CV Finding: {top_finding}
CV Risk Level: {risk_level}

═══════════════════════════════════════════
CV MODEL OUTPUTS (visual analysis only):
═══════════════════════════════════════════
{findings_text}

═══════════════════════════════════════════
CLINICAL KNOWLEDGE BASE (RAG):
═══════════════════════════════════════════
{rag_context}

═══════════════════════════════════════════
RECONCILIATION RULES:
═══════════════════════════════════════════
1. CV MODEL IS A SUGGESTER: It analyzes pixels only. It can be fooled by 
   visual similarities. Treat its output as one data point, not the answer.

2. RAG CONTEXT IS CLINICAL TRUTH: Use medical literature to validate OR 
   challenge the CV prediction. If they agree, confirm it. If they conflict, 
   explain which is more likely and why.

3. MALIGNANCY OVERRIDE: If CV predicts benign but location or literature 
   suggests malignancy risk — you MUST flag this clearly. Patient safety 
   takes absolute priority over model confidence.

4. CONFIDENCE MATTERS: A finding at 95% confidence is very different from 
   35%. Reflect this uncertainty in your clinical language.

5. DIFFERENTIAL DIAGNOSIS: Always mention the top 2-3 alternative diagnoses 
   from RAG context. Explain what would differentiate them clinically.

6. HONEST UNCERTAINTY: If evidence is ambiguous, say so clearly rather than 
   forcing a confident conclusion. Use language like "consistent with", 
   "cannot exclude", "warrants further evaluation".

{specific_instructions}

═══════════════════════════════════════════
GENERATE REPORT WITH EXACTLY THESE SECTIONS:
═══════════════════════════════════════════

## 📋 SCAN SUMMARY
2-3 sentences. State the primary CV finding and its confidence. If there is 
a contradiction between CV output and clinical context, clearly state:
"While the visual model suggests [X] with [Y]% confidence, the clinical 
location of [location] and medical literature raises concern for [Z]."

## 🔍 DETAILED FINDINGS
Explain each significant CV finding in plain language. Group related findings 
together. Explain what they mean clinically, not just what they are.
For X-ray: what do these findings together suggest?
For skin: what do the confidence scores tell us about certainty?
For retinal: what stage does this represent?

## 🧠 CLINICAL REASONING
This is the most important section. 
- Does the CV prediction align with medical literature for this case?
- What does the RAG context say about the top finding?
- Are there any red flags the CV model may have missed?
- What is the most likely clinical diagnosis considering ALL evidence?
- What are the top 2 differential diagnoses and how would you distinguish them?

## 📍 LOCATION ANALYSIS
(Include for skin scans. For xray/retinal, briefly mention relevant 
anatomical location significance.)
For skin: How does the body location change the diagnosis probability?
Which conditions are more or less likely at this specific location?
Does the location support or contradict the CV model's prediction?

## ⚠️ RISK ASSESSMENT
Risk Level: {risk_level}
Explain the reasoning behind the risk level. If clinical context increases 
risk beyond what the CV model calculated, explicitly state the upgraded risk 
and why. Use the RAG evidence to justify.

## 💊 CLINICAL RECOMMENDATIONS
3-5 specific, actionable recommendations tailored to:
- The specific findings
- The body location (for skin)
- The patient's risk profile
Number each recommendation clearly.

## 👨‍⚕️ REFERRAL SUGGESTION
Specify: which specialist, urgency level, and what information to bring.
Urgency options: Routine (weeks) / Soon (1-2 weeks) / Urgent (days) / 
Emergency (same day)

## ℹ️ DISCLAIMER
This report was generated by an AI-assisted triage tool using computer 
vision and medical knowledge retrieval. All findings must be confirmed by 
a qualified medical professional. This does not constitute a medical 
diagnosis and should not replace clinical examination.

Keep the tone professional, objective, and appropriately cautious.
Prioritize patient safety in all recommendations."""

    print(f"[Report Agent] Reasoning clinical report for {patient_name}...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a Senior Clinical Diagnostic Agent with expertise 
in radiology, dermatology, and ophthalmology. Your role is to REASON about 
medical findings, not just format them.

Your core responsibilities:
- Challenge CV model outputs using medical literature and clinical context
- Identify when visual predictions conflict with clinical probability  
- Always consider differential diagnoses beyond the top CV prediction
- Prioritize patient safety — flag malignancy risks even when uncertain
- Be honest about uncertainty rather than projecting false confidence
- Use location, symptoms, and risk factors to refine diagnosis probability

You think like a senior clinician reviewing a junior resident's assessment.
You never blindly accept the CV model's output — you interrogate it."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    report = response.choices[0].message.content
    print(f"[Report Agent] Clinical reasoning report generated successfully")
    return report