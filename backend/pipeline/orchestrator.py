from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from backend.agents.vision_agent import run_vision_agent
from backend.agents.rag_agent import run_rag_agent
from backend.agents.report_agent import run_report_agent


# ─── State Definition ───────────────────────────────────────────────
# This is the shared state that gets passed between all agents
class MediScanState(TypedDict):
    # Inputs
    patient_name: str
    image_path: str
    scan_hint: Optional[str]       # optional user hint: 'xray', 'skin', 'retinal'

    # Agent 1 outputs
    scan_type: Optional[str]
    findings: Optional[list]
    risk_level: Optional[str]
    top_finding: Optional[str]

    # Agent 2 outputs
    rag_context: Optional[str]

    # Agent 3 outputs
    final_report: Optional[str]

    # Error handling
    error: Optional[str]


# ─── Node Functions (one per agent) ─────────────────────────────────

def vision_node(state: MediScanState) -> MediScanState:
    """Node 1: Run the Vision Triage Agent"""
    print("\n[Orchestrator] Running Vision Agent...")
    try:
        result = run_vision_agent(
            image_path=state["image_path"],
            hint=state.get("scan_hint")
        )
        return {
            **state,
            "scan_type": result["scan_type"],
            "findings": result["findings"],
            "risk_level": result["risk_level"],
            "top_finding": result["top_finding"],
            "error": None
        }
    except Exception as e:
        print(f"[Orchestrator] Vision Agent error: {e}")
        return {**state, "error": str(e)}


def rag_node(state: MediScanState) -> MediScanState:
    """Node 2: Run the RAG Literature Agent"""
    print("\n[Orchestrator] Running RAG Agent...")
    try:
        context = run_rag_agent(
            scan_type=state["scan_type"],
            findings=state["findings"]
        )
        return {**state, "rag_context": context, "error": None}
    except Exception as e:
        print(f"[Orchestrator] RAG Agent error: {e}")
        return {**state, "rag_context": "No additional context available.", "error": None}


def report_node(state: MediScanState) -> MediScanState:
    """Node 3: Run the Report Synthesis Agent"""
    print("\n[Orchestrator] Running Report Agent...")
    try:
        report = run_report_agent(
            patient_name=state["patient_name"],
            scan_type=state["scan_type"],
            findings=state["findings"],
            risk_level=state["risk_level"],
            top_finding=state["top_finding"],
            rag_context=state["rag_context"]
        )
        return {**state, "final_report": report, "error": None}
    except Exception as e:
        print(f"[Orchestrator] Report Agent error: {e}")
        return {**state, "error": str(e)}


# ─── Conditional Edge ────────────────────────────────────────────────

def should_continue(state: MediScanState) -> str:
    """Stop the pipeline if a critical error occurred in vision agent."""
    if state.get("error") and not state.get("scan_type"):
        print("[Orchestrator] Critical error, stopping pipeline.")
        return "end"
    return "continue"


# ─── Build the Graph ─────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(MediScanState)

    # Add nodes
    graph.add_node("vision_agent", vision_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("report_agent", report_node)

    # Entry point
    graph.set_entry_point("vision_agent")

    # Vision → conditional check → RAG
    graph.add_conditional_edges(
        "vision_agent",
        should_continue,
        {
            "continue": "rag_agent",
            "end": END
        }
    )

    # RAG → Report → End
    graph.add_edge("rag_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


# ─── Main Entry Point ────────────────────────────────────────────────

pipeline = build_pipeline()


def run_pipeline(patient_name: str, image_path: str, scan_hint: str = None) -> dict:
    """
    Run the full MediScan pipeline.

    Args:
        patient_name: name of the patient
        image_path: local path to the uploaded image
        scan_hint: optional type hint ('xray', 'skin', 'retinal')

    Returns:
        Final state dict with report and all findings
    """
    initial_state: MediScanState = {
        "patient_name": patient_name,
        "image_path": image_path,
        "scan_hint": scan_hint,
        "scan_type": None,
        "findings": None,
        "risk_level": None,
        "top_finding": None,
        "rag_context": None,
        "final_report": None,
        "error": None
    }

    print(f"\n{'='*50}")
    print(f"[MediScan] Starting pipeline for: {patient_name}")
    print(f"{'='*50}")

    final_state = pipeline.invoke(initial_state)

    print(f"\n{'='*50}")
    print(f"[MediScan] Pipeline complete. Risk: {final_state.get('risk_level')}")
    print(f"{'='*50}\n")

    return final_state