import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Initialize ChromaDB client
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection(topic: str):
    """Get or create a ChromaDB collection for a topic."""
    return client.get_or_create_collection(name=f"medical_{topic}")


def run_rag_agent(scan_type: str, findings: list) -> str:
    """
    Agent 2 — RAG Literature Agent.
    Takes CV findings and retrieves relevant medical knowledge.

    Args:
        scan_type: 'chest_xray', 'skin_lesion', or 'retinal'
        findings: list of dicts with condition + confidence

    Returns:
        string of relevant medical context
    """
    if not findings:
        return "No specific findings to look up."

    # Build a query from top findings
    top_conditions = [f["condition"] for f in findings[:3]]
    query = f"Clinical information about {', '.join(top_conditions)} in {scan_type} imaging"

    print(f"[RAG Agent] Querying knowledge base: {query}")

    # Map scan type to collection topic
    topic_map = {
        "chest_xray": "xray",
        "skin_lesion": "skin",
        "retinal": "retinal"
    }
    topic = topic_map.get(scan_type, "xray")

    try:
        collection = get_collection(topic)
        count = collection.count()

        if count == 0:
            print("[RAG Agent] Knowledge base empty, using fallback context")
            return get_fallback_context(scan_type, top_conditions)

        # Embed the query
        query_embedding = embedder.encode(query).tolist()

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, count)
        )

        if results and results["documents"]:
            context = "\n\n".join(results["documents"][0])
            print(f"[RAG Agent] Found {len(results['documents'][0])} relevant chunks")
            return context
        else:
            return get_fallback_context(scan_type, top_conditions)

    except Exception as e:
        print(f"[RAG Agent] Error querying ChromaDB: {e}")
        return get_fallback_context(scan_type, top_conditions)


def get_fallback_context(scan_type: str, conditions: list) -> str:
    """
    Fallback medical context when ChromaDB is empty.
    Hardcoded clinical summaries for common conditions.
    """
    fallbacks = {
        "Pneumonia": "Pneumonia is an infection that inflames air sacs in lungs. Treatment includes antibiotics for bacterial pneumonia. Chest X-ray shows consolidation or infiltrates.",
        "Cardiomegaly": "Cardiomegaly indicates an enlarged heart, often associated with heart failure, hypertension, or cardiomyopathy. Requires echocardiogram follow-up.",
        "Effusion": "Pleural effusion is fluid accumulation around the lungs. Can indicate heart failure, infection, or malignancy. May require thoracentesis.",
        "Atelectasis": "Atelectasis is partial or complete lung collapse. Common post-surgery. Treatment includes breathing exercises and physiotherapy.",
        "Melanoma": "Melanoma is the most dangerous skin cancer. Early detection is critical. ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution.",
        "Basal Cell Carcinoma": "Most common skin cancer. Rarely metastasizes but locally destructive. Treatment: surgical excision or Mohs surgery.",
        "No DR": "No diabetic retinopathy detected. Continue regular annual screening for diabetic patients.",
        "Moderate DR": "Moderate non-proliferative diabetic retinopathy. Indicates microaneurysms and dot hemorrhages. Referral to ophthalmologist recommended.",
        "Proliferative DR": "Proliferative diabetic retinopathy — advanced stage with new blood vessel growth. Urgent ophthalmology referral required. Risk of vision loss.",
    }

    context_parts = []
    for condition in conditions:
        if condition in fallbacks:
            context_parts.append(fallbacks[condition])

    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return f"Standard clinical evaluation recommended for {', '.join(conditions)}. Consult relevant specialist based on findings."