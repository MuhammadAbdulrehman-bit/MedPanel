import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection(topic: str):
    return client.get_or_create_collection(name=f"medical_{topic}")


def run_rag_agent(scan_type: str, findings: list, body_location: str = None) -> str:
    """
    Agent 2 — RAG Literature Agent.
    Retrieves relevant medical knowledge based on CV findings and body location.
    Enhanced with Clinical Safety Queries to catch potential misclassifications.
    """
    if not findings:
        return "No specific findings to look up."

    top_conditions = [f["condition"] for f in findings[:3]]

    # Build primary query — include location if provided
    if body_location and scan_type == "skin":
        query = f"skin lesion on {body_location}: {', '.join(top_conditions)} differential diagnosis"
        print(f"[RAG Agent] Location-aware query: {query}")
    else:
        query = f"Clinical information about {', '.join(top_conditions)} in {scan_type} imaging"
        print(f"[RAG Agent] Querying knowledge base: {query}")

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
            print("[RAG Agent] Knowledge base empty, using fallback")
            return get_fallback_context(scan_type, top_conditions, body_location)

        # 1. PRIMARY QUERY: Search based on what the CV model found
        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(4, count) # REDUCED from 8 to 4 to stay under token limits
        )

        context = ""
        if results and results["documents"]:
            context = "\n\n".join(results["documents"][0])
            print(f"[RAG Agent] Found {len(results['documents'][0])} relevant chunks for primary findings")

        # 2. CLINICAL SAFETY QUERY: If it's skin, always check high-risk conditions for that location
        if scan_type == "skin" and body_location:
            safety_query = f"High risk malignant skin conditions for {body_location} location BCC SCC AK squamous cell carcinoma"
            print(f"[RAG Agent] Running Safety Query for high-risk location: {body_location}")
            
            safety_embedding = embedder.encode(safety_query).tolist()
            safety_results = collection.query(
                query_embeddings=[safety_embedding],
                n_results=min(2, count) # REDUCED from 4 to 2 to stay under token limits
            )
            
            if safety_results and safety_results["documents"]:
                safety_context = "\n\n--- CLINICAL LOCATION-BASED RISKS ---\n" 
                safety_context += "\n".join(safety_results["documents"][0])
                context = context + "\n" + safety_context
                print("[RAG Agent] Injected safety context for differential diagnosis")

        # FINAL CAP: Ensure context string doesn't exceed ~2500 chars (approx 700 tokens)
        return context[:2500] if context else get_fallback_context(scan_type, top_conditions, body_location)

    except Exception as e:
        print(f"[RAG Agent] Error: {e}")
        return get_fallback_context(scan_type, top_conditions, body_location)


def get_fallback_context(scan_type: str, conditions: list, body_location: str = None) -> str:
    """Fallback medical knowledge if ChromaDB query fails or is empty."""
    fallbacks = {
        "Pneumonia": "Pneumonia presents as consolidation or infiltrates on X-ray. Bacterial pneumonia often shows localized opacity.",
        "Cardiomegaly": "Cardiomegaly indicates an enlarged heart, often assessed via the cardiothoracic ratio (>0.5).",
        "Effusion": "Pleural effusion is fluid in the pleural space, often causing blunting of the costophrenic angles.",
        "Melanoma": "Melanoma is highly malignant. Visual features include asymmetry and irregular borders.",
        "Basal Cell Carcinoma": "BCC is the most common skin cancer, occurring frequently on sun-exposed areas like the face.",
        "Actinic Keratosis": "AK is a precancerous lesion common on sun-exposed skin. Lower lip AK carries higher risk of SCC.",
        "Dermatofibroma": "Dermatofibroma is a benign fibrous nodule, most common on the lower extremities.",
        "No DR": "No diabetic retinopathy detected. Annual screening is standard protocol.",
        "Moderate DR": "Moderate non-proliferative DR requires ophthalmology follow-up within 3-6 months.",
        "Proliferative DR": "Proliferative DR is vision-threatening and requires urgent ophthalmology referral.",
    }

    context_parts = ["--- FALLBACK CLINICAL GUIDELINES ---"]
    for condition in conditions:
        if condition in fallbacks:
            context_parts.append(fallbacks[condition])

    if body_location and scan_type == "skin":
        context_parts.append(
            f"Location Context: Lesions on the {body_location} require careful differential diagnosis. "
            f"Facial and head locations significantly increase the statistical likelihood of BCC or AK."
        )

    return "\n\n".join(context_parts)