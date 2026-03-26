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
    """
    if not findings:
        return "No specific findings to look up."

    top_conditions = [f["condition"] for f in findings[:3]]

    # Build query — include location if provided
    if body_location and scan_type == "skin_lesion":
        query = f"skin lesion on {body_location}: {', '.join(top_conditions)} differential diagnosis location"
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

        query_embedding = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(4, count)
        )

        if results and results["documents"]:
            context = "\n\n".join(results["documents"][0])
            print(f"[RAG Agent] Found {len(results['documents'][0])} relevant chunks")

            # If location provided for skin, do a second targeted query
            if body_location and scan_type == "skin_lesion":
                location_query = f"{body_location} skin lesion location differential"
                location_embedding = embedder.encode(location_query).tolist()
                location_results = collection.query(
                    query_embeddings=[location_embedding],
                    n_results=min(2, count)
                )
                if location_results and location_results["documents"]:
                    extra = "\n\n".join(location_results["documents"][0])
                    context = context + "\n\n" + extra
                    print("[RAG Agent] Added location-specific context")

            return context
        else:
            return get_fallback_context(scan_type, top_conditions, body_location)

    except Exception as e:
        print(f"[RAG Agent] Error: {e}")
        return get_fallback_context(scan_type, top_conditions, body_location)


def get_fallback_context(scan_type: str, conditions: list, body_location: str = None) -> str:
    fallbacks = {
        "Pneumonia": "Pneumonia presents as consolidation or infiltrates on X-ray. Infiltration + Consolidation together strongly suggest bacterial pneumonia.",
        "Cardiomegaly": "Cardiomegaly indicates enlarged heart, often associated with heart failure. Requires echocardiogram follow-up.",
        "Effusion": "Pleural effusion is fluid around the lungs. Can indicate heart failure, infection, or malignancy.",
        "Melanoma": "Melanoma is the most dangerous skin cancer. ABCDE rule applies. Can occur anywhere on body.",
        "Basal Cell Carcinoma": "BCC is most common on sun-exposed areas especially face and neck. Rarely on trunk.",
        "Actinic Keratosis": "AK occurs almost exclusively on sun-exposed skin. Lower lip location is high risk for SCC progression.",
        "Dermatofibroma": "Dermatofibroma predominantly found on lower legs. Benign but requires monitoring.",
        "No DR": "No diabetic retinopathy detected. Continue annual screening.",
        "Moderate DR": "Moderate DR requires ophthalmology referral within 3-6 months.",
        "Proliferative DR": "Proliferative DR requires urgent ophthalmology referral within 1 week.",
    }

    context_parts = []
    for condition in conditions:
        if condition in fallbacks:
            context_parts.append(fallbacks[condition])

    if body_location and scan_type == "skin_lesion":
        context_parts.append(
            f"Location context: Lesion reported on {body_location}. "
            f"Location is important for differential diagnosis of skin lesions. "
            f"BCC and AK occur almost exclusively on sun-exposed areas. "
            f"Dermatofibroma is most common on lower legs."
        )

    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return f"Standard clinical evaluation recommended for {', '.join(conditions)}."