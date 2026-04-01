import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "rag knowledge")

client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Map filenames to their category ─────────────────────────────────
DISEASE_CATEGORY_MAP = {
    # Chest X-Ray
    "pneumonia": "xray",
    "pleural_effusion": "xray",
    "cardiomegaly": "xray",
    "atelectasis": "xray",
    "pneumothorax": "xray",
    "pulmonary_edema": "xray",
    "emphysema": "xray",
    "consolidation": "xray",

    # Skin Lesion
    "melanoma": "skin",
    "basal_cell_carcinoma": "skin",
    "actinic_keratosis": "skin",
    "melanocytic_nevus": "skin",
    "dermatofibroma": "skin",
    "vascular_lesion": "skin",
    "benign_keratosis": "skin",

    # Retinal
    "no_dr": "retinal",
    "mild_dr": "retinal",
    "moderate_dr": "retinal",
    "severe_dr": "retinal",
    "proliferative_dr": "retinal",
}


def parse_sections(text: str, filename: str) -> list:
    """
    Parse a disease txt file into meaningful chunks.
    Each section becomes its own chunk for better retrieval.
    """
    chunks = []
    current_section = ""
    current_content = []
    disease_name = filename.replace("_", " ").replace(".txt", "").title()

    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect section headers (ALL CAPS lines or lines ending with :)
        is_header = (
            line.isupper() or
            (line.endswith(":") and len(line.split()) <= 4) or
            any(line.startswith(h) for h in [
                "DISEASE:", "CATEGORY:", "ALIASES:", "DESCRIPTION:",
                "VISUAL APPEARANCE:", "TYPICAL LOCATIONS:", "SYMPTOMS:",
                "RISK FACTORS:", "RADIOLOGICAL", "DIFFERENTIAL DIAGNOSIS:",
                "LOCATION", "SEVERITY", "RECOMMENDED ACTION:",
                "COMMON MISTAKES:", "RADIOLOGICAL PROGRESSION:",
                "CLINICAL SIGNS:"
            ])
        )

        if is_header:
            # Save previous section as a chunk
            if current_section and current_content:
                chunk_text = f"{disease_name} - {current_section}\n" + \
                             "\n".join(current_content)
                if len(chunk_text.strip()) > 50:
                    chunks.append(chunk_text.strip())
            current_section = line.rstrip(":")
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section and current_content:
        chunk_text = f"{disease_name} - {current_section}\n" + \
                     "\n".join(current_content)
        if len(chunk_text.strip()) > 50:
            chunks.append(chunk_text.strip())

    # Also add a full summary chunk combining key sections
    full_summary = f"Complete clinical information about {disease_name}:\n{text[:2000]}"
    chunks.append(full_summary)

    return chunks


def ingest_knowledge():
    """
    Reads all .txt files from rag knowledge folder,
    parses them into chunks, and indexes into ChromaDB.
    """
    if not os.path.exists(KNOWLEDGE_PATH):
        print(f"[Ingest] ERROR: Knowledge folder not found at {KNOWLEDGE_PATH}")
        return

    # Delete and recreate collections
    for topic in ["xray", "skin", "retinal"]:
        collection_name = f"medical_{topic}"
        try:
            client.delete_collection(collection_name)
            print(f"[Ingest] Deleted old collection: {collection_name}")
        except Exception:
            pass
        client.create_collection(collection_name)
        print(f"[Ingest] Created collection: {collection_name}")

    # Track counts
    counts = {"xray": 0, "skin": 0, "retinal": 0}
    files_processed = 0

    # Process each txt file
    txt_files = [f for f in os.listdir(KNOWLEDGE_PATH) if f.endswith(".txt")]
    print(f"\n[Ingest] Found {len(txt_files)} knowledge files")

    for filename in sorted(txt_files):
        disease_key = filename.replace(".txt", "").lower()
        topic = DISEASE_CATEGORY_MAP.get(disease_key)

        if not topic:
            print(f"[Ingest] WARNING: No category mapping for {filename} — skipping")
            continue

        filepath = os.path.join(KNOWLEDGE_PATH, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                print(f"[Ingest] WARNING: {filename} is empty — skipping")
                continue

            chunks = parse_sections(text, filename)
            collection = client.get_collection(f"medical_{topic}")

            for i, chunk in enumerate(chunks):
                embedding = embedder.encode(chunk).tolist()
                chunk_id = f"{disease_key}_{i}"
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "disease": disease_key,
                        "topic": topic,
                        "section": i,
                        "filename": filename
                    }]
                )
                counts[topic] += 1

            files_processed += 1
            print(f"[Ingest] ✅ {filename} → {len(chunks)} chunks → medical_{topic}")

        except Exception as e:
            print(f"[Ingest] ERROR processing {filename}: {e}")

    print(f"\n[Ingest] ══════════════════════════════════")
    print(f"[Ingest] Files processed: {files_processed}/{len(txt_files)}")
    print(f"[Ingest] Chunks indexed:")
    print(f"[Ingest]   medical_xray:    {counts['xray']} chunks")
    print(f"[Ingest]   medical_skin:    {counts['skin']} chunks")
    print(f"[Ingest]   medical_retinal: {counts['retinal']} chunks")
    print(f"[Ingest]   TOTAL:           {sum(counts.values())} chunks")
    print(f"[Ingest] ══════════════════════════════════")
    print(f"[Ingest] ✅ Knowledge base built successfully")


if __name__ == "__main__":
    ingest_knowledge()