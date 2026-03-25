import os
import chromadb
from sentence_transformers import SentenceTransformer
import fitz 

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Medical Knowledge ───────────────────────────────────────────────
# Curated clinical summaries based on PubMed + WHO guidelines

XRAY_KNOWLEDGE = [
    "Pneumonia on chest X-ray presents as consolidation or infiltrates, typically in lower lobes. Bacterial pneumonia shows lobar consolidation while viral pneumonia shows bilateral interstitial infiltrates. Treatment includes antibiotics for bacterial cases and supportive care.",
    "Pleural effusion appears as blunting of costophrenic angles on chest X-ray. Can indicate heart failure, pneumonia, malignancy, or pulmonary embolism. Small effusions may resolve spontaneously while large effusions require thoracentesis.",
    "Cardiomegaly is diagnosed when cardiothoracic ratio exceeds 0.5 on PA chest X-ray. Associated with heart failure, cardiomyopathy, pericardial effusion. Requires echocardiogram for further evaluation.",
    "Atelectasis presents as linear or plate-like opacities on chest X-ray, commonly postoperative. Treatment includes incentive spirometry, chest physiotherapy, and ambulation. Lobar collapse requires bronchoscopy evaluation.",
    "Pneumothorax appears as visceral pleural line with absent lung markings peripherally. Tension pneumothorax is a medical emergency requiring immediate needle decompression. Small pneumothorax may resolve with observation and supplemental oxygen.",
    "Pulmonary edema on chest X-ray shows bilateral perihilar haziness, Kerley B lines, and cardiomegaly. Indicates left heart failure or fluid overload. Requires urgent treatment with diuretics and oxygen therapy.",
    "Lung nodules on chest X-ray require CT follow-up. Nodules under 6mm in low-risk patients can be monitored annually. Nodules over 8mm require PET scan or biopsy. Spiculated margins suggest malignancy.",
    "Consolidation on chest X-ray indicates airspace filling with fluid, pus, blood or cells. Causes include pneumonia, pulmonary edema, hemorrhage, and malignancy. Air bronchograms within consolidation suggest pneumonia.",
    "Emphysema on chest X-ray shows hyperinflation, flattened diaphragms, increased AP diameter, and bullae. Associated with COPD and smoking. Pulmonary function tests confirm diagnosis.",
    "Pleural thickening on chest X-ray may indicate previous infection, asbestos exposure, or malignancy. Mesothelioma presents with unilateral pleural thickening. CT chest recommended for further characterization.",
]

SKIN_KNOWLEDGE = [
    "Melanoma is the most dangerous form of skin cancer, characterized by asymmetry, irregular borders, multiple colors, diameter greater than 6mm, and evolution over time (ABCDE criteria). Early detection dramatically improves survival rates. Wide local excision is primary treatment.",
    "Basal cell carcinoma is the most common skin cancer, appearing as pearly or waxy bumps on sun-exposed areas. Rarely metastasizes but causes local tissue destruction. Treatment options include surgical excision, Mohs surgery, topical chemotherapy, and radiation.",
    "Actinic keratosis are precancerous scaly patches from UV damage. Risk of progression to squamous cell carcinoma is 1-10%. Treatment includes cryotherapy, topical fluorouracil, photodynamic therapy, and laser resurfacing.",
    "Melanocytic nevi are benign moles that require monitoring for changes. Dysplastic nevi have irregular features and higher melanoma risk. Annual skin examination recommended for patients with multiple nevi or family history of melanoma.",
    "Squamous cell carcinoma presents as scaly red patches, open sores, or warts on sun-exposed skin. More aggressive than basal cell carcinoma with metastatic potential. Treatment is surgical excision with clear margins.",
    "Dermatofibroma are benign fibrous nodules commonly on lower legs, firm and hyperpigmented. Dimple sign positive on lateral compression. No treatment required unless symptomatic or cosmetically concerning.",
    "Seborrheic keratosis are benign warty growths with stuck-on appearance. More common with age. No malignant potential. Removal indicated for symptomatic or cosmetically concerning lesions via cryotherapy or curettage.",
    "Vascular lesions include hemangiomas, port-wine stains, and pyogenic granulomas. Infantile hemangiomas involute spontaneously. Port-wine stains associated with Sturge-Weber syndrome. Pyogenic granulomas bleed easily and require removal.",
]

RETINAL_KNOWLEDGE = [
    "Diabetic retinopathy is the leading cause of blindness in working-age adults. Non-proliferative DR shows microaneurysms, dot hemorrhages, and hard exudates. Proliferative DR involves neovascularization with risk of vitreous hemorrhage and tractional retinal detachment.",
    "Mild non-proliferative diabetic retinopathy shows microaneurysms only. Annual dilated eye exam recommended. Optimize blood glucose control with HbA1c target below 7%. Blood pressure control below 130/80 mmHg reduces progression risk.",
    "Moderate non-proliferative diabetic retinopathy shows more extensive microaneurysms, dot and blot hemorrhages, hard exudates, and cotton wool spots. Ophthalmology referral within 3-6 months. Strict glycemic and blood pressure control essential.",
    "Severe non-proliferative diabetic retinopathy defined by 4-2-1 rule: hemorrhages in 4 quadrants, venous beading in 2 quadrants, or intraretinal microvascular abnormalities in 1 quadrant. Urgent ophthalmology referral within 1 month. High risk of progression to PDR.",
    "Proliferative diabetic retinopathy requires urgent ophthalmology referral within 1 week. Treatment includes panretinal photocoagulation laser, anti-VEGF injections, and vitrectomy for non-clearing vitreous hemorrhage or tractional detachment.",
    "Diabetic macular edema can occur at any stage of DR and is the most common cause of vision loss in diabetic patients. First-line treatment is anti-VEGF injections. Focal laser photocoagulation for non-center-involving DME.",
    "Regular diabetic eye screening reduces blindness risk by 90% through early detection and treatment. Screening frequency: annually for type 1 diabetes after 5 years duration, at diagnosis for type 2 diabetes.",
    "Hypertensive retinopathy presents with arteriovenous nicking, copper/silver wiring, flame hemorrhages, and papilledema in severe cases. Indicates systemic hypertension. Blood pressure control is primary treatment.",
]

# def extract_pdf_text(pdf_path: str) -> list:
#     doc = fitz.open(pdf_path)
#     chunks = []
#     for page in doc:
#         text = page.get_text()
#         # Split into paragraphs
#         paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
#         chunks.extend(paragraphs)
#     return chunks

def ingest_knowledge():
    """Build ChromaDB collections from medical knowledge."""

    topics = [
        ("xray", XRAY_KNOWLEDGE),
        ("skin", SKIN_KNOWLEDGE),
        ("retinal", RETINAL_KNOWLEDGE),
    ]

    for topic, documents in topics:
        collection_name = f"medical_{topic}"
        print(f"\n[Ingest] Building collection: {collection_name}")

        # Delete existing collection to rebuild fresh
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(collection_name)

        # Generate embeddings and store
        for i, doc in enumerate(documents):
            embedding = embedder.encode(doc).tolist()
            collection.add(
                ids=[f"{topic}_{i}"],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[{"topic": topic, "index": i}]
            )
            print(f"[Ingest] Added chunk {i+1}/{len(documents)}")

        print(f"[Ingest] ✅ {collection_name} ready with {len(documents)} chunks")

    print("\n[Ingest] ✅ All knowledge bases built successfully")


if __name__ == "__main__":
    ingest_knowledge()