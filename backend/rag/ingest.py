import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Chest X-Ray Knowledge ───────────────────────────────────────────
XRAY_KNOWLEDGE = [
    "Pneumonia on chest X-ray presents as consolidation or infiltrates, typically in lower lobes. Bacterial pneumonia shows lobar consolidation while viral pneumonia shows bilateral interstitial infiltrates. Infiltration + Consolidation findings together are strongly consistent with bacterial pneumonia requiring antibiotic treatment.",
    "Pleural effusion appears as blunting of costophrenic angles on chest X-ray. Can indicate heart failure, pneumonia, malignancy, or pulmonary embolism. Small effusions may resolve spontaneously while large effusions require thoracentesis.",
    "Cardiomegaly is diagnosed when cardiothoracic ratio exceeds 0.5 on PA chest X-ray. Associated with heart failure, cardiomyopathy, pericardial effusion. Requires echocardiogram for further evaluation.",
    "Atelectasis presents as linear or plate-like opacities on chest X-ray, commonly postoperative. Treatment includes incentive spirometry, chest physiotherapy, and ambulation.",
    "Pneumothorax appears as visceral pleural line with absent lung markings peripherally. Tension pneumothorax is a medical emergency requiring immediate needle decompression.",
    "Pulmonary edema on chest X-ray shows bilateral perihilar haziness, Kerley B lines, and cardiomegaly. Indicates left heart failure or fluid overload. Requires urgent treatment with diuretics.",
    "Infiltration combined with fever and productive cough is highly consistent with pneumonia. Unilateral infiltrates suggest bacterial pneumonia. Bilateral infiltrates suggest viral pneumonia or pulmonary edema.",
    "Consolidation with air bronchograms strongly suggests pneumonia. When combined with infiltration findings, clinical correlation with symptoms confirms pneumonia diagnosis.",
    "Lung nodules on chest X-ray require CT follow-up. Nodules under 6mm in low-risk patients can be monitored annually. Nodules over 8mm require PET scan or biopsy.",
    "Emphysema on chest X-ray shows hyperinflation, flattened diaphragms, increased AP diameter, and bullae. Associated with COPD and smoking history.",
]

# ─── Skin Lesion Knowledge — General ────────────────────────────────
SKIN_KNOWLEDGE = [
    "Melanoma is the most dangerous skin cancer characterized by asymmetry, irregular borders, multiple colors, diameter greater than 6mm (ABCDE criteria). Early detection dramatically improves survival. Commonly occurs on back in men and legs in women but can appear anywhere including nail beds.",
    "Basal Cell Carcinoma is the most common skin cancer appearing as pearly or waxy bumps. Predominantly occurs on sun-exposed areas especially face, nose, cheeks, forehead, and neck. Rarely appears on trunk or extremities. Rarely metastasizes but causes local tissue destruction.",
    "Actinic Keratosis are precancerous scaly patches from UV damage. Almost exclusively found on sun-exposed areas: face, scalp, lower lip, back of hands, and forearms. Lower lip location is particularly high risk for progression to squamous cell carcinoma.",
    "Melanocytic Nevus are benign moles requiring monitoring for changes. Can appear anywhere on body. Dysplastic nevi have irregular features and higher melanoma risk. Annual skin examination recommended for patients with multiple nevi.",
    "Dermatofibroma are benign fibrous nodules predominantly found on lower legs, occasionally on arms. Firm and hyperpigmented with positive dimple sign on lateral compression. Rare on face or trunk.",
    "Vascular lesions including hemangiomas and pyogenic granulomas can appear anywhere on body. Common on face, scalp, and extremities. Pyogenic granulomas bleed easily and require removal.",
    "Benign Keratosis are warty growths with stuck-on appearance more common with age. Can appear anywhere but frequently on trunk, face, and extremities. No malignant potential.",
    "Squamous Cell Carcinoma commonly arises from Actinic Keratosis on sun-exposed skin. High risk locations include lower lip, ear, and back of hands. SCC on lip or ear has higher metastatic potential.",
]

# ─── Skin Lesion Knowledge — Location Specific ───────────────────────
SKIN_LOCATION_KNOWLEDGE = [
    "Face lesions differential: BCC is most common on nose, cheeks, and forehead. Melanoma on face often appears as irregular pigmented patch. Actinic Keratosis on face presents as rough scaly patches on forehead and cheeks. Seborrheic Keratosis common on temples.",
    "Lower lip lesions are high risk. Actinic Keratosis on lower lip (actinic cheilitis) can progress to squamous cell carcinoma. BCC rarely occurs on lips. Any persistent lower lip lesion requires urgent biopsy.",
    "Scalp lesions: Actinic Keratosis common on bald scalp. Melanoma can be hidden in hair-bearing scalp. BCC and SCC both common on scalp due to sun exposure.",
    "Back of hands and forearms: Classic location for Actinic Keratosis due to chronic sun exposure. BCC less common here. Melanoma can occur on dorsal hand.",
    "Lower leg lesions: Dermatofibroma is the most common benign lesion on lower legs, especially in women. Vascular lesions also common on legs. Melanoma on legs common in women.",
    "Trunk lesions: Melanocytic Nevus most common on trunk. Seborrheic Keratosis very common on trunk in older adults. Melanoma on back is common in men.",
    "Nail and finger lesions: Subungual Melanoma presents as dark streak under nail. Any new pigmented lesion under nail requires urgent evaluation especially in darker skin types.",
    "Ear lesions: SCC is the most common malignancy on the ear. BCC also occurs on ear. Any ulcerated or non-healing lesion on ear requires biopsy.",
    "Mucous membranes and genitalia: Melanoma can occur on mucous membranes. Any pigmented lesion on genitalia or oral mucosa requires evaluation.",
    "Sun-exposed vs non-sun-exposed: BCC and AK almost exclusively on sun-exposed skin. Melanoma and nevi can occur anywhere including non-sun-exposed areas.",
]

# ─── Retinal Knowledge ───────────────────────────────────────────────
RETINAL_KNOWLEDGE = [
    "Diabetic retinopathy is the leading cause of blindness in working-age adults. Non-proliferative DR shows microaneurysms, dot hemorrhages, and hard exudates. Proliferative DR involves neovascularization with risk of vitreous hemorrhage.",
    "Mild non-proliferative diabetic retinopathy shows microaneurysms only. Annual dilated eye exam recommended. Optimize blood glucose with HbA1c target below 7%.",
    "Moderate non-proliferative diabetic retinopathy shows microaneurysms, dot and blot hemorrhages, hard exudates, and cotton wool spots. Ophthalmology referral within 3-6 months required.",
    "Severe non-proliferative diabetic retinopathy defined by 4-2-1 rule. Urgent ophthalmology referral within 1 month. High risk of progression to proliferative DR.",
    "Proliferative diabetic retinopathy requires urgent ophthalmology referral within 1 week. Treatment includes panretinal photocoagulation laser and anti-VEGF injections.",
    "Diabetic macular edema can occur at any stage and is the most common cause of vision loss. First-line treatment is anti-VEGF injections.",
    "Regular diabetic eye screening reduces blindness risk by 90% through early detection. Screen annually for type 1 diabetes after 5 years, at diagnosis for type 2.",
    "Hypertensive retinopathy presents with arteriovenous nicking, flame hemorrhages, and papilledema in severe cases. Blood pressure control is primary treatment.",
]


def ingest_knowledge():
    topics = [
        ("xray", XRAY_KNOWLEDGE),
        ("skin", SKIN_KNOWLEDGE + SKIN_LOCATION_KNOWLEDGE),
        ("retinal", RETINAL_KNOWLEDGE),
    ]

    for topic, documents in topics:
        collection_name = f"medical_{topic}"
        print(f"\n[Ingest] Building collection: {collection_name}")

        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(collection_name)

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