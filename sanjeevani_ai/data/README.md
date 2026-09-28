# Sanjeevani-AI Dataset Documentation & Clinical Pipeline

## 1. Supported Clinical Datasets

| Dataset | Primary Use | Label Type | Notes on Lesions |
|---|---|---|---|
| **APTOS 2019 Blindness Detection** | Severity Classification | Patient/Image-level ICDR (0–4) | **No lesion annotations.** Used for global DR grading and transfer learning. |
| **IDRiD (Indian Diabetic Retinopathy Image Dataset)** | Concept Head Supervised Training | ICDR (0–4) + Lesion Pixel Segmentations | Contains ground-truth masks for **Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates** captured in Indian eye clinics. |
| **Messidor-2** | External Out-of-Distribution Validation | ICDR (0–4) + Macular Edema risk | Used for external generalization testing to detect dataset shift across international clinics. |

## 2. Patient-Level Splitting Principle

A critical failure in amateur ML prototypes is **patient leakage** (e.g. putting the right eye of patient X in train, and the left eye in test). This inflates validation accuracy through patient-specific background features (choroidal melanin, optical disc geometry) rather than learning diabetic retinal pathology.

Sanjeevani-AI's `partition_manifests_by_patient` enforces:
- `GroupShuffleSplit` on `patient_id`.
- All images from any individual patient are strictly quarantined to a single partition (`train`, `val`, or `test`).

## 3. Ground-Truth Concepts vs Pseudo-Labels

Sanjeevani-AI strictly distinguishes:
- **IMPLEMENTED (IDRiD Annotations)**: Microaneurysms, Hemorrhages, Hard Exudates, Cotton-Wool Spots, Neovascularization, Macular Involvement.
- **EXPERIMENTAL**: Venous Beading, IRMA (Intraretinal Microvascular Abnormalities) — trained via weak ordinal supervision and prior rules.
- **NOT TRAINED / NOT VALIDATED**: Any concept without verified training annotations is clearly surfaced in the UI as experimental.
