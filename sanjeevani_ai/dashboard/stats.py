"""
sanjeevani_ai.dashboard.stats
-----------------------------
Rural Vision Camp Population Health & Tele-Ophthalmology Dashboard.

Tracks aggregate metrics across community screening camps:
- Cumulative screened cohort volume
- Autonomous optical gradability rate
- Specialty referral yield (ICDR Grade >= 2)
- Sight-threatening proliferative and macular cases
- Severity stage distribution and recent patient ledger
"""

from typing import Dict, List, Any
import datetime

# Representative realistic cohort data from rural screening camps
DEFAULT_CAMP_PATIENTS: List[Dict[str, Any]] = [
    {
        "id": "PAT-KA-2026-101",
        "name": "Rameshwar Rao",
        "age": 58,
        "gender": "Male",
        "abha_id": "91-4521-8734-9012",
        "diabetes_years": 12,
        "hba1c": 8.6,
        "bp": "142/88",
        "eye": "OD",
        "grade": 2,
        "grade_name": "Moderate NPDR",
        "confidence": 88.4,
        "uncertainty": "LOW",
        "macular_threat": False,
        "gradable": True,
        "referral_needed": True,
        "urgency": "Urgent (1-3 Mo)",
        "matched_hospital": "Narayana Nethralaya",
        "timestamp": "09:15 AM",
    },
    {
        "id": "PAT-KA-2026-102",
        "name": "Savithramma K.",
        "age": 63,
        "gender": "Female",
        "abha_id": "91-3829-1145-4421",
        "diabetes_years": 18,
        "hba1c": 10.2,
        "bp": "158/96",
        "eye": "Bilateral",
        "grade": 4,
        "grade_name": "Proliferative DR",
        "confidence": 92.1,
        "uncertainty": "LOW",
        "macular_threat": True,
        "gradable": True,
        "referral_needed": True,
        "urgency": "Emergency (48h-2 Wk)",
        "matched_hospital": "Minto Ophthalmic Hospital",
        "timestamp": "09:32 AM",
    },
    {
        "id": "PAT-KA-2026-103",
        "name": "Basavaraj G.",
        "age": 51,
        "gender": "Male",
        "abha_id": "91-7721-9981-2311",
        "diabetes_years": 4,
        "hba1c": 6.8,
        "bp": "128/82",
        "eye": "OS",
        "grade": 0,
        "grade_name": "No DR",
        "confidence": 95.8,
        "uncertainty": "LOW",
        "macular_threat": False,
        "gradable": True,
        "referral_needed": False,
        "urgency": "Routine Annual",
        "matched_hospital": "Vision Center (Primary)",
        "timestamp": "09:48 AM",
    },
    {
        "id": "PAT-KA-2026-104",
        "name": "Lakshmi Bai",
        "age": 47,
        "gender": "Female",
        "abha_id": "91-1249-6632-8874",
        "diabetes_years": 7,
        "hba1c": 7.4,
        "bp": "134/84",
        "eye": "OD",
        "grade": 1,
        "grade_name": "Mild NPDR",
        "confidence": 84.2,
        "uncertainty": "MEDIUM",
        "macular_threat": False,
        "gradable": True,
        "referral_needed": False,
        "urgency": "Follow-up (6-12 Mo)",
        "matched_hospital": "District Hospital Eye OPD",
        "timestamp": "10:05 AM",
    },
    {
        "id": "PAT-KA-2026-105",
        "name": "Mohammed Farooq",
        "age": 66,
        "gender": "Male",
        "abha_id": "91-5541-2390-7819",
        "diabetes_years": 21,
        "hba1c": 9.4,
        "bp": "164/92",
        "eye": "Bilateral",
        "grade": 3,
        "grade_name": "Severe NPDR",
        "confidence": 89.7,
        "uncertainty": "LOW",
        "macular_threat": True,
        "gradable": True,
        "referral_needed": True,
        "urgency": "Urgent Laser (2-4 Wk)",
        "matched_hospital": "Narayana Nethralaya",
        "timestamp": "10:22 AM",
    },
    {
        "id": "PAT-KA-2026-106",
        "name": "Parvathamma M.",
        "age": 59,
        "gender": "Female",
        "abha_id": "91-9982-3412-5509",
        "diabetes_years": 9,
        "hba1c": 7.9,
        "bp": "136/86",
        "eye": "OS",
        "grade": 2,
        "grade_name": "Moderate NPDR",
        "confidence": 87.0,
        "uncertainty": "LOW",
        "macular_threat": False,
        "gradable": True,
        "referral_needed": True,
        "urgency": "Urgent (1-3 Mo)",
        "matched_hospital": "Sankara Nethralaya",
        "timestamp": "10:45 AM",
    },
    {
        "id": "PAT-KA-2026-107",
        "name": "Anjanappa N.",
        "age": 70,
        "gender": "Male",
        "abha_id": "91-6634-1029-4482",
        "diabetes_years": 15,
        "hba1c": 8.1,
        "bp": "140/90",
        "eye": "OD",
        "grade": 0,
        "grade_name": "No DR",
        "confidence": 93.5,
        "uncertainty": "LOW",
        "macular_threat": False,
        "gradable": True,
        "referral_needed": False,
        "urgency": "Routine Annual",
        "matched_hospital": "Vision Center (Primary)",
        "timestamp": "11:10 AM",
    },
    {
        "id": "PAT-KA-2026-108",
        "name": "Gangadhara Murthy",
        "age": 54,
        "gender": "Male",
        "abha_id": "91-8841-5523-9031",
        "diabetes_years": 8,
        "hba1c": 7.2,
        "bp": "130/80",
        "eye": "OD",
        "grade": 0,
        "grade_name": "No DR",
        "confidence": 96.2,
        "uncertainty": "LOW",
        "macular_threat": False,
        "gradable": False,
        "referral_needed": False,
        "urgency": "Retake Required",
        "matched_hospital": "Recaptured on site",
        "timestamp": "11:28 AM",
    }
]

# Session memory cache for live additions during testing
_CAMP_REGISTRY: List[Dict[str, Any]] = list(DEFAULT_CAMP_PATIENTS)


def log_screening_record(
    patient_data: Dict[str, Any],
    screening_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Dynamically registers a live screened patient into the camp cohort dashboard.
    """
    is_gradable = screening_result.get("is_gradable", True)
    pred_grade = screening_result.get("predicted_grade", 0)
    grade_label = screening_result.get("grade_label", "Unknown")
    conf = screening_result.get("confidence", 0.85)
    confidence = (conf * 100.0) if conf <= 1.0 else conf
    unc = screening_result.get("uncertainty_report")
    unc_level = getattr(unc, "uncertainty_level", "LOW") if unc else "LOW"
    quad = screening_result.get("quadrant_info", {})
    mac_threat = quad.get("macular_threat", False)

    referral_needed = (pred_grade >= 2) or mac_threat or (unc_level == "HIGH")

    if pred_grade == 4 or mac_threat:
        urgency = "Emergency (48h-2 Wk)"
    elif pred_grade == 3:
        urgency = "Urgent Laser (2-4 Wk)"
    elif pred_grade == 2:
        urgency = "Urgent (1-3 Mo)"
    elif pred_grade == 1:
        urgency = "Follow-up (6-12 Mo)"
    else:
        urgency = "Routine Annual"

    now_str = datetime.datetime.now().strftime("%I:%M %p")

    record = {
        "id": patient_data.get("patient_id", f"PAT-LIVE-{len(_CAMP_REGISTRY)+1:03d}"),
        "name": patient_data.get("patient_name", "Anonymous Patient"),
        "age": patient_data.get("patient_age", 55),
        "gender": patient_data.get("patient_gender", "Other"),
        "abha_id": patient_data.get("abha_id", "Not Provided"),
        "diabetes_years": patient_data.get("diabetes_duration", 5),
        "hba1c": patient_data.get("hba1c", 7.5),
        "bp": f"{patient_data.get('bp_systolic', 130)}/{patient_data.get('bp_diastolic', 82)}",
        "eye": patient_data.get("eye_side", "OD"),
        "grade": pred_grade,
        "grade_name": grade_label,
        "confidence": round(confidence, 1),
        "uncertainty": unc_level,
        "macular_threat": mac_threat,
        "gradable": is_gradable,
        "referral_needed": referral_needed,
        "urgency": urgency,
        "matched_hospital": patient_data.get("matched_hospital", "Narayana Nethralaya"),
        "timestamp": now_str,
    }

    # Avoid duplicate addition of same token
    for idx, existing in enumerate(_CAMP_REGISTRY):
        if existing["id"] == record["id"]:
            _CAMP_REGISTRY[idx] = record
            return record

    _CAMP_REGISTRY.insert(0, record)
    return record


def get_camp_statistics() -> Dict[str, Any]:
    """
    Computes real-time cohort statistics from the screening camp database.
    """
    total = len(_CAMP_REGISTRY)
    if total == 0:
        return {
            "total_screened": 0,
            "gradable_count": 0,
            "ungradable_count": 0,
            "gradability_rate": 100.0,
            "referral_count": 0,
            "referral_rate": 0.0,
            "sight_threatening_count": 0,
            "grade_distribution": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "records": [],
        }

    gradable_count = sum(1 for r in _CAMP_REGISTRY if r.get("gradable", True))
    ungradable_count = total - gradable_count
    gradability_rate = (gradable_count / total) * 100.0

    referral_count = sum(1 for r in _CAMP_REGISTRY if r.get("referral_needed", False))
    referral_rate = (referral_count / total) * 100.0

    sight_threatening_count = sum(
        1 for r in _CAMP_REGISTRY
        if (r.get("grade", 0) >= 3 or r.get("macular_threat", False)) and r.get("gradable", True)
    )

    grade_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for r in _CAMP_REGISTRY:
        if r.get("gradable", True):
            g = min(4, max(0, r.get("grade", 0)))
            grade_dist[g] += 1

    return {
        "total_screened": total,
        "gradable_count": gradable_count,
        "ungradable_count": ungradable_count,
        "gradability_rate": round(gradability_rate, 1),
        "referral_count": referral_count,
        "referral_rate": round(referral_rate, 1),
        "sight_threatening_count": sight_threatening_count,
        "grade_distribution": grade_dist,
        "records": list(_CAMP_REGISTRY),
    }
