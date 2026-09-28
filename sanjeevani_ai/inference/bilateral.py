"""
sanjeevani_ai.inference.bilateral
---------------------------------
Bilateral (OD / OS) Dual-Eye Assessment & Inter-Ocular Asymmetry Engine.

Clinical Rationale:
In systemic diabetes mellitus, retinal microvascular damage is overwhelmingly bilateral
and roughly symmetric. Marked asymmetry (gap of >= 2 ICDR grades between the right eye
OD and left eye OS) is a major clinical red flag indicating potential:
1. Unilateral Carotid Artery Stenosis / Occlusion (Ipsilateral hypoperfusion or ocular ischemia)
2. Prior Asymmetric Laser Photocoagulation or Anti-VEGF Therapy
3. Co-existing Retinal Vein Occlusion (CRVO / BRVO)

Patient-level referral urgency is always governed by the MORE SEVERE eye.
"""

from typing import Dict, Any, Optional
import numpy as np

from ..models.severity_head import ICDR_GRADES


def analyze_bilateral_asymmetry(
    result_od: Dict[str, Any],
    result_os: Dict[str, Any],
    patient_id: str = "PATIENT-BILATERAL"
) -> Dict[str, Any]:
    """
    Synthesizes OD (Right Eye) and OS (Left Eye) examinations into a unified patient-level study.
    """
    grade_od = result_od.get("predicted_grade", 0)
    grade_os = result_os.get("predicted_grade", 0)

    grade_gap = abs(grade_od - grade_os)
    worse_eye = "OD (Right Eye)" if grade_od >= grade_os else "OS (Left Eye)"
    highest_grade = max(grade_od, grade_os)

    is_asymmetric = (grade_gap >= 2)

    # Carotid Stenosis / Ocular Ischemic Alert
    if is_asymmetric:
        clinical_alert = (
            f"⚠ Significant inter-ocular asymmetry detected: OD is {ICDR_GRADES[grade_od]} whereas "
            f"OS is {ICDR_GRADES[grade_os]} (Asymmetry Gap = {grade_gap} grades). "
            f"Diabetic retinopathy is typically symmetric. Marked unilateral disparity warrants urgent "
            f"carotid Doppler ultrasound evaluation to rule out ipsilateral internal carotid artery stenosis "
            f"or ocular ischemic syndrome."
        )
        asymmetry_status = "MARKED_ASYMMETRY (High Clinical Concern)"
    elif grade_gap == 1:
        clinical_alert = (
            f"Mild inter-ocular disparity (OD: {ICDR_GRADES[grade_od]} vs OS: {ICDR_GRADES[grade_os]}). "
            f"Within standard acceptable clinical variation across bilateral eyes."
        )
        asymmetry_status = "MILD_DISPARITY (Acceptable)"
    else:
        clinical_alert = (
            f"Bilateral symmetry confirmed: both eyes exhibit identical severity stage ({ICDR_GRADES[highest_grade]}). "
            f"Consistent with generalized systemic diabetic microvascular burden."
        )
        asymmetry_status = "SYMMETRIC"

    # Patient-Level Action Plan
    referable_patient = (highest_grade >= 2)
    sight_threatening_patient = (
        highest_grade >= 3
        or result_od.get("quadrant_info", {}).get("macular_threat", False)
        or result_os.get("quadrant_info", {}).get("macular_threat", False)
    )

    return {
        "patient_id": patient_id,
        "od_grade": grade_od,
        "od_grade_name": ICDR_GRADES[grade_od],
        "os_grade": grade_os,
        "os_grade_name": ICDR_GRADES[grade_os],
        "asymmetry_gap": grade_gap,
        "asymmetry_status": asymmetry_status,
        "is_asymmetric": is_asymmetric,
        "worse_eye": worse_eye,
        "patient_severity_grade": highest_grade,
        "patient_severity_name": ICDR_GRADES[highest_grade],
        "referable_patient": referable_patient,
        "sight_threatening": sight_threatening_patient,
        "carotid_ischemia_alert": clinical_alert,
    }
