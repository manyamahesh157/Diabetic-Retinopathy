"""
sanjeevani_ai.reporting.clinical_report
---------------------------------------
Structured Clinical Screening Report & ABDM/FHIR-Compliant Schema Builder.

Formats the complete screening study into an interoperable JSON representation
designed for integration with the Ayushman Bharat Digital Mission (ABDM) and
tele-ophthalmology Electronic Health Record (EHR) systems.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json

from ..models.severity_head import ICDR_GRADES
from .nlg import generate_clinical_rationale, CLINICAL_RECOMMENDATIONS


def build_screening_report(
    patient_id: str,
    eye_side: str,  # "OD" (Right Eye) or "OS" (Left Eye)
    quality_report: Any,
    predicted_grade: int,
    calibrated_probs: List[float],
    uncertainty_report: Any,
    concept_evidence: List[Dict[str, Any]],
    quadrant_analysis: Dict[str, Any],
    agreement_analysis: Dict[str, Any],
    counterfactual_analysis: Dict[str, Any],
    software_version: str = "1.0.0-hackathon-rc1",
    is_demo_mode: bool = False,
    patient_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs an ABDM/FHIR-ready structured clinical screening report dictionary.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    confidence_pct = round(calibrated_probs[predicted_grade] * 100.0, 1)

    # Generate deterministic textual explanations
    nlg_texts = generate_clinical_rationale(
        predicted_grade=predicted_grade,
        confidence_pct=confidence_pct,
        concept_evidence=concept_evidence,
        uncertainty_level=uncertainty_report.uncertainty_level,
        quadrant_analysis=quadrant_analysis,
        counterfactual_info=counterfactual_analysis,
    )

    rec_meta = CLINICAL_RECOMMENDATIONS[predicted_grade]

    # Overall Human Review Flag logic
    human_review_required = (
        not quality_report.is_gradable
        or uncertainty_report.is_flagged_for_review
        or agreement_analysis.get("is_flagged", False)
        or predicted_grade >= 2
    )

    review_triggers = []
    if not quality_report.is_gradable:
        review_triggers.append("Image failed quality gate.")
    if uncertainty_report.is_flagged_for_review:
        review_triggers.append("High predictive/epistemic uncertainty.")
    if agreement_analysis.get("is_flagged", False):
        review_triggers.append("Explanation methods disagreed.")
    if predicted_grade >= 2:
        review_triggers.append("Referable DR detected (ICDR Grade >= 2).")

    # Construct ABDM/FHIR-compliant structured dictionary
    profile = patient_profile or {}
    report = {
        "resource_type": "DiagnosticReport",
        "standard_system": "ABDM / FHIR-TeleOphthalmology-Draft",
        "case_metadata": {
            "case_id": f"SANJ-{patient_id}-{int(datetime.now(timezone.utc).timestamp())}",
            "patient_id": patient_id,
            "patient_name": profile.get("patient_name", "N/A"),
            "patient_age": profile.get("patient_age", "N/A"),
            "patient_gender": profile.get("patient_gender", "N/A"),
            "abha_id": profile.get("abha_id", "N/A"),
            "contact_number": profile.get("contact_number", "N/A"),
            "diabetes_duration_years": profile.get("diabetes_duration", "N/A"),
            "hba1c_level": profile.get("hba1c", "N/A"),
            "blood_pressure": f"{profile.get('bp_systolic', 'N/A')}/{profile.get('bp_diastolic', 'N/A')}" if "bp_systolic" in profile else "N/A",
            "eye_side": eye_side.upper(),
            "timestamp": timestamp,
            "system_name": "Sanjeevani-AI",
            "software_version": software_version,
            "mode": "DEMO MODE (Illustrative Output)" if is_demo_mode else "RESEARCH MODEL (Not Clinically Validated)",
        },
        "image_quality_assessment": {
            "gradable": quality_report.is_gradable,
            "quality_score": round(quality_report.quality_score, 3),
            "blur_metric": round(quality_report.blur_score, 1),
            "mean_brightness": round(quality_report.mean_brightness, 1),
            "glare_percentage": round(quality_report.glare_percentage, 2),
            "fov_fraction": round(quality_report.fov_fraction, 3),
            "rb_chromaticity_ratio": round(quality_report.rb_ratio, 2),
            "detected_issues": quality_report.reasons,
            "technician_action_guidance": quality_report.technician_actions,
        },
        "dr_screening_result": {
            "icdr_grade": predicted_grade,
            "grade_label": ICDR_GRADES[predicted_grade],
            "confidence_percentage": confidence_pct,
            "class_probabilities": [round(p, 4) for p in calibrated_probs],
            "referable_dr": (predicted_grade >= 2),
            "sight_threatening_dr": (predicted_grade >= 3 or quadrant_analysis.get("macular_threat", False)),
        },
        "uncertainty_quantification": {
            "category": uncertainty_report.uncertainty_level,
            "predictive_entropy": round(uncertainty_report.predictive_entropy, 3),
            "mutual_information_bald": round(uncertainty_report.mutual_information, 4),
            "mc_dropout_variance": round(uncertainty_report.mc_variance, 4),
            "tta_variance": round(uncertainty_report.tta_variance, 4),
            "clinical_significance": uncertainty_report.clinical_note,
        },
        "concept_evidence_observations": [
            {
                "concept_id": c["id"],
                "concept_name": c["name"],
                "category": c["category"],
                "presence_probability": c["presence_prob"],
                "severity_score": c["severity_score"],
                "contribution_weight": c["attribution_to_grade"],
                "evidence_status": c["status_text"],
                "dataset_source": c["dataset_source"],
                "validation_tier": c["validation_status"],
            }
            for c in concept_evidence
        ],
        "anatomical_quadrant_burden": {
            "distribution": quadrant_analysis.get("quadrant_counts", {}),
            "total_focal_hotspots": quadrant_analysis.get("total_hotspots", 0),
            "macular_threat": quadrant_analysis.get("macular_threat", False),
            "macular_burden_score": round(quadrant_analysis.get("macular_zone_burden", 0.0), 3),
        },
        "dual_explainability_consensus": {
            "agreement_score": agreement_analysis.get("agreement_score", 0.0),
            "hotspot_iou": agreement_analysis.get("iou", 0.0),
            "spatial_correlation": agreement_analysis.get("correlation", 0.0),
            "consensus_status": agreement_analysis.get("status", "UNSPECIFIED"),
            "is_flagged": agreement_analysis.get("is_flagged", False),
            "trust_evaluation_note": agreement_analysis.get("clinical_note", ""),
        },
        "counterfactual_analysis": counterfactual_analysis,
        "clinical_action_plan": {
            "urgency_level": rec_meta["urgency"],
            "recommended_timeframe": rec_meta["timeframe"],
            "mandatory_human_review": human_review_required,
            "human_review_triggers": review_triggers,
            "recommended_action": nlg_texts["recommended_action"],
            "patient_plain_language_summary": nlg_texts["patient_explanation"],
        },
        "nlg_narrative": nlg_texts,
        "legal_and_safety_disclaimer": (
            "IMPORTANT: Sanjeevani-AI is an assistive artificial intelligence screening prototype "
            "designed for research and triage assistance in low-resource environments. It is NOT a cleared "
            "diagnostic medical device and does NOT establish a definitive medical diagnosis. All findings, "
            "especially borderline, high-uncertainty, or referable grades, require independent clinical evaluation "
            "by a licensed ophthalmologist."
        ),
    }

    # Cryptographic Tamper-Proof Audit Seal
    from .security import generate_record_tamper_proof_hash
    report["integrity_audit_seal"] = {
        "sha256_hash": generate_record_tamper_proof_hash(report),
        "hash_algorithm": "SHA-256",
        "audit_status": "AUTHENTIC_UNMODIFIED_STUDY",
    }

    return report


def export_report_to_json(report: Dict[str, Any], filepath: Optional[str] = None) -> str:
    """Serializes report dictionary into formatted JSON safely handling numpy types."""
    def _default_encoder(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    json_str = json.dumps(report, indent=2, default=_default_encoder)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str
