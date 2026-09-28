"""Unit tests for Structured Clinical Reporting, JSON Export, and PDF Generation."""

import json
import numpy as np
import pytest
from sanjeevani_ai.preprocessing.quality import QualityReport
from sanjeevani_ai.models.uncertainty import UncertaintyReport
from sanjeevani_ai.reporting.clinical_report import build_screening_report, export_report_to_json
from sanjeevani_ai.reporting.pdf import generate_pdf_report


@pytest.fixture
def mock_screening_data():
    qr = QualityReport(
        is_gradable=True, quality_score=0.91, blur_score=120.0,
        mean_brightness=110.0, glare_percentage=0.2, fov_fraction=0.68,
        rb_ratio=2.2, reasons=[], technician_actions=[]
    )
    ur = UncertaintyReport(
        uncertainty_level="LOW", predictive_entropy=0.22,
        mutual_information=0.008, mc_variance=0.004, tta_variance=0.003,
        is_flagged_for_review=False, clinical_note="Stable prediction.",
        pass_distributions=[]
    )
    c_evidence = [
        {
            "id": "microaneurysms", "name": "Microaneurysms", "category": "NPDR Early",
            "validation_status": "IMPLEMENTED", "dataset_source": "IDRiD",
            "presence_prob": 0.88, "severity_score": 0.70, "attribution_to_grade": 0.40,
            "status_text": "Detected", "badge": "DETECTED", "supports_grade": True
        }
    ]
    quad_info = {
        "quadrant_counts": {"Superior-Temporal": 3, "Inferior-Temporal": 1},
        "total_hotspots": 4, "macular_threat": False, "macular_zone_burden": 0.05
    }
    agree_info = {
        "agreement_score": 0.75, "iou": 0.65, "correlation": 0.80,
        "status": "HIGH_CONSENSUS", "is_flagged": False, "clinical_note": "Consensus"
    }
    cf_info = {
        "downward_transition": {
            "success": True, "current_grade_name": "Moderate NPDR",
            "target_grade_name": "Mild NPDR", "summary": "Requires reduction in hemorrhages",
            "primary_sensitive_concepts": ["Hemorrhages"]
        }
    }
    return qr, ur, c_evidence, quad_info, agree_info, cf_info


def test_build_screening_report_abdm_structure(mock_screening_data):
    qr, ur, c_evidence, quad_info, agree_info, cf_info = mock_screening_data
    report = build_screening_report(
        patient_id="TEST-PID-99", eye_side="OD", quality_report=qr,
        predicted_grade=2, calibrated_probs=[0.05, 0.15, 0.70, 0.08, 0.02],
        uncertainty_report=ur, concept_evidence=c_evidence, quadrant_analysis=quad_info,
        agreement_analysis=agree_info, counterfactual_analysis=cf_info
    )

    assert report["resource_type"] == "DiagnosticReport"
    assert "image_quality_assessment" in report
    assert "dr_screening_result" in report
    assert "uncertainty_quantification" in report
    assert "concept_evidence_observations" in report
    assert "clinical_action_plan" in report


def test_export_report_to_json(mock_screening_data):
    qr, ur, c_evidence, quad_info, agree_info, cf_info = mock_screening_data
    report = build_screening_report(
        patient_id="TEST-PID-99", eye_side="OD", quality_report=qr,
        predicted_grade=2, calibrated_probs=[0.05, 0.15, 0.70, 0.08, 0.02],
        uncertainty_report=ur, concept_evidence=c_evidence, quadrant_analysis=quad_info,
        agreement_analysis=agree_info, counterfactual_analysis=cf_info
    )

    json_str = export_report_to_json(report)
    parsed = json.loads(json_str)
    assert parsed["case_metadata"]["patient_id"] == "TEST-PID-99"


def test_generate_pdf_report(mock_screening_data):
    qr, ur, c_evidence, quad_info, agree_info, cf_info = mock_screening_data
    report = build_screening_report(
        patient_id="TEST-PID-99", eye_side="OD", quality_report=qr,
        predicted_grade=2, calibrated_probs=[0.05, 0.15, 0.70, 0.08, 0.02],
        uncertainty_report=ur, concept_evidence=c_evidence, quadrant_analysis=quad_info,
        agreement_analysis=agree_info, counterfactual_analysis=cf_info
    )

    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    pdf_bytes = generate_pdf_report(report, dummy_img, dummy_img)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
