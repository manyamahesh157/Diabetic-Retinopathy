"""Unit tests for Novel Features: Bilateral Engine, Digital Loupe, Multi-Model Consensus, Multilingual Leaflet, SHA-256 Seal."""

import numpy as np
import torch
import pytest

from sanjeevani_ai.inference.bilateral import analyze_bilateral_asymmetry
from sanjeevani_ai.models.consensus import ModelConsensusEngine
from sanjeevani_ai.models.sanjeevani_model import SanjeevaniModel
from sanjeevani_ai.xai.loupe import extract_digital_loupe
from sanjeevani_ai.reporting.multilingual import get_multilingual_patient_guidance
from sanjeevani_ai.reporting.security import generate_record_tamper_proof_hash


def test_bilateral_asymmetry_symmetric():
    res_od = {"predicted_grade": 2, "quadrant_info": {"macular_threat": False}}
    res_os = {"predicted_grade": 2, "quadrant_info": {"macular_threat": False}}

    summary = analyze_bilateral_asymmetry(res_od, res_os, patient_id="TEST-SYM")
    assert summary["asymmetry_gap"] == 0
    assert summary["is_asymmetric"] is False
    assert "SYMMETRIC" in summary["asymmetry_status"]


def test_bilateral_asymmetry_carotid_alert():
    # OD is Grade 4, OS is Grade 1 -> Gap of 3 grades
    res_od = {"predicted_grade": 4, "quadrant_info": {"macular_threat": True}}
    res_os = {"predicted_grade": 1, "quadrant_info": {"macular_threat": False}}

    summary = analyze_bilateral_asymmetry(res_od, res_os, patient_id="TEST-ASYM")
    assert summary["asymmetry_gap"] == 3
    assert summary["is_asymmetric"] is True
    assert "carotid" in summary["carotid_ischemia_alert"].lower()
    assert summary["patient_severity_grade"] == 4


def test_digital_loupe_magnification():
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    loupe = extract_digital_loupe(img, center_xy=(100, 100), patch_size=32, zoom_factor=4)

    assert loupe["zoomed_rgb"].shape == (128, 128, 3)
    assert loupe["green_enhanced"].shape == (128, 128)
    assert len(loupe["transect_profile"]) == 128


def test_multimodel_consensus_engine():
    model = SanjeevaniModel(backbone_name="efficientnet_b0", pretrained=False)
    engine = ModelConsensusEngine(model)
    tensor = torch.randn(1, 3, 224, 224)
    probs = np.array([0.05, 0.15, 0.70, 0.08, 0.02])
    c_probs = np.array([0.8, 0.7, 0.6, 0.1, 0.0, 0.0, 0.0, 0.2])

    consensus = engine.evaluate_consensus(tensor, probs, c_probs)
    assert "consensus_grade" in consensus
    assert consensus["agreement_ratio"] > 0.0
    assert len(consensus["models_breakdown"]) == 3


def test_multilingual_guidance():
    hindi = get_multilingual_patient_guidance(grade=2, language="Hindi (हिंदी)")
    assert "नेत्र" in hindi["title"] or "जांच" in hindi["title"]

    kannada = get_multilingual_patient_guidance(grade=4, language="Kannada (ಕನ್ನಡ)")
    assert "ಕಣ್ಣಿನ" in kannada["title"] or "ತುರ್ತು" in kannada["title"]


def test_tamper_proof_sha256_seal():
    report = {
        "case_metadata": {"case_id": "SANJ-001", "patient_id": "P1", "eye_side": "OD", "timestamp": "2026-09-28T10:00:00Z"},
        "dr_screening_result": {"icdr_grade": 2},
        "image_quality_assessment": {"quality_score": 0.95},
        "concept_evidence_observations": [
            {"concept_id": "microaneurysms", "presence_probability": 0.85, "severity_score": 0.7}
        ]
    }
    h1 = generate_record_tamper_proof_hash(report)
    assert len(h1) == 64  # Valid SHA-256 hexadecimal string

    # Modifying the grade must change the hash (tamper detection)
    report["dr_screening_result"]["icdr_grade"] = 1
    h2 = generate_record_tamper_proof_hash(report)
    assert h1 != h2


def test_hospital_referral_matching():
    from sanjeevani_ai.referral.hospitals import get_recommended_referrals

    # Severe case should trigger RED_EMERGENCY and Apex Tertiary centers
    ref_severe = get_recommended_referrals(severity_grade=4, region="Karnataka / Bengaluru", has_macular_threat=True)
    assert ref_severe["urgency_code"] == "RED_EMERGENCY"
    assert len(ref_severe["hospitals"]) >= 1
    assert "ABDM-REF-RED" in ref_severe["abdm_referral_token"]

    # Mild case should trigger routine or follow-up
    ref_mild = get_recommended_referrals(severity_grade=1, region="All India")
    assert ref_mild["urgency_code"] == "YELLOW_FOLLOWUP"
    assert len(ref_mild["hospitals"]) >= 1


def test_camp_dashboard_stats_and_logging():
    from sanjeevani_ai.dashboard.stats import get_camp_statistics, log_screening_record

    stats = get_camp_statistics()
    initial_total = stats["total_screened"]
    assert initial_total >= 5
    assert 0 <= stats["gradability_rate"] <= 100.0
    assert 0 <= stats["referral_rate"] <= 100.0

    # Test dynamic registration
    dummy_patient = {
        "patient_id": "TEST-DASH-999",
        "patient_name": "Test Screening Patient",
        "patient_age": 60,
        "patient_gender": "Male",
        "abha_id": "91-0000-1111-2222",
        "diabetes_duration": 10,
        "hba1c": 8.0,
        "bp_systolic": 135,
        "bp_diastolic": 85,
        "eye_side": "OD",
    }
    dummy_result = {
        "is_gradable": True,
        "predicted_grade": 3,
        "grade_label": "Severe NPDR",
        "confidence": 0.91,
        "quadrant_info": {"macular_threat": True},
    }
    rec = log_screening_record(dummy_patient, dummy_result)
    assert rec["id"] == "TEST-DASH-999"
    assert rec["referral_needed"] is True

    new_stats = get_camp_statistics()
    assert new_stats["total_screened"] == initial_total + 1


def test_multilingual_translations_dictionary():
    from sanjeevani_ai.i18n.translations import get_ui_text, TRANSLATIONS

    langs = ["English", "Hindi (हिंदी)", "Kannada (ಕನ್ನಡ)", "Tamil (தமிழ்)", "Telugu (తెలుగు)"]
    for lang in langs:
        assert lang in TRANSLATIONS
        title = get_ui_text("app_title", lang=lang)
        assert len(title) > 0
        dash_title = get_ui_text("dashboard_title", lang=lang)
        assert len(dash_title) > 0
        hosp_title = get_ui_text("hospitals_title", lang=lang)
        assert len(hosp_title) > 0

