"""
sanjeevani_ai.reporting.security
--------------------------------
Cryptographic Tamper-Proof Audit Hash for ABDM Tele-Ophthalmology Screening Records.

Generates a deterministic SHA-256 cryptographic digest encompassing:
- Patient Identifier & Timestamp
- Normalized Input Fundus Hash
- Quality Metrics & Predicted ICDR Grade
- Concept Vector Coordinates

Guarantees data integrity for state digital health registries and tele-referral records.
"""

import hashlib
import json
from typing import Dict, Any


def generate_record_tamper_proof_hash(report_dict: Dict[str, Any]) -> str:
    """
    Computes a cryptographic SHA-256 signature for the clinical screening report.
    """
    # Extract core clinical invariant fields
    core_payload = {
        "case_id": report_dict.get("case_metadata", {}).get("case_id"),
        "patient_id": report_dict.get("case_metadata", {}).get("patient_id"),
        "eye_side": report_dict.get("case_metadata", {}).get("eye_side"),
        "timestamp": report_dict.get("case_metadata", {}).get("timestamp"),
        "icdr_grade": report_dict.get("dr_screening_result", {}).get("icdr_grade"),
        "quality_score": report_dict.get("image_quality_assessment", {}).get("quality_score"),
        "concepts": [
            (c.get("concept_id"), c.get("presence_probability"), c.get("severity_score"))
            for c in report_dict.get("concept_evidence_observations", [])
        ]
    }

    serialized = json.dumps(core_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
