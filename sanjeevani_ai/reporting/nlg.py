"""
sanjeevani_ai.reporting.nlg
---------------------------
Deterministic Natural Language Generation (NLG) for Clinical Rationale & Reports.

CRITICAL DESIGN PRINCIPLE:
Never use an unconstrained Large Language Model (LLM) to invent or synthesize medical findings.
All text is deterministically derived from model-extracted concept activations, quadrant
counts, uncertainty levels, and the official International Clinical Diabetic Retinopathy (ICDR) scale.
"""

from typing import Dict, Any, List
import numpy as np

ICDR_CLINICAL_CRITERIA = {
    0: (
        "No apparent retinopathy. Retinal vascular architecture, optic disc margin, and foveal reflex "
        "appear within standard limits with no visible microvascular lesions."
    ),
    1: (
        "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Characterized by isolated microaneurysms "
        "(tiny red capillary outpouchings). No definite retinal hemorrhages, hard exudates, or venous abnormalities."
    ),
    2: (
        "Moderate Non-Proliferative Diabetic Retinopathy (NPDR). Presence of microaneurysms accompanied by "
        "scattered intraretinal hemorrhages and/or hard exudates (lipid deposits). Findings exceed mild NPDR but "
        "do not meet the 4-2-1 threshold for severe disease."
    ),
    3: (
        "Severe Non-Proliferative Diabetic Retinopathy (NPDR). Meets one or more criteria of the clinical 4-2-1 rule: "
        "extensive intraretinal hemorrhages (>20 per quadrant in all 4 quadrants), definite venous beading in >=2 quadrants, "
        "or prominent IRMA in >=1 quadrant. High impending risk of neovascular proliferation."
    ),
    4: (
        "Proliferative Diabetic Retinopathy (PDR). Sight-threatening condition characterized by pathological "
        "neovascularization (NVD at the optic disc or NVE elsewhere on the retina), with or without preretinal/vitreous hemorrhage."
    ),
}

CLINICAL_RECOMMENDATIONS = {
    0: {
        "urgency": "Routine Surveillance",
        "timeframe": "Annual repeat screening in 12 months",
        "action": "Reinforce strict glycemic and blood pressure control. Continue annual dilated retinal photographic screening.",
        "level": "LOW_RISK"
    },
    1: {
        "urgency": "Close Follow-Up",
        "timeframe": "Re-screen in 9 to 12 months",
        "action": "Optimize systemic metabolic control (HbA1c, BP, lipid panel). Repeat photographic screening within 9-12 months.",
        "level": "MILD_RISK"
    },
    2: {
        "urgency": "Ophthalmology Consultation",
        "timeframe": "Specialist evaluation within 3 to 6 months",
        "action": "Refer to an ophthalmologist for comprehensive dilated fundus examination and baseline OCT macular evaluation.",
        "level": "MODERATE_RISK"
    },
    3: {
        "urgency": "High Priority Specialist Referral",
        "timeframe": "Ophthalmologist evaluation within 2 to 4 weeks",
        "action": "Prompt evaluation by a retina specialist. Assessment for panretinal photocoagulation (PRP) or anti-VEGF therapy readiness.",
        "level": "HIGH_RISK"
    },
    4: {
        "urgency": "URGENT SIGHT-THREATENING REFERRAL",
        "timeframe": "Immediate ophthalmologist review within 48 to 72 hours",
        "action": "Emergency retina specialist referral for immediate evaluation, fluorescein angiography, and urgent intervention (PRP / Anti-VEGF / Vitrectomy).",
        "level": "CRITICAL_RISK"
    },
}


def generate_clinical_rationale(
    predicted_grade: int,
    confidence_pct: float,
    concept_evidence: List[Dict[str, Any]],
    uncertainty_level: str,
    quadrant_analysis: Dict[str, Any],
    counterfactual_info: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generates deterministic clinical text sections from structured model evidence.
    """
    detected_concepts = [c["name"] for c in concept_evidence if c["presence_prob"] >= 0.40]
    high_concepts = [c["name"] for c in concept_evidence if c["presence_prob"] >= 0.70]

    # 1. Evidence Summary
    if high_concepts:
        concept_summary = f"Definite visual evidence detected for: {', '.join(high_concepts)}."
    elif detected_concepts:
        concept_summary = f"Suggestive/moderate evidence detected for: {', '.join(detected_concepts)}."
    else:
        concept_summary = "No definite microvascular lesion patterns identified across the inspected field."

    # 2. Quadrant Narrative
    q_counts = quadrant_analysis.get("quadrant_counts", {})
    active_quadrants = [q for q, count in q_counts.items() if count > 0]
    if len(active_quadrants) >= 3:
        quadrant_summary = (
            f"Lesion distribution is widespread, involving {len(active_quadrants)} retinal quadrants: "
            f"{', '.join(active_quadrants)}."
        )
    elif active_quadrants:
        quadrant_summary = (
            f"Lesion focus is localized primarily in the: {', '.join(active_quadrants)}."
        )
    else:
        quadrant_summary = "Uniform retinal background with no focal lesion clustering."

    # 3. Macular Threat Note
    if quadrant_analysis.get("macular_threat", False):
        macular_note = (
            "⚠ Significant exudation or edema signature observed in the macular/perifoveal zone. "
            "High risk of Clinically Significant Macular Edema (CSME); prioritized specialist OCT evaluation required."
        )
    else:
        macular_note = "Central macular region currently free of primary exudative clustering."

    # 4. Severity Reasoning
    reasoning = (
        f"The model assigned {predicted_grade} based on the observed lesion composition. "
        f"{ICDR_CLINICAL_CRITERIA[predicted_grade]} {concept_summary} {quadrant_summary}"
    )

    # 5. Borderline / Counterfactual Commentary
    down_cf = counterfactual_info.get("downward_transition")
    if down_cf and down_cf.get("success"):
        cf_note = (
            f"Sensitivity analysis: The decision boundary between {down_cf['current_grade_name']} "
            f"and {down_cf['target_grade_name']} is primarily modulated by "
            f"{', '.join(down_cf.get('primary_sensitive_concepts', []))}."
        )
    else:
        cf_note = "Decision boundary is well-separated from neighboring ordinal severity stages."

    rec = CLINICAL_RECOMMENDATIONS[predicted_grade]

    # 6. Patient-Friendly Translation
    patient_text = (
        f"Your retinal screening photograph shows findings consistent with {rec['urgency'].lower()}. "
        f"Your screening result indicates: {rec['timeframe']}. "
        f"Please share this report with your eye doctor or physician. "
        f"Maintaining healthy blood sugar, cholesterol, and blood pressure levels is the best way to protect your vision."
    )

    return {
        "concept_summary": concept_summary,
        "quadrant_summary": quadrant_summary,
        "macular_note": macular_note,
        "severity_reasoning": reasoning,
        "counterfactual_note": cf_note,
        "recommended_action": rec["action"],
        "urgency_timeframe": rec["timeframe"],
        "patient_explanation": patient_text,
    }
