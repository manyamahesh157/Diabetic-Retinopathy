"""
sanjeevani_ai.reporting.pdf
---------------------------
Certified Clinical PDF Screening Report Generator using ReportLab.

Creates an archival, publication-grade screening summary document suitable for
inclusion in tele-ophthalmology referral packets or physical patient files.
"""

import io
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)


def _convert_numpy_to_rl_image(np_rgb_uint8: np.ndarray, width: float, height: float) -> RLImage:
    """Encodes a numpy RGB uint8 image into an in-memory PNG flowable for ReportLab."""
    pil_img = Image.fromarray(np_rgb_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def generate_pdf_report(
    report_dict: Dict[str, Any],
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    output_path: Optional[str] = None
) -> bytes:
    """
    Generates a PDF screening report.
    Returns bytes of the compiled PDF.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2563EB")
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569")
    )
    section_head = ParagraphStyle(
        "SecHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1E293B")
    )
    body_text = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155")
    )
    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_text,
        fontName="Helvetica-Bold"
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#DC2626")
    )

    story = []

    # 1. Header Banner
    header_table_data = [
        [
            Paragraph("SANJEEVANI-AI CLINICAL SCREENING REPORT", title_style),
            Paragraph("AYUSHMAN BHARAT DIGITAL MISSION (ABDM) READY<br/>TELE-OPHTHALMOLOGY TRIAGE", subtitle_style)
        ]
    ]
    t_head = Table(header_table_data, colWidths=[340, 200])
    t_head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(t_head)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # 2. Case Demographics & Meta
    meta = report_dict.get("case_metadata", {})
    dr = report_dict.get("dr_screening_result", {})
    unc = report_dict.get("uncertainty_quantification", {})
    qa = report_dict.get("image_quality_assessment", {})

    meta_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {meta.get('patient_id', 'N/A')}", meta_style),
            Paragraph(f"<b>Eye Examined:</b> {meta.get('eye_side', 'OD')} (Oculus Dexter)", meta_style),
            Paragraph(f"<b>Timestamp:</b> {meta.get('timestamp', '')[:19]}", meta_style),
        ],
        [
            Paragraph(f"<b>Patient Name:</b> {meta.get('patient_name', 'N/A')}", meta_style),
            Paragraph(f"<b>Age / Sex:</b> {meta.get('patient_age', 'N/A')} Y / {meta.get('patient_gender', 'N/A')}", meta_style),
            Paragraph(f"<b>ABHA ID:</b> {meta.get('abha_id', 'N/A')}", meta_style),
        ],
        [
            Paragraph(f"<b>Diabetes Duration:</b> {meta.get('diabetes_duration_years', 'N/A')} Yrs", meta_style),
            Paragraph(f"<b>HbA1c / BP:</b> {meta.get('hba1c_level', 'N/A')}% | {meta.get('blood_pressure', 'N/A')}", meta_style),
            Paragraph(f"<b>Operating Mode:</b> {meta.get('mode', 'Demo')[:24]}", meta_style),
        ],
        [
            Paragraph(f"<b>Case Ref:</b> {meta.get('case_id', 'N/A')}", meta_style),
            Paragraph(f"<b>Image Quality:</b> {'PASS (Gradable)' if qa.get('gradable') else 'FAIL (Rejected)'}", meta_style),
            Paragraph(f"<b>ABDM Token:</b> {meta.get('patient_id', 'REF')}-2026", meta_style),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[180, 180, 180])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 3. Side-by-Side Fundus Photographs
    img_w, img_h = 2.4 * inch, 2.4 * inch
    rl_orig = _convert_numpy_to_rl_image(original_rgb, img_w, img_h)
    rl_overlay = _convert_numpy_to_rl_image(overlay_rgb, img_w, img_h)

    images_table_data = [
        [
            Paragraph("<b>Original Preprocessed Retina</b>", meta_style),
            Paragraph("<b>Dual-XAI Saliency Overlay (Grad-CAM++ / IG)</b>", meta_style)
        ],
        [rl_orig, rl_overlay]
    ]
    t_imgs = Table(images_table_data, colWidths=[270, 270])
    t_imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 10))

    # 4. Screening Findings & Uncertainty Indicator
    grade_idx = dr.get("icdr_grade", 0)
    grade_label = dr.get("grade_label", "Unknown")
    conf_pct = dr.get("confidence_percentage", 0.0)

    # Color mapping for grades
    grade_colors = ["#059669", "#D97706", "#EA580C", "#DC2626", "#7F1D1D"]
    banner_bg = colors.HexColor(grade_colors[min(grade_idx, 4)])

    finding_table_data = [
        [
            Paragraph(f"<font color='white'><b>PREDICTED ICDR GRADE: {grade_label.upper()}</b></font>", section_head),
            Paragraph(f"<font color='white'><b>Confidence: {conf_pct}% | Uncertainty: {unc.get('category')}</b></font>", section_head),
        ]
    ]
    t_finding = Table(finding_table_data, colWidths=[320, 220])
    t_finding.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_bg),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_finding)
    story.append(Spacer(1, 8))

    # 5. Concept Evidence Table
    story.append(Paragraph("<b>CLINICAL LESION / CONCEPT EVIDENCE (CONCEPT BOTTLENECK)</b>", section_head))
    story.append(Spacer(1, 3))

    concept_headers = ["Clinical Concept", "Category", "Confidence", "Severity Extent", "Validation Status"]
    concept_rows = [[
        Paragraph(f"<b>{h}</b>", meta_style) for h in concept_headers
    ]]

    for obs in report_dict.get("concept_evidence_observations", [])[:8]:
        p_val = obs.get("presence_probability", 0.0)
        sev_val = obs.get("severity_score", 0.0)
        status = obs.get("evidence_status", "")
        concept_rows.append([
            Paragraph(obs.get("concept_name", ""), body_text),
            Paragraph(obs.get("category", ""), meta_style),
            Paragraph(f"{p_val:.1%}", bold_body if p_val >= 0.40 else body_text),
            Paragraph(f"{sev_val:.2f}", body_text),
            Paragraph(obs.get("validation_tier", "EXPERIMENTAL"), meta_style),
        ])

    t_concepts = Table(concept_rows, colWidths=[150, 110, 85, 85, 110])
    t_concepts.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_concepts)
    story.append(Spacer(1, 8))

    # 6. Quadrant & Counterfactual Summary
    quad = report_dict.get("anatomical_quadrant_burden", {})
    cf = report_dict.get("counterfactual_analysis", {})
    plan = report_dict.get("clinical_action_plan", {})
    nlg = report_dict.get("nlg_narrative", {})

    cf_down = cf.get("downward_transition") or {}
    cf_text = cf_down.get("summary", "Decision boundary well separated from adjacent grades.")

    analysis_table_data = [
        [
            Paragraph("<b>Lesion Quadrant Distribution:</b>", bold_body),
            Paragraph(nlg.get("quadrant_summary", ""), body_text),
        ],
        [
            Paragraph("<b>Macular CSME Threat:</b>", bold_body),
            Paragraph("High Perifoveal Exudative Risk" if quad.get("macular_threat") else "No definite foveal exudates", body_text),
        ],
        [
            Paragraph("<b>Counterfactual Sensitivity:</b>", bold_body),
            Paragraph(cf_text, body_text),
        ],
        [
            Paragraph("<b>Recommended Action:</b>", bold_body),
            Paragraph(f"<b>{plan.get('urgency_level', 'Consult')}:</b> {plan.get('recommended_action', '')}", bold_body),
        ]
    ]
    t_analysis = Table(analysis_table_data, colWidths=[140, 400])
    t_analysis.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_analysis)
    story.append(Spacer(1, 10))

    # 7. Safety Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#EF4444"), spaceAfter=5))
    story.append(Paragraph(
        "<b>CLINICAL DISCLAIMER & REGULATORY NOTICE:</b> Sanjeevani-AI is an investigational screening prototype "
        "and not an autonomous diagnostic medical device. It does not replace an in-person ophthalmic dilated slit-lamp examination. "
        "All borderline, ungradable, referable (Grade >= 2), or high-uncertainty findings mandate confirmation by a certified ophthalmologist.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
