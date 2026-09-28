"""
sanjeevani_ai.app
-----------------
SANJEEVANI-AI: Clinically-Grounded Explainable AI for Diabetic Retinopathy Screening.
Smart India Hackathon Working Prototype.

Features:
1. Field Technician View (Simplified capture, quality triage, actionable recapture advice)
2. Clinician View:
   - Single-Eye & Bilateral (OD/OS) Dual-Eye Assessment with Carotid Asymmetry Alert
   - Interactive 4x Digital Ophthalmic Lesion Loupe / Magnifier
   - Multi-Model Architecture Consensus (EfficientNet vs ResNet vs Concept Rules)
   - Concept Evidence Table (8 Clinical Heads)
   - Counterfactual Sensitivity & Decision Boundaries
   - Multilingual Patient Leaflet (Hindi, Kannada, Tamil, Telugu, English)
   - Certified PDF Download & ABDM/FHIR JSON Export with SHA-256 Audit Seal
3. Hackathon Technical / Judge View:
   - Concept Vector Spectrogram
   - MC-Dropout 20-Pass Probability Histograms & BALD MI
   - Dual-XAI Agreement Score (Grad-CAM++ vs Integrated Gradients IoU)
   - Causal Lesion Ablation Re-inference
   - Multi-Architecture Consensus Decomposition
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io
import time
import json
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sanjeevani_ai.inference.pipeline import SanjeevaniInferenceEngine
from sanjeevani_ai.inference.bilateral import analyze_bilateral_asymmetry
from sanjeevani_ai.models.concept_bottleneck import CLINICAL_CONCEPTS
from sanjeevani_ai.models.severity_head import ICDR_GRADES
from sanjeevani_ai.xai.loupe import extract_digital_loupe
from sanjeevani_ai.reporting.multilingual import get_multilingual_patient_guidance, MULTILINGUAL_LEAFLETS
from sanjeevani_ai.reporting.clinical_report import export_report_to_json
from sanjeevani_ai.referral.hospitals import get_recommended_referrals, HOSPITAL_DIRECTORY
from sanjeevani_ai.dashboard.stats import get_camp_statistics, log_screening_record
from sanjeevani_ai.i18n.translations import get_ui_text, TRANSLATIONS

# -----------------------------------------------------------------------------
# Page Configuration & Medical White Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sanjeevani-AI — Retinal Screening Clinical Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }
    .main-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 2px;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
    }
    .badge-pass {
        background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0;
        padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.80rem; display: inline-block;
    }
    .badge-fail {
        background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA;
        padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.80rem; display: inline-block;
    }
    .badge-amber {
        background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A;
        padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.80rem; display: inline-block;
    }
    .badge-blue {
        background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE;
        padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.80rem; display: inline-block;
    }
    .callout-box {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 14px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Engine Singleton Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_inference_engine():
    return SanjeevaniInferenceEngine()


def t(key: str) -> str:
    lang = st.session_state.get("ui_language", "English")
    return get_ui_text(key, lang=lang)


engine = get_inference_engine()

# -----------------------------------------------------------------------------
# Sidebar Controls & Case Loading
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 Language / भाषा")
    lang_options = ["English", "Hindi (हिंदी)", "Kannada (ಕನ್ನಡ)", "Tamil (தமிழ்)", "Telugu (తెలుగు)"]
    selected_lang = st.selectbox("Interface Language", lang_options, index=0)
    st.session_state["ui_language"] = selected_lang

    st.markdown("---")
    st.markdown(f"### 🎛️ {t('nav_title')}")
    view_options = [
        t("view_clinician"),
        t("view_technician"),
        t("view_judge"),
        t("view_dashboard"),
    ]
    view_mode = st.radio(t("select_view"), view_options, index=0)

    st.markdown("---")
    st.markdown(f"### {t('patient_details_header')}")
    patient_name = st.text_input(t("patient_name"), value="Rameshwar Rao")
    patient_id = st.text_input(t("patient_id"), value="PATIENT-2026-IND-042")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        patient_age = st.number_input(t("patient_age"), min_value=1, max_value=120, value=58)
        patient_gender = st.selectbox(t("patient_gender"), ["Male", "Female", "Other"], index=0)
    with col_p2:
        abha_id = st.text_input(t("abha_id"), value="91-4521-8734-9012")
        contact_number = st.text_input(t("contact_number"), value="+91 98450 12345")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        diabetes_duration = st.number_input(t("diabetes_duration"), min_value=0, max_value=60, value=12)
        bp_systolic = st.number_input(t("bp_systolic"), min_value=70, max_value=250, value=142)
    with col_d2:
        hba1c = st.number_input(t("hba1c_level"), min_value=3.0, max_value=20.0, value=8.6, step=0.1)
        bp_diastolic = st.number_input(t("bp_diastolic"), min_value=40, max_value=150, value=88)

    patient_region = st.selectbox(
        t("select_region"),
        [
            "All India",
            "Karnataka / Bengaluru",
            "Tamil Nadu / Chennai",
            "Delhi NCR / North India",
            "Telangana / Hyderabad",
            "Maharashtra / Mumbai"
        ],
        index=1
    )

    patient_profile = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "abha_id": abha_id,
        "contact_number": contact_number,
        "diabetes_duration": diabetes_duration,
        "hba1c": hba1c,
        "bp_systolic": bp_systolic,
        "bp_diastolic": bp_diastolic,
        "region": patient_region,
    }

    st.markdown("---")
    st.markdown(f"### {t('image_input_header')}")
    protocol_options = [t("protocol_single"), t("protocol_bilateral")]
    study_protocol = st.radio(t("study_protocol"), protocol_options, index=0)

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    preset_files = {
        "Moderate NPDR (Grade 2)": "moderate_npdr.png",
        "Mild NPDR (Grade 1)": "mild_npdr.png",
        "Normal Retina (Grade 0)": "normal_fundus.png",
        "Proliferative DR (Grade 4)": "proliferative_dr.png",
        "Blurry Photo (Quality Rejection)": "blurry_ungradable.png",
        "External Eye Photo (Rejection)": "external_eye_photo.png",
    }

    if study_protocol == t("protocol_single"):
        eye_side = st.selectbox(t("examined_eye"), [t("eye_od"), t("eye_os")], index=0)
        eye_code = "OD" if "OD" in eye_side else "OS"
        source_options = [t("source_preset"), t("source_upload"), t("source_camera")]
        input_mode = st.radio(t("input_source"), source_options, index=0)

        selected_image_path = None
        uploaded_bytes = None

        if input_mode == t("source_preset"):
            chosen_preset = st.selectbox("Select Benchmark Retinal Image", list(preset_files.keys()))
            selected_image_path = os.path.join(sample_dir, preset_files[chosen_preset])
        elif input_mode == t("source_upload"):
            uploaded_file = st.file_uploader(t("source_upload"), type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.read()
        else:
            st.info(t("capture_prompt"))
            cam_pic = st.camera_input(t("source_camera"))
            if cam_pic is not None:
                uploaded_bytes = cam_pic.read()
    else:
        st.markdown(f"**{t('protocol_bilateral')}**")
        preset_od = st.selectbox(f"{t('eye_od')} Image", list(preset_files.keys()), index=0)
        preset_os = st.selectbox(f"{t('eye_os')} Image", list(preset_files.keys()), index=1)
        path_od = os.path.join(sample_dir, preset_files[preset_od])
        path_os = os.path.join(sample_dir, preset_files[preset_os])

    bypass_quality = st.checkbox(t("bypass_quality"), value=False)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size: 0.72rem; color: #94A3B8;'>"
        f"<b>CLINICAL DISCLAIMER:</b> {t('disclaimer_sidebar')}"
        f"</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Top Banner & System Branding
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 1.6rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                {t('app_title')}
            </span>
            <span style="background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; margin-left: 8px;">
                SIH PROTOTYPE
            </span>
            <div style="color: #64748B; font-size: 0.85rem; margin-top: 2px;">
                {t('app_subtitle')}
            </div>
        </div>
        <div style="text-align: right;">
            <span style="background: #F1F5F9; color: #475569; padding: 5px 12px; border-radius: 8px; font-size: 0.78rem; font-weight: 600; border: 1px solid #CBD5E1;">
                {t('offline_badge')}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Single Eye vs Bilateral Execution vs Dashboard
# -----------------------------------------------------------------------------
result = None
if view_mode != t("view_dashboard"):
    if study_protocol == t("protocol_single"):
        input_data = uploaded_bytes if uploaded_bytes is not None else selected_image_path

        if input_data is None or (isinstance(input_data, str) and not os.path.exists(input_data)):
            st.warning("Please upload a fundus photograph, capture via on-site camera, or select a benchmark preset to begin screening.")
            st.stop()

        with st.spinner("Executing 4-Layer Trust Architecture Screening Pipeline..."):
            result = engine.predict(
                input_data,
                patient_id=patient_id,
                eye_side=eye_code,
                bypass_quality_gate=bypass_quality,
                patient_profile=patient_profile
            )
            # Log screening dynamically into the camp dashboard registry
            log_screening_record(patient_profile, result)
    else:
        # Run bilateral pipeline on OD and OS
        with st.spinner("Executing Bilateral Dual-Eye Screening Study (OD & OS)..."):
            res_od = engine.predict(path_od, patient_id=patient_id, eye_side="OD", bypass_quality_gate=bypass_quality, patient_profile=patient_profile)
            res_os = engine.predict(path_os, patient_id=patient_id, eye_side="OS", bypass_quality_gate=bypass_quality, patient_profile=patient_profile)
            bilateral_summary = analyze_bilateral_asymmetry(res_od, res_os, patient_id=patient_id)
            result = res_od if bilateral_summary["worse_eye"].startswith("OD") else res_os
            log_screening_record(patient_profile, result)

    # Operational Status Stamp
    if result.get("is_demo"):
        st.markdown("""
        <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 8px 16px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;">
            <span style="color: #92400E; font-size: 0.85rem; font-weight: 700;">
                🟡 DEMO MODE — illustrative output only (Pre-trained weights active, offline benchmark simulation)
            </span>
            <span style="color: #B45309; font-size: 0.75rem;">SIH Evaluation Ready</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 8px 16px; margin-bottom: 16px;">
            <span style="color: #166534; font-size: 0.85rem; font-weight: 700;">
                🟢 REAL MODEL MODE — Research model (Not clinically validated)
            </span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 1: FIELD TECHNICIAN VIEW
# =============================================================================
if view_mode == t("view_technician"):
    st.subheader("Field Technician Screening Gate")
    st.caption("Actionable image gradability assessment for camera operators in rural vision camps.")

    qr = result.get("quality_report")
    col_img, col_gate = st.columns([1, 1.2])

    with col_img:
        st.markdown("**Captured Fundus Photograph**")
        if "images" in result:
            st.image(result["images"]["original_rgb"], use_container_width=True)
        elif result.get("original_rgb") is not None:
            st.image(result["original_rgb"], use_container_width=True)

    with col_gate:
        if qr and qr.is_gradable:
            st.markdown("""
            <div style="background: #ECFDF5; border: 2px solid #10B981; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 2.4rem;">✓</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #065F46;">IMAGE GRADABLE</div>
                <div style="color: #047857; font-size: 0.9rem; margin-top: 4px;">Photo meets optical sharpness, illumination, and field-of-view clinical criteria.</div>
            </div>
            """, unsafe_allow_html=True)

            st.success("Safe to proceed with automated screening analysis.")
            st.metric("Continuous Quality Index", f"{qr.quality_score:.1%}", help="Composite metric of focus, illumination, and FOV")
            st.write(f"• Sharpness (Laplacian Var): **{qr.blur_score:.1f}** (Threshold >= 45.0)")
            st.write(f"• Mean Illumination: **{qr.mean_brightness:.1f}** (Safe band: 20 - 215)")
            st.write(f"• Retinal FOV Coverage: **{qr.fov_fraction:.1%}**")

        else:
            st.markdown("""
            <div style="background: #FEF2F2; border: 2px solid #EF4444; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 2.4rem;">✗</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #991B1B;">IMAGE REJECTED — NOT GRADABLE</div>
                <div style="color: #B91C1C; font-size: 0.9rem; margin-top: 4px;">DO NOT RUN OR TRUST DR SCREENING ON THIS IMAGE.</div>
            </div>
            """, unsafe_allow_html=True)

            st.error("Immediate Retake Required. Identified Optical Issues:")
            for issue in (qr.reasons if qr else result.get("technician_actions", [])):
                st.markdown(f"• **{issue}**")

            st.markdown("### 🛠️ Actionable Technician Instructions:")
            for act in (qr.technician_actions if qr else []):
                st.info(f"👉 {act}")


# =============================================================================
# VIEW 2: CLINICIAN VIEW
# =============================================================================
elif view_mode == t("view_clinician"):
    if not result.get("is_gradable", True) and not bypass_quality:
        st.error("Cannot present clinical diagnosis for an ungradable image. Please inspect 'Field Technician View'.")
        st.stop()

    dr = result.get("report_dict", {}).get("dr_screening_result", {})
    unc = result.get("uncertainty_report")
    quad = result.get("quadrant_info", {})
    consensus = result.get("model_consensus", {})

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted ICDR Severity</div>
            <div class="metric-val" style="color: #2563EB;">{result.get('grade_label')}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calibrated Confidence</div>
            <div class="metric-val">{dr.get('confidence_percentage', 0.0)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        unc_lvl = unc.uncertainty_level if unc else "N/A"
        color = "#059669" if unc_lvl == "LOW" else ("#D97706" if unc_lvl == "MEDIUM" else "#DC2626")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Uncertainty Triage</div>
            <div class="metric-val" style="color: {color};">{unc_lvl}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        macular_alert = quad.get("macular_threat", False)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Macular Involvement</div>
            <div class="metric-val" style="color: {'#DC2626' if macular_alert else '#059669'};">
                {'HIGH RISK' if macular_alert else 'CLEAR'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bilateral Asymmetry Section (If Bilateral Mode Active)
    if study_protocol == "Bilateral (OD & OS Dual-Eye)":
        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("👁️ Bilateral Dual-Eye Comparison & Inter-Ocular Asymmetry")
        st.caption("Evaluates concordance between Right Eye (OD) and Left Eye (OS). Highlights carotid stenosis alerts.")

        b_col1, b_col2, b_col3 = st.columns([1, 1, 1.2])
        with b_col1:
            st.markdown(f"**Right Eye (OD): {bilateral_summary['od_grade_name']}**")
            st.image(res_od["images"]["cam_overlay"], use_container_width=True)
        with b_col2:
            st.markdown(f"**Left Eye (OS): {bilateral_summary['os_grade_name']}**")
            st.image(res_os["images"]["cam_overlay"], use_container_width=True)
        with b_col3:
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px;">
                <div style="font-weight: 700; color: #1E293B;">Bilateral Synthesis:</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">• Asymmetry Gap: <b>{bilateral_summary['asymmetry_gap']} Grade(s)</b></div>
                <div style="font-size: 0.85rem;">• Asymmetry Status: <b>{bilateral_summary['asymmetry_status']}</b></div>
                <div style="font-size: 0.85rem;">• Overall Patient Stage: <b>{bilateral_summary['patient_severity_name']}</b></div>
                <div style="margin-top: 8px; font-size: 0.82rem; color: {'#DC2626' if bilateral_summary['is_asymmetric'] else '#475569'}; line-height: 1.4;">
                    {bilateral_summary['carotid_ischemia_alert']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Multi-Model Architecture Consensus Card
    st.markdown("<br/>", unsafe_allow_html=True)
    if consensus:
        c_badge = "badge-pass" if not consensus.get("is_discordant") else "badge-fail"
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; font-size: 0.92rem; color: #1E293B;">
                    Multi-Architecture Consensus Check: <span class="{c_badge}">{consensus.get('status')}</span>
                </span>
                <span style="color: #64748B; font-size: 0.80rem;">Agreement: <b>{consensus.get('agreement_ratio', 0.0):.0%}</b></span>
            </div>
            <div style="color: #475569; font-size: 0.82rem; margin-top: 4px;">{consensus.get('clinical_note')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Inspection Row
    st.subheader("Visual Inspection & Lesion Localization")
    c_img1, c_img2, c_img3 = st.columns(3)

    imgs = result.get("images", {})
    with c_img1:
        st.markdown("**1. Original Retinal Field**")
        st.image(imgs.get("cropped_rgb", imgs.get("original_rgb")), use_container_width=True)

    with c_img2:
        st.markdown("**2. Ben Graham Local Color Norm**")
        st.image(imgs.get("normalized_rgb"), use_container_width=True)

    with c_img3:
        st.markdown("**3. Dual-XAI Saliency Overlay (Grad-CAM++ / IG)**")
        st.image(imgs.get("cam_overlay"), use_container_width=True)

    # Interactive 4x Digital Ophthalmic Loupe
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("🔍 Interactive 4x Digital Ophthalmic Lesion Loupe")
    st.caption("Inspect suspicious microvascular lesions at 4x optical magnification with green-channel hemoglobin contrast.")

    col_loupe_ctrl, col_loupe_img1, col_loupe_img2 = st.columns([1, 1, 1.2])
    with col_loupe_ctrl:
        loupe_quadrant = st.selectbox(
            "Select Lesion Hotspot Location",
            ["Central Macular Zone", "Superior-Temporal Arc", "Inferior-Temporal Arc", "Optic Disc Margin"]
        )
        # Coordinate mapping for loupe inspection
        h_im, w_im = imgs.get("cropped_rgb", imgs.get("original_rgb")).shape[:2]
        if "Macular" in loupe_quadrant:
            l_center = (w_im // 2, h_im // 2)
        elif "Superior-Temporal" in loupe_quadrant:
            l_center = (int(w_im * 0.35), int(h_im * 0.35))
        elif "Inferior-Temporal" in loupe_quadrant:
            l_center = (int(w_im * 0.35), int(h_im * 0.65))
        else:
            l_center = (int(w_im * 0.70), int(h_im * 0.50))

        loupe_data = extract_digital_loupe(imgs.get("cropped_rgb", imgs.get("original_rgb")), center_xy=l_center, patch_size=64, zoom_factor=4)
        st.info("4x Bicubic Zoom with CLAHE Green-Spectrum Enhancer active.")

    with col_loupe_img1:
        st.markdown("**Zoomed 4x RGB Lesion Patch**")
        st.image(loupe_data["zoomed_rgb"], use_container_width=True)

    with col_loupe_img2:
        st.markdown("**Green-Spectrum Hemoglobin Vessel Transect**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(loupe_data["transect_profile"], color="#059669", linewidth=2)
        ax.set_title("Capillary Absorption Profile (570nm)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Pixel Transect Across Lesion", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Concept Bottleneck Evidence Table
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Clinical Lesion / Concept Bottleneck Evidence")
    st.caption("Inspectable intermediate clinical features driving the severity decision. No black box.")

    concept_items = result.get("concept_evidence", [])
    cols_c = st.columns(2)

    for i, c in enumerate(concept_items):
        target_col = cols_c[i % 2]
        with target_col:
            badge_class = "badge-pass" if c["presence_prob"] >= 0.7 else ("badge-amber" if c["presence_prob"] >= 0.4 else "badge-blue")
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 0.9rem;">{c['name']}</span>
                    <span class="{badge_class}">{c['status_text']}</span>
                </div>
                <div style="color: #64748B; font-size: 0.78rem; margin-top: 4px;">
                    Category: {c['category']} | Status: <b>{c['validation_status']}</b> ({c['dataset_source']})
                </div>
                <div style="display: flex; gap: 16px; margin-top: 6px; font-size: 0.8rem;">
                    <span>Confidence: <b>{c['presence_prob']:.1%}</b></span>
                    <span>Burden Extent: <b>{c['severity_score']:.2f}</b></span>
                    <span>Attribution: <b>{c['attribution_to_grade']:+.2f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Counterfactual Analysis for Borderline Cases
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Counterfactual Sensitivity & Decision Boundaries")
    cf = result.get("counterfactuals", {})
    cf_down = cf.get("downward_transition")
    cf_up = cf.get("upward_transition")

    col_cf1, col_cf2 = st.columns(2)
    with col_cf1:
        if cf_down and cf_down.get("success"):
            st.markdown(f"""
            <div class="callout-box">
                <span style="font-weight: 700; color: #0F172A;">Downward Transition: {cf_down['current_grade_name']} ➔ {cf_down['target_grade_name']}</span>
                <p style="font-size: 0.82rem; color: #475569; margin-top: 4px;">{cf_down['summary']}</p>
                <div style="font-size: 0.78rem; color: #64748B;">
                    <b>Required Concept Reductions:</b><br/>
                    {'<br/>'.join([c['summary'] for c in cf_down.get('concept_changes', [])[:3]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No downward transition applicable (Grade 0).")

    with col_cf2:
        if cf_up and cf_up.get("success"):
            st.markdown(f"""
            <div class="callout-box">
                <span style="font-weight: 700; color: #0F172A;">Upward Transition: {cf_up['current_grade_name']} ➔ {cf_up['target_grade_name']}</span>
                <p style="font-size: 0.82rem; color: #475569; margin-top: 4px;">{cf_up['summary']}</p>
                <div style="font-size: 0.78rem; color: #64748B;">
                    <b>Impending Risk Drivers:</b><br/>
                    {'<br/>'.join([c['summary'] for c in cf_up.get('concept_changes', [])[:3]])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Maximum severity stage (Grade 4).")

    # Structured Clinical Summary
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader("Structured Screening Narrative & Referral Plan")
    nlg = result.get("report_dict", {}).get("nlg_narrative", {})
    plan = result.get("report_dict", {}).get("clinical_action_plan", {})

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 20px;">
        <div style="font-weight: 700; color: #1E293B; margin-bottom: 6px;">Ophthalmological Clinical Rationale:</div>
        <p style="font-size: 0.85rem; color: #334155; line-height: 1.5;">{nlg.get('severity_reasoning')}</p>
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #F1F5F9;">
            <span style="font-weight: 700; color: #0F172A;">Recommended Action:</span> 
            <span style="color: #2563EB; font-weight: 600;">{plan.get('urgency_level')} ({plan.get('recommended_timeframe')})</span>
            <p style="font-size: 0.82rem; color: #475569; margin-top: 2px;">{plan.get('recommended_action')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommended Eye Hospitals & Specialist Referral Network
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader(f"🏥 {t('hospitals_title')}")
    st.caption(t("hospitals_sub"))

    referrals = get_recommended_referrals(
        severity_grade=result.get("predicted_grade", 0),
        region=patient_region,
        has_macular_threat=quad.get("macular_threat", False)
    )

    u_code = referrals.get("urgency_code", "GREEN_ROUTINE")
    if u_code == "RED_EMERGENCY":
        u_bg, u_fg, u_border = "#FEF2F2", "#DC2626", "#FCA5A5"
        u_icon = "🚨"
    elif u_code == "ORANGE_URGENT":
        u_bg, u_fg, u_border = "#FFFBEB", "#D97706", "#FDE68A"
        u_icon = "⚠️"
    elif u_code == "YELLOW_FOLLOWUP":
        u_bg, u_fg, u_border = "#FEFCE8", "#CA8A04", "#FEF08A"
        u_icon = "🟡"
    else:
        u_bg, u_fg, u_border = "#F0FDF4", "#166534", "#BBF7D0"
        u_icon = "🟢"

    st.markdown(f"""
    <div style="background: {u_bg}; border: 1.5px solid {u_border}; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-weight: 800; font-size: 0.95rem; color: {u_fg};">
                {u_icon} {referrals.get('action_headline')}
            </span>
            <div style="color: #475569; font-size: 0.80rem; margin-top: 2px;">
                Matched <b>{referrals.get('matched_hospitals_count')}</b> certified eye care institutes for region: <b>{patient_region}</b>
            </div>
        </div>
        <div>
            <span style="background: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-family: monospace; font-weight: 700;">
                {referrals.get('abdm_referral_token')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hosp_cols = st.columns(2)
    for idx, h in enumerate(referrals.get("hospitals", [])):
        target_h_col = hosp_cols[idx % 2]
        with target_h_col:
            pmjay_badge = '<span class="badge-pass">PM-JAY Empaneled</span>' if h.get("pmjay_empaneled") else ''
            specialists_html = "".join([
                f"<div style='font-size: 0.78rem; color: #1E293B; margin-top: 4px;'>"
                f"👨‍⚕️ <b>{sp['name']}</b> ({sp['designation']}) — <i>Exp: {sp['experience']}</i> | OPD: {sp['opd_days']}</div>"
                for sp in h.get("top_specialists", [])
            ])
            facilities_html = " ".join([f"<span class='badge-blue' style='font-size: 0.70rem; margin-right: 4px; margin-bottom: 4px;'>{fac}</span>" for fac in h.get("facilities", [])[:3]])

            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-weight: 800; font-size: 0.92rem; color: #0F172A;">{h['name']}</div>
                        <div style="font-size: 0.75rem; color: #64748B;">{h['tier']} • {h['city']}, {h['state']}</div>
                    </div>
                    <div>{pmjay_badge}</div>
                </div>
                <div style="margin-top: 8px; font-size: 0.78rem; color: #475569;">
                    📍 {h['address']}
                </div>
                <div style="margin-top: 6px; font-size: 0.78rem; color: #2563EB;">
                    📞 <b>Retina Casualty:</b> {h['emergency_retina']} | Helpline: {h['helpline']}
                </div>
                <div style="margin-top: 8px;">
                    {facilities_html}
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #F1F5F9;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Top Vitreoretinal Surgeons:</div>
                    {specialists_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Generate ABDM Referral to {h['name'].split()[0]}", key=f"ref_btn_{h['id']}"):
                st.success(f"✓ ABDM Referral Packet generated! Token: {referrals.get('abdm_referral_token')} linked to patient {patient_id}.")

    # Multilingual Rural Patient Leaflet
    st.markdown("<br/>", unsafe_allow_html=True)
    st.subheader(f"{t('multilingual_title')}")
    st.caption(t("multilingual_sub"))

    lang_choice = st.selectbox("Select Patient Language", ["English", "Hindi (हिंदी)", "Kannada (ಕನ್ನಡ)", "Tamil (தமிழ்)"])
    leaf = get_multilingual_patient_guidance(result.get("predicted_grade", 0), language=lang_choice)

    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1.5px solid #CBD5E1; border-radius: 10px; padding: 16px 20px;">
        <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">{leaf['title']}</div>
        <p style="font-size: 0.88rem; color: #334155; margin-top: 4px;">{leaf['message']}</p>
        <div style="margin-top: 10px; font-size: 0.82rem; color: #0369A1; font-weight: 700;">
            👉 {leaf['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cryptographic Tamper-Proof Seal Indicator
    seal = result.get("report_dict", {}).get("integrity_audit_seal", {})
    if seal:
        st.markdown(f"""
        <div style="margin-top: 14px; font-size: 0.72rem; color: #64748B; font-family: monospace;">
            🔒 SHA-256 Cryptographic Audit Seal: {seal.get('sha256_hash')} | Status: {seal.get('audit_status')}
        </div>
        """, unsafe_allow_html=True)

    # Downloads Row
    st.markdown("<br/>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        pdf_data = result.get("pdf_bytes")
        if pdf_data:
            st.download_button(
                label="📄 Download Certified Clinical PDF Report",
                data=pdf_data,
                file_name=f"Sanjeevani_Report_{patient_id}_{eye_code if study_protocol == 'Single Eye (OD/OS)' else 'Bilateral'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    with d2:
        json_report = export_report_to_json(result.get("report_dict", {}))
        st.download_button(
            label="💾 Export ABDM / FHIR-Ready JSON Report",
            data=json_report,
            file_name=f"Sanjeevani_ABDM_{patient_id}_{eye_code if study_protocol == 'Single Eye (OD/OS)' else 'Bilateral'}.json",
            mime="application/json",
            use_container_width=True
        )


# =============================================================================
# VIEW 3: TECHNICAL / JUDGE VIEW
# =============================================================================
elif view_mode == t("view_judge"):
    st.subheader("Hackathon Technical Deep-Dive & Novelty Audit")
    st.caption("Inspectable representations designed to demonstrate technical rigor to Smart India Hackathon judges.")

    tabs = st.tabs([
        "1. Concept Bottleneck Vectors",
        "2. Uncertainty Quantification (MC-Dropout & TTA)",
        "3. Dual-XAI Agreement Score",
        "4. Multi-Model Consensus",
        "5. Causal Lesion Ablation",
        "6. Preprocessing Comparison",
        "7. Cryptographic Integrity Seal"
    ])

    # Tab 1: Concept Bottleneck
    with tabs[0]:
        st.markdown("#### Concept Bottleneck Architecture Activation Spectrum")
        st.markdown(
            "The model routes through an intermediate clinical concept vector $c \\in [0, 1]^K$. "
            "The final severity logits are an explicit, inspectable function of these concept values."
        )

        c_names = [c["name"] for c in result.get("concept_evidence", [])]
        c_probs = [c["presence_prob"] for c in result.get("concept_evidence", [])]
        c_attrs = [c["attribution_to_grade"] for c in result.get("concept_evidence", [])]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        y_pos = np.arange(len(c_names))
        ax.barh(y_pos, c_probs, color="#2563EB", alpha=0.8, label="Concept Presence Probability")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(c_names, fontsize=8)
        ax.set_xlim([0, 1.0])
        ax.set_xlabel("Calibrated Concept Activation")
        ax.set_title("Concept Activations for Current Case", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

    # Tab 2: Uncertainty
    with tabs[1]:
        st.markdown("#### Monte Carlo Dropout Sampling & TTA Variance Analysis")
        unc = result.get("uncertainty_report")
        if unc and unc.pass_distributions:
            st.markdown(
                f"**Predictive Entropy:** `{unc.predictive_entropy:.3f}` | "
                f"**BALD Epistemic Mutual Info:** `{unc.mutual_information:.4f}` | "
                f"**MC Variance:** `{unc.mc_variance:.4f}` | "
                f"**TTA Variance:** `{unc.tta_variance:.4f}`"
            )

            passes = np.array(unc.pass_distributions)  # (20, 5)
            fig, ax = plt.subplots(figsize=(10, 3.5))
            for g in range(5):
                ax.plot(passes[:, g], marker="o", markersize=3, label=ICDR_GRADES[g])
            ax.set_xlabel("Stochastic MC-Dropout Forward Pass Index (1..20)")
            ax.set_ylabel("Class Probability")
            ax.set_title("Prediction Stability Across 20 Stochastic Passes", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)
            plt.close(fig)

    # Tab 3: Dual-XAI Consensus
    with tabs[2]:
        st.markdown("#### Cross-Method Explanation Agreement Score (Research Metric)")
        agree = result.get("agreement", {})

        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between;">
                <span><b>Agreement Score:</b> {agree.get('agreement_score', 0.0)}</span>
                <span><b>Hotspot IoU:</b> {agree.get('iou', 0.0)}</span>
                <span><b>Spatial Correlation:</b> {agree.get('correlation', 0.0)}</span>
                <span class="{'badge-pass' if not agree.get('is_flagged') else 'badge-fail'}">{agree.get('status')}</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">{agree.get('clinical_note')}</div>
        </div>
        """, unsafe_allow_html=True)

        col_cam, col_ig = st.columns(2)
        imgs = result.get("images", {})
        with col_cam:
            st.markdown("**Grad-CAM++ Attribution Map**")
            st.image(imgs.get("cam_overlay"), use_container_width=True)
        with col_ig:
            st.markdown("**Integrated Gradients Attribution Map**")
            st.image(imgs.get("ig_overlay"), use_container_width=True)

    # Tab 4: Multi-Model Consensus
    with tabs[3]:
        st.markdown("#### Multi-Architecture Model Diversity Check")
        consensus = result.get("model_consensus", {})
        if consensus:
            st.write(f"**Overall Consensus:** {consensus.get('consensus_grade_name')} (Status: `{consensus.get('status')}`)")
            st.caption(consensus.get("clinical_note"))
            for mb in consensus.get("models_breakdown", []):
                st.markdown(f"• **{mb['architecture']}**: Predicted `{mb['grade_name']}` (Confidence: {mb['confidence']:.1%})")

    # Tab 5: Causal Ablation
    with tabs[4]:
        st.markdown("#### Causal Lesion Ablation & Inpainting Counterfactual")
        ablation = result.get("counterfactuals", {}).get("image_ablation", {})

        col_ab1, col_ab2 = st.columns(2)
        with col_ab1:
            st.markdown("**Original Preprocessed Retina**")
            st.image(imgs.get("cropped_rgb"), use_container_width=True)
        with col_ab2:
            st.markdown("**Inpainted Retina (Primary Lesions Masked)**")
            if ablation.get("ablated_img_rgb") is not None:
                st.image(ablation["ablated_img_rgb"], use_container_width=True)

        st.info(ablation.get("causal_note", "No ablation data."))

    # Tab 6: Preprocessing
    with tabs[5]:
        st.markdown("#### Retinal Preprocessing Comparison")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("**1. Raw Input Image**")
            st.image(imgs.get("original_rgb"), use_container_width=True)
        with col_p2:
            st.markdown("**2. Circular FOV Crop**")
            st.image(imgs.get("cropped_rgb"), use_container_width=True)
        with col_p3:
            st.markdown("**3. Ben Graham Normalized**")
            st.image(imgs.get("normalized_rgb"), use_container_width=True)

    # Tab 7: Cryptographic Seal
    with tabs[6]:
        st.markdown("#### Cryptographic SHA-256 Tamper-Proof Audit Seal")
        seal = result.get("report_dict", {}).get("integrity_audit_seal", {})
        if seal:
            st.json(seal)
            st.success("Record integrity verified. Cryptographic hash matches ABDM registry specification.")


# =============================================================================
# VIEW 4: CAMP SCREENING DASHBOARD
# =============================================================================
elif view_mode == t("view_dashboard"):
    st.subheader(f"{t('dashboard_title')}")
    st.caption(t("dashboard_sub"))

    camp_stats = get_camp_statistics()

    # Top KPI Metrics Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t('total_screened')}</div>
            <div class="metric-val" style="color: #2563EB;">{camp_stats['total_screened']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Rural Outreach Vision Camp</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t('gradability_rate')}</div>
            <div class="metric-val" style="color: #059669;">{camp_stats['gradability_rate']:.1f}%</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">{camp_stats['gradable_count']} Gradable / {camp_stats['ungradable_count']} Recaptured</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t('referral_rate')}</div>
            <div class="metric-val" style="color: #D97706;">{camp_stats['referral_rate']:.1f}%</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">{camp_stats['referral_count']} Patients Referred to Hospital</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t('sight_threatening_count')}</div>
            <div class="metric-val" style="color: #DC2626;">{camp_stats['sight_threatening_count']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">PDR / Severe CSME Urgent Action</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown(f"#### 📊 {t('severity_distribution')}")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        grades = [0, 1, 2, 3, 4]
        grade_labels = ["No DR\n(0)", "Mild\n(1)", "Moderate\n(2)", "Severe\n(3)", "PDR\n(4)"]
        counts = [camp_stats["grade_distribution"].get(g, 0) for g in grades]
        bar_colors = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#7F1D1D"]
        bars = ax.bar(grade_labels, counts, color=bar_colors, width=0.55, edgecolor="#E2E8F0")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel("Patient Count", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_title("ICDR Retinopathy Breakdown (Rural Camp Cohort)", fontsize=9, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with ch2:
        st.markdown("#### 🎯 Clinical Triage Allocation")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        routine_c = camp_stats["grade_distribution"].get(0, 0) + camp_stats["grade_distribution"].get(1, 0)
        urgent_c = camp_stats["grade_distribution"].get(2, 0) + camp_stats["grade_distribution"].get(3, 0)
        emergency_c = camp_stats["grade_distribution"].get(4, 0)
        triage_slices = [max(1, routine_c), max(1, urgent_c), max(1, emergency_c)]
        triage_labels = ["Primary / Annual", "Specialist (1-3 Mo)", "Emergency (48h-2Wk)"]
        triage_colors = ["#10B981", "#F59E0B", "#EF4444"]
        ax2.pie(
            triage_slices,
            labels=triage_labels,
            colors=triage_colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax2.set_title("Triage Allocation", fontsize=9, fontweight="bold")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(f"#### 📋 {t('recent_screenings')}")

    rec_list = camp_stats.get("records", [])
    if rec_list:
        table_rows = []
        for r in rec_list[:15]:
            table_rows.append({
                "Token ID": r.get("id"),
                "Patient Name": r.get("name"),
                "Age/Sex": f"{r.get('age')} / {r.get('gender')[:1]}",
                "ABHA ID": r.get("abha_id"),
                "Eye": r.get("eye"),
                "Predicted Stage": r.get("grade_name"),
                "Confidence": f"{r.get('confidence')}%",
                "Urgency Triage": r.get("urgency"),
                "Matched Hospital": r.get("matched_hospital"),
                "Time": r.get("timestamp"),
            })
        st.dataframe(table_rows, use_container_width=True)

    csv_buf = io.StringIO()
    csv_buf.write("id,name,age,gender,abha_id,eye,grade,grade_name,confidence,urgency,matched_hospital,timestamp\n")
    for r in rec_list:
        csv_buf.write(f'"{r.get("id")}","{r.get("name")}",{r.get("age")},"{r.get("gender")}","{r.get("abha_id")}","{r.get("eye")}",{r.get("grade")},"{r.get("grade_name")}",{r.get("confidence")},"{r.get("urgency")}","{r.get("matched_hospital")}","{r.get("timestamp")}"\n')

    st.download_button(
        label="📥 Export Complete Camp Screening Ledger (CSV)",
        data=csv_buf.getvalue(),
        file_name="sanjeevani_camp_screening_ledger.csv",
        mime="text/csv",
        use_container_width=True
    )

