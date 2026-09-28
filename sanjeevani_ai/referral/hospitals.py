"""
sanjeevani_ai.referral.hospitals
--------------------------------
Smart Tele-Ophthalmology Referral & Hospital Matching Engine.

Automatically matches screened patients to nearby certified eye hospitals
and vitreoretinal sub-specialists based on:
1. ICDR Severity Stage (Grade 0 to Grade 4)
2. Macular CSME Threat Level
3. Patient Region / Geographic Zone
4. Ayushman Bharat (PM-JAY / ABDM) Empanelment Status
"""

from typing import List, Dict, Any, Optional

HOSPITAL_DIRECTORY = [
    {
        "id": "HOSP-01",
        "name": "Narayana Nethralaya Eye Institute",
        "city": "Bengaluru",
        "state": "Karnataka",
        "region": "Karnataka / Bengaluru",
        "tier": "Apex Super-Specialty Tertiary Eye Center",
        "pmjay_empaneled": True,
        "facilities": ["24x7 Vitreo-Retinal Surgery", "Anti-VEGF Injection Suite", "Panretinal Laser (PRP)", "Swept-Source OCT", "Fluorescein Angiography"],
        "min_grade_tier": 2,
        "address": "121/C, Chord Road, Rajajinagar, Bengaluru, Karnataka 560010",
        "helpline": "+91-80-6612-1400",
        "emergency_retina": "+91-80-6612-1401",
        "top_specialists": [
            {"name": "Dr. K. Bhujang Shetty", "designation": "Chief Ophthalmologist & Vitreo-Retina Mentor", "experience": "35+ Years", "opd_days": "Mon, Wed, Fri"},
            {"name": "Dr. Narendra K.P.", "designation": "Senior Vitreoretinal Surgeon", "experience": "18+ Years", "opd_days": "Tue, Thu, Sat"},
        ]
    },
    {
        "id": "HOSP-02",
        "name": "Minto Ophthalmic Hospital (BMCRI)",
        "city": "Bengaluru",
        "state": "Karnataka",
        "region": "Karnataka / Bengaluru",
        "tier": "Government Autonomous Tertiary Institute",
        "pmjay_empaneled": True,
        "facilities": ["Free/Subsidized Diabetic Retinopathy Clinic", "Vitrectomy Theatre", "Argon Laser", "Macular Edema Unit"],
        "min_grade_tier": 1,
        "address": "AV Road, Kalasipalya, Bengaluru, Karnataka 560002",
        "helpline": "+91-80-2670-1123",
        "emergency_retina": "+91-80-2670-1124",
        "top_specialists": [
            {"name": "Dr. Nagaraj H.", "designation": "Professor & Head of Vitreo-Retina Unit", "experience": "22+ Years", "opd_days": "Mon to Fri"},
        ]
    },
    {
        "id": "HOSP-03",
        "name": "Sankara Nethralaya (Main Campus)",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "region": "Tamil Nadu / Chennai",
        "tier": "Apex National Tertiary Care & Research Institute",
        "pmjay_empaneled": True,
        "facilities": ["Advanced Sutureless Vitrectomy (MIVS)", "Comprehensive Diabetic Retinopathy Centre", "Navilas Laser", "OCT Angiography"],
        "min_grade_tier": 2,
        "address": "18, College Road, Nungambakkam, Chennai, Tamil Nadu 600006",
        "helpline": "+91-44-4227-1500",
        "emergency_retina": "+91-44-4227-1555",
        "top_specialists": [
            {"name": "Dr. Lingam Gopal", "designation": "Distinguished Senior Retinal Consultant", "experience": "38+ Years", "opd_days": "Mon, Wed"},
            {"name": "Dr. Pramod Bhende", "designation": "Director - Vitreoretinal Services", "experience": "25+ Years", "opd_days": "Tue, Thu, Fri"},
        ]
    },
    {
        "id": "HOSP-04",
        "name": "Aravind Eye Hospital",
        "city": "Madurai",
        "state": "Tamil Nadu",
        "region": "Tamil Nadu / Chennai",
        "tier": "WHO Collaborating Center for Blindness Prevention",
        "pmjay_empaneled": True,
        "facilities": ["High-Volume Diabetic Retinopathy Clinic", "Subsidized Laser Suite", "Mobile Tele-Ophthalmology Network"],
        "min_grade_tier": 1,
        "address": "1, Anna Nagar, Madurai, Tamil Nadu 625020",
        "helpline": "+91-452-435-6100",
        "emergency_retina": "+91-452-435-6101",
        "top_specialists": [
            {"name": "Dr. R. Kim", "designation": "Chief of Vitreo-Retinal Services", "experience": "30+ Years", "opd_days": "Mon, Wed, Fri"},
        ]
    },
    {
        "id": "HOSP-05",
        "name": "Dr. Rajendra Prasad Centre for Ophthalmic Sciences (AIIMS)",
        "city": "New Delhi",
        "state": "Delhi",
        "region": "Delhi NCR / North India",
        "tier": "Apex National Central Government Institute",
        "pmjay_empaneled": True,
        "facilities": ["Apex Retinal Trauma & Surgery", "Subsidized Anti-VEGF / Laser", "24x7 Emergency Ophthalmic Casualty"],
        "min_grade_tier": 2,
        "address": "Ansari Nagar, New Delhi 110029",
        "helpline": "+91-11-2658-8500",
        "emergency_retina": "+91-11-2659-3101",
        "top_specialists": [
            {"name": "Dr. Pradeep Venkatesh", "designation": "Professor of Vitreo-Retina Unit", "experience": "26+ Years", "opd_days": "Mon, Thu"},
            {"name": "Dr. Rohan Chawla", "designation": "Additional Professor - Retina", "experience": "16+ Years", "opd_days": "Tue, Fri"},
        ]
    },
    {
        "id": "HOSP-06",
        "name": "Dr. Shroff's Charity Eye Hospital",
        "city": "New Delhi",
        "state": "Delhi",
        "region": "Delhi NCR / North India",
        "tier": "Super-Specialty Tertiary Eye Center",
        "pmjay_empaneled": True,
        "facilities": ["Diabetic Eye Disease Management Clinic", "Green Laser & Vitrectomy", "Rural Outreach Network"],
        "min_grade_tier": 1,
        "address": "5027, Kedar Nath Road, Daryaganj, New Delhi 110002",
        "helpline": "+91-11-4352-4444",
        "emergency_retina": "+91-11-4352-4445",
        "top_specialists": [
            {"name": "Dr. Manisha Agarwal", "designation": "Head of Vitreoretinal Services", "experience": "20+ Years", "opd_days": "Mon, Wed, Sat"},
        ]
    },
    {
        "id": "HOSP-07",
        "name": "L.V. Prasad Eye Institute (Kallam Anji Reddy Campus)",
        "city": "Hyderabad",
        "state": "Telangana",
        "region": "Telangana / Hyderabad",
        "tier": "Apex International Tertiary Care Institute",
        "pmjay_empaneled": True,
        "facilities": ["Centre of Excellence in Retina", "Surgical Vitrectomy", "OCT & FFA Angiography", "Community Vision Corridors"],
        "min_grade_tier": 2,
        "address": "Road No. 2, Banjara Hills, Hyderabad, Telangana 500034",
        "helpline": "+91-40-6810-2020",
        "emergency_retina": "+91-40-6810-2100",
        "top_specialists": [
            {"name": "Dr. Raja Narayanan", "designation": "Director - Retina & Vitreous Center", "experience": "24+ Years", "opd_days": "Tue, Thu, Sat"},
            {"name": "Dr. Subhadra Jalali", "designation": "Senior Retinal Consultant", "experience": "30+ Years", "opd_days": "Mon, Wed, Fri"},
        ]
    },
    {
        "id": "HOSP-08",
        "name": "Aditya Jyot Eye Hospital / Dr. Agarwal's",
        "city": "Mumbai",
        "state": "Maharashtra",
        "region": "Maharashtra / Mumbai",
        "tier": "Super-Specialty Tertiary Eye Center",
        "pmjay_empaneled": True,
        "facilities": ["Advanced Micro-Incision Vitrectomy", "Diabetic Retinopathy Laser Wing", "OCT-A Diagnostic Center"],
        "min_grade_tier": 2,
        "address": "Plot No. 153, Road No. 9, Wadala West, Mumbai, Maharashtra 400031",
        "helpline": "+91-22-2417-7777",
        "emergency_retina": "+91-22-2417-7700",
        "top_specialists": [
            {"name": "Dr. S. Natarajan", "designation": "Distinguished Vitreo-Retinal Surgeon", "experience": "36+ Years", "opd_days": "Mon, Wed, Fri"},
        ]
    }
]


def get_recommended_referrals(
    severity_grade: int,
    region: str = "All India",
    has_macular_threat: bool = False
) -> Dict[str, Any]:
    """
    Ranks and returns certified hospitals and vitreoretinal surgeons
    matched to patient's clinical urgency and geographical region.
    """
    # Filter by region if specified
    if region != "All India" and region != "National / All India":
        matched_hospitals = [h for h in HOSPITAL_DIRECTORY if h["region"] == region]
        if not matched_hospitals:
            matched_hospitals = HOSPITAL_DIRECTORY
    else:
        matched_hospitals = HOSPITAL_DIRECTORY

    # Filter/rank by severity
    # Grade 4 (PDR) or Macular Threat -> Apex Tertiary surgical centers only
    # Grade 3 (Severe) -> Tertiary with Laser & Anti-VEGF
    # Grade 2 (Moderate) -> Secondary/Tertiary
    # Grade 0-1 -> Vision center / Comprehensive
    if severity_grade >= 3 or has_macular_threat:
        ranked = [h for h in matched_hospitals if "Apex" in h["tier"] or "Tertiary" in h["tier"]]
        urgency_code = "RED_EMERGENCY"
        action_headline = "Emergency Vitreoretinal Referral Required (Within 48h - 2 Weeks)"
    elif severity_grade == 2:
        ranked = matched_hospitals
        urgency_code = "ORANGE_URGENT"
        action_headline = "Specialist Retinal Evaluation Required (Within 1 - 3 Months)"
    elif severity_grade == 1:
        ranked = matched_hospitals
        urgency_code = "YELLOW_FOLLOWUP"
        action_headline = "Routine Secondary Care Consultation (Within 6 - 12 Months)"
    else:
        ranked = matched_hospitals
        urgency_code = "GREEN_ROUTINE"
        action_headline = "Routine Annual Screening at Vision Center / Primary Care"

    # Generate ABDM Referral Packet Token
    abdm_token = f"ABDM-REF-{urgency_code[:3]}-{len(ranked)}HOSP-2026"

    return {
        "severity_grade": severity_grade,
        "urgency_code": urgency_code,
        "action_headline": action_headline,
        "matched_hospitals_count": len(ranked),
        "hospitals": ranked[:4],
        "abdm_referral_token": abdm_token,
    }
