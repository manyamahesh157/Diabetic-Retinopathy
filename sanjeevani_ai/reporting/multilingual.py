"""
sanjeevani_ai.reporting.multilingual
------------------------------------
Multilingual Rural Patient Education Leaflet Engine.

Translates clinical DR screening conclusions and self-care recommendations
into regional Indian languages (Hindi, Kannada, Tamil, Telugu, English)
for rural primary health centers and mobile vision camps.
"""

from typing import Dict, Any

MULTILINGUAL_LEAFLETS = {
    "English": {
        0: {
            "title": "Eye Screening Summary: Normal (No Diabetic Retinopathy)",
            "message": "Your retinal photo appears healthy with no visible signs of diabetic damage. Maintain strict control of blood sugar and blood pressure. Get your eyes screened again in 12 months.",
            "action": "Next screening in 1 year. Regular daily walks and balanced diabetic diet."
        },
        1: {
            "title": "Eye Screening Summary: Mild Retinopathy (Early Signs)",
            "message": "Minor tiny red micro-dots (microaneurysms) were noticed in your retina. This is an early warning sign. Vision loss can be prevented by controlling HbA1c and cholesterol.",
            "action": "Repeat retinal screening in 9 to 12 months. Consult your primary physician to optimize diabetes medications."
        },
        2: {
            "title": "Eye Screening Summary: Moderate Retinopathy (Referral Needed)",
            "message": "Scattered bleeding spots and lipid deposits were found in the retina. Progression can threaten your eyesight if left unattended.",
            "action": "Please visit an Eye Specialist (Ophthalmologist) within 3 to 6 months for a detailed dilated examination."
        },
        3: {
            "title": "Eye Screening Summary: Severe Retinopathy (High Priority)",
            "message": "Extensive blood vessel damage noticed across your retina. High risk of progressing to eyesight-threatening disease.",
            "action": "Urgent consultation with an eye specialist within 2 to 4 weeks. Specialized laser or injection treatments may be discussed."
        },
        4: {
            "title": "Eye Screening Summary: Proliferative Retinopathy (Emergency)",
            "message": "Abnormal, fragile new blood vessels have started growing inside the eye. These can bleed and cause sudden vision loss.",
            "action": "URGENT HOSPITAL VISIT REQUIRED within 48 to 72 hours. Immediate specialist treatment (retina laser/injections) is essential to preserve your vision."
        },
    },
    "Hindi (हिंदी)": {
        0: {
            "title": "नेत्र जांच सारांश: सामान्य (डायबिटीज का कोई लक्षण नहीं)",
            "message": "आपके पर्दे (रेटिना) की फोटो स्वस्थ दिखती है। ब्लड शुगर और ब्लड प्रेशर को नियंत्रित रखें। 12 महीने बाद दोबारा आंखों की जांच कराएं।",
            "action": "अगली जांच 1 वर्ष बाद। नियमित व्यायाम और संतुलित आहार लें।"
        },
        1: {
            "title": "नेत्र जांच सारांश: हल्का रेटिनोपैथी (शुरुआती चेतावनी)",
            "message": "आंख के पर्दे पर खून की नसों में छोटे लाल धब्बे दिखे हैं। समय पर शुगर कंट्रोल करने से आंखों की रोशनी पूरी तरह सुरक्षित रह सकती है।",
            "action": "9 से 12 महीने में दोबारा जांच कराएं। अपने डॉक्टर से मिलकर शुगर की दवाइयों की समीक्षा करें।"
        },
        2: {
            "title": "नेत्र जांच सारांश: मध्यम रेटिनोपैथी (विशेषज्ञ सलाह जरूरी)",
            "message": "रेटिना में खून के धब्बे और पीले वसा के जमाव दिखे हैं। बिना इलाज के यह आपकी रोशनी को नुकसान पहुंचा सकता है।",
            "action": "अगले 3 से 6 महीने के भीतर किसी नेत्र विशेषज्ञ (आई डॉक्टर) से पुतली फैलाकर पूरी जांच कराएं।"
        },
        3: {
            "title": "नेत्र जांच सारांश: गंभीर रेटिनोपैथी (उच्च प्राथमिकता)",
            "message": "आंख के पर्दे में रक्त वाहिकाओं को व्यापक नुकसान हुआ है। आंखों की रोशनी को बड़ा खतरा हो सकता है।",
            "action": "2 से 4 सप्ताह के भीतर किसी रेटिना विशेषज्ञ से मिलें। लेजर या इंजेक्शन की आवश्यकता हो सकती है।"
        },
        4: {
            "title": "नेत्र जांच सारांश: अति-गंभीर प्रोलिफेरेटिव रेटिनोपैथी (आपातकालीन)",
            "message": "आंख के भीतर नई नाजुक रक्त वाहिकाएं बनने लगी हैं, जिनसे खून बहने और अचानक रोशनी जाने का गंभीर खतरा है।",
            "action": "48 से 72 घंटों के भीतर तत्काल बड़े नेत्र अस्पताल जाएं। रोशनी बचाने के लिए तुरंत विशेषज्ञ उपचार आवश्यक है।"
        },
    },
    "Kannada (ಕನ್ನಡ)": {
        0: {
            "title": "ಕಣ್ಣಿನ ತಪಾಸಣಾ ಸಾರಾಂಶ: ಸಾಮಾನ್ಯ (ಯಾವುದೇ ಹಾನಿಯಿಲ್ಲ)",
            "message": "ನಿಮ್ಮ ಕಣ್ಣಿನ ಪರದೆಯು (ರೆಟಿನಾ) ಆರೋಗ್ಯಕರವಾಗಿ ಕಂಡುಬಂದಿದೆ. ರಕ್ತದ ಸಕ್ಕರೆ ಮತ್ತು ರಕ್ತದೊತ್ತಡವನ್ನು ನಿಯಂತ್ರಣದಲ್ಲಿಡಿ. 12 ತಿಂಗಳಲ್ಲಿ ಮತ್ತೊಮ್ಮೆ ಪರೀಕ್ಷಿಸಿಕೊಳ್ಳಿ.",
            "action": "ಮುಂದಿನ ತಪಾಸಣೆ 1 ವರ್ಷದ ನಂತರ. ನಿತ್ಯ ನಡಿಗೆ ಮತ್ತು ಪಥ್ಯ ಆಹಾರ ಪಾಲಿಸಿ."
        },
        1: {
            "title": "ಕಣ್ಣಿನ ತಪಾಸಣಾ ಸಾರಾಂಶ: ಸೌಮ್ಯ ರೆಟಿನೋಪತಿ (ಆರಂಭಿಕ ಎಚ್ಚರಿಕೆ)",
            "message": "ಕಣ್ಣಿನ ಪರದೆಯಲ್ಲಿ ರಕ್ತನಾಳಗಳ ಸಣ್ಣ ಕೆಂಪು ಚುಕ್ಕೆಗಳು ಕಂಡುಬಂದಿವೆ. ಸಕ್ಕರೆ ನಿಯಂತ್ರಣದಿಂದ ದೃಷ್ಟಿ ನಷ್ಟವನ್ನು ತಡೆಯಬಹುದು.",
            "action": "9 ರಿಂದ 12 ತಿಂಗಳಲ್ಲಿ ಮರುಪರೀಕ್ಷೆ ಮಾಡಿಸಿ. ನಿಮ್ಮ ವೈದ್ಯರೊಂದಿಗೆ ಸಮಾಲೋಚಿಸಿ."
        },
        2: {
            "title": "ಕಣ್ಣಿನ ತಪಾಸಣಾ ಸಾರಾಂಶ: ಮಧ್ಯಮ ರೆಟಿನೋಪತಿ (ವೈದ್ಯರ ಭೇಟಿ ಅಗತ್ಯ)",
            "message": "ಕಣ್ಣಿನ ಪರದೆಯಲ್ಲಿ ರಕ್ತದ ಕಲೆಗಳು ಮತ್ತು ಹಳದಿ ಕೊಬ್ಬಿನ ನಿಕ್ಷೇಪಗಳು ಕಂಡುಬಂದಿವೆ. ಚಿಕಿತ್ಸೆ ಪಡೆಯದಿದ್ದರೆ ದೃಷ್ಟಿಗೆ ಹಾನಿಯಾಗಬಹುದು.",
            "action": "ಮುಂದಿನ 3 ರಿಂದ 6 ತಿಂಗಳೊಳಗೆ ನೇತ್ರ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ ವಿವರವಾದ ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ."
        },
        3: {
            "title": "ಕಣ್ಣಿನ ತಪಾಸಣಾ ಸಾರಾಂಶ: ತೀವ್ರ ರೆಟಿನೋಪತಿ (ಹೆಚ್ಚಿನ ಆದ್ಯತೆ)",
            "message": "ಕಣ್ಣಿನ ರಕ್ತನಾಳಗಳಿಗೆ ಹೆಚ್ಚಿನ ಹಾನಿಯಾಗಿದೆ. ದೃಷ್ಟಿ ಅಪಾಯದಲ್ಲಿರುವ ಸಾಧ್ಯತೆಯಿದೆ.",
            "action": "2 ರಿಂದ 4 ವಾರಗಳೊಳಗೆ ಕಣ್ಣಿನ ರೆಟಿನಾ ತಜ್ಞರನ್ನು ಭೇಟಿ ಮಾಡಿ. ಲೇಸರ್ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿರಬಹುದು."
        },
        4: {
            "title": "ಕಣ್ಣಿನ ತಪಾಸಣಾ ಸಾರಾಂಶ: ತುರ್ತು ಪ್ರೊಲಿಫರೇಟಿವ್ ರೆಟಿನೋಪತಿ (ಅಪಾಯಕಾರಿ)",
            "message": "ಕಣ್ಣಿನೊಳಗೆ ಅಸಹಜ ರಕ್ತನಾಳಗಳು ಬೆಳೆಯುತ್ತಿದ್ದು, ದಿಢೀರ್ ದೃಷ್ಟಿಹೀನತೆಯ ಅಪಾಯವಿದೆ.",
            "action": "48 ರಿಂದ 72 ಗಂಟೆಗಳೊಳಗೆ ತುರ್ತಾಗಿ ಕಣ್ಣಿನ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ ಚಿಕಿತ್ಸೆ ಪಡೆಯುವುದು ಅತ್ಯಗತ್ಯ."
        },
    },
    "Tamil (தமிழ்)": {
        0: {
            "title": "கண் பரிசோதனை முடிவு: இயல்பு நிலை (பாதிப்பில்லை)",
            "message": "உங்கள் விழித்திரை ஆரோக்கியமாக உள்ளது. ரத்த சர்க்கரை அளவை கட்டுப்பாட்டில் வையுங்கள். 12 மாதங்களுக்குப் பிறகு மீண்டும் பரிசோதிக்கவும்.",
            "action": "அடுத்த பரிசோதனை 1 வருடத்தில். தினசரி உடற்பயிற்சி மற்றும் சர்க்கரை கட்டுப்பாடு அவசியம்."
        },
        2: {
            "title": "கண் பரிசோதனை முடிவு: மிதமான ரெட்டினோபதி (மருத்துவர் ஆலோசனை தேவை)",
            "message": "விழித்திரையில் சிறிய ரத்தக் கசிவுகள் மற்றும் கொழுப்பு படிவுகள் தெரிகின்றன. பார்வை இழப்பைத் தடுக்க சிகிச்சை தேவை.",
            "action": "அடுத்த 3 முதல் 6 மாதங்களுக்குள் கண் மருத்துவரை நேரில் சந்தித்து முழு விழித்திரை பரிசோதனை செய்துகொள்ளவும்."
        },
        4: {
            "title": "கண் பரிசோதனை முடிவு: அவசர நிலை (பார்வை இழப்பு அபாயம்)",
            "message": "கண்ணுக்குள் ஆபத்தான புதிய ரத்த நாளங்கள் வளரத் தொடங்கியுள்ளன. இதனால் திடீர் பார்வை இழப்பு ஏற்படலாம்.",
            "action": "48 முதல் 72 மணி நேரத்திற்குள் உடனடியாக கண் மருத்துவமனைக்கு சென்று சிகிச்சை பெற வேண்டும்."
        }
    }
}


def get_multilingual_patient_guidance(grade: int, language: str = "English") -> Dict[str, str]:
    """Retrieves patient education text for the given grade and language."""
    lang_dict = MULTILINGUAL_LEAFLETS.get(language, MULTILINGUAL_LEAFLETS["English"])
    # Fallback to English if grade not present in dialect
    if grade in lang_dict:
        return lang_dict[grade]
    elif grade in MULTILINGUAL_LEAFLETS["English"]:
        return MULTILINGUAL_LEAFLETS["English"][grade]
    else:
        return MULTILINGUAL_LEAFLETS["English"][0]
