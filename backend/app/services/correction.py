import requests
from typing import Dict, Optional

# Biomechanical knowledge graph templates
BIOMECHANICAL_TEMPLATES = {
    "warrior_2": {
        "knee_l": {
            "issue_low": {
                "en": "Bend your left knee more to bring it directly over your ankle.",
                "hi": "अपने बाएं घुटने को थोड़ा और मोड़ें ताकि वह टखने के ठीक ऊपर रहे।",
                "bn": "আপনার বাম হাঁটু আরও বাঁকুন যাতে এটি গোড়ালির ঠিক উপরে থাকে।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Bend your right knee more to align it directly over your ankle.",
                "hi": "अपने दाएं घुटने को थोड़ा और मोड़ें ताकि वह टखने के ठीक ऊपर रहे।",
                "bn": "আপনার ডান হাঁটু আরও বাঁকুন যাতে এটি গোড়ালির ঠিক উপরে থাকে।"
            }
        }
    },
    "chair_pose": {
        "knee_l": {
            "issue_low": {
                "en": "Sit deeper to bring your thighs closer to parallel to the floor.",
                "hi": "जांघों को फर्श के समानांतर लाने के लिए थोड़ा नीचे बैठें।",
                "bn": "উরু মেঝের সমান্তরাল করার জন্য আর একটু নিচে বসুন।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Sit deeper to align both thighs towards parallel to the floor.",
                "hi": "दोनों जांघों को फर्श के समानांतर करने के लिए थोड़ा नीचे बैठें।",
                "bn": "দুই উরু মেঝের সমান্তরাল করার জন্য আর একটু নিচে বসুন।"
            }
        }
    },
    "cobra_pose": {
        "neck": {
            "issue_low": {
                "en": "Elongate your neck and gaze forward, relaxing your shoulders.",
                "hi": "अपनी गर्दन को लंबी करें और सामने की ओर देखें, कंधों को ढीला छोड़ें।",
                "bn": "আপনার ঘাড় সোজা করুন এবং সামনের দিকে তাকান, কাঁধ শিথিল রাখুন।"
            }
        }
    }
}

DEFAULT_CORRECTIONS = {
    "en": "Maintain your posture, breathing steadily.",
    "hi": "अपनी मुद्रा बनाए रखें, लगातार सांस लेते रहें।",
    "bn": "আপনার অঙ্গভঙ্গি বজায় রাখুন, নিয়মিত শ্বাস নিন।"
}

def generate_safe_correction(
    pose_id: str,
    deviations: Dict[str, float],
    language: str = "en",
    groq_api_key: Optional[str] = None
) -> tuple:
    """
    Enforces a 3-stage validation pipeline:
    Stage 1: Symbolic Rules Engine + Local template matching.
    Stage 2: Optional Groq API translation/paraphrase inside safe boundaries.
    Stage 3: Post-generation safety validation screening.
    """
    pose_templates = BIOMECHANICAL_TEMPLATES.get(pose_id, {})
    
    # Stage 1: Find highest deviation joint exceeding safe threshold (10 degrees)
    target_joint = None
    max_dev = 0.0
    for joint, dev_val in deviations.items():
        if joint in pose_templates and dev_val > max_dev:
            if dev_val > 10.0:
                max_dev = dev_val
                target_joint = joint
                
    # Retrieve pre-approved template
    if target_joint:
        correction_text = pose_templates[target_joint]["issue_low"].get(language, "Adjust your alignment.")
    else:
        correction_text = DEFAULT_CORRECTIONS.get(language, "Adjust your alignment.")
        
    is_safe = True
    
    # Stage 2: Paraphrase using LLM if Groq API key is available
    if groq_api_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a professional yoga instructor. Translate or paraphrase the following instruction: '{correction_text}'. Do NOT suggest stretching further or pushing deeper. Keep it safe and under 15 words."
                    }
                ],
                "temperature": 0.2
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                candidate_text = response.json()["choices"][0]["message"]["content"].strip()
                
                # Stage 3: Post-Generation Safety Validation
                forbidden_words = ["push", "force", "hurt", "pain", "stretch more", "further"]
                if not any(word in candidate_text.lower() for word in forbidden_words):
                    correction_text = candidate_text
                else:
                    is_safe = False # Flag candidate as unsafe; fallback to rule-based template
        except Exception:
            pass # Fallback to pre-approved templates
            
    return correction_text, is_safe
