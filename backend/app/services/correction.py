import os
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
    },
    "plank": {
        "elbow_l": {
            "issue_low": {
                "en": "Extend your left elbow to keep your arm straight.",
                "hi": "अपने बाएं हाथ को सीधा रखने के लिए अपनी बाईं कोहनी को फैलाएं।",
                "bn": "আপনার বাম হাত সোজা রাখতে বাম কনুই প্রসারিত করুন।"
            }
        },
        "elbow_r": {
            "issue_low": {
                "en": "Extend your right elbow to keep your arm straight.",
                "hi": "अपने दाएं हाथ को सीधा रखने के लिए अपनी दाईं कोहनी को फैलाएं।",
                "bn": "আপনার ডান হাত সোজা রাখতে ডান কনুই প্রসারিত করুন।"
            }
        },
        "trunk_l": {
            "issue_low": {
                "en": "Keep your spine and body in a straight line, do not let your hips sag.",
                "hi": "अपनी रीढ़ और शरीर को एक सीधी रेखा में रखें, अपने नितंबों को नीचे न झुकने दें।",
                "bn": "আপনার মেরুদণ্ড এবং শরীর সোজা রাখুন, নিতম্ব ঝুলে যেতে দেবেন না।"
            }
        }
    },
    "tree_pose": {
        "knee_l": {
            "issue_low": {
                "en": "Bend your left knee more and place your foot on your inner thigh.",
                "hi": "अपने बाएं घुटने को अधिक मोड़ें और अपने पैर को अपनी आंतरिक जांघ पर रखें।",
                "bn": "আপনার বাম হাঁটু আরও বাঁকুন এবং পায়ের পাতাটি উরুর ভেতরের দিকে রাখুন।"
            }
        },
        "hip_abduct_l": {
            "issue_low": {
                "en": "Open your left hip outward to bring your knee to the side.",
                "hi": "अपने बाएं कूल्हे को बाहर की ओर खोलें ताकि आपका घुटना बगल की ओर हो सके।",
                "bn": "আপনার বাম হাঁটু পাশের দিকে আনতে বাম হিপ বাইরের দিকে প্রসারিত করুন।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Keep your standing right leg fully straight and strong.",
                "hi": "अपने खड़े हुए दाएं पैर को पूरी तरह से सीधा और मजबूत रखें।",
                "bn": "দাঁড়িয়ে থাকা ডান পা সম্পূর্ণ সোজা এবং শক্ত রাখুন।"
            }
        }
    },
    "mountain_pose": {
        "knee_l": {
            "issue_low": {
                "en": "Keep your left leg straight and engage your thigh muscles.",
                "hi": "अपने बाएं पैर को सीधा रखें और अपनी जांघ की मांसपेशियों को सक्रिय करें।",
                "bn": "আপনার বাম পা সোজা রাখুন এবং উরুর পেশী সক্রিয় করুন।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Keep your right leg straight and engage your thigh muscles.",
                "hi": "अपने दाएं पैर को सीधा रखें और अपनी जांघ की मांसपेशियों को सक्रिय करें।",
                "bn": "আপনার ডান পা সোজা রাখুন এবং উরুর পেশী সক্রিয় করুন।"
            }
        },
        "trunk_l": {
            "issue_low": {
                "en": "Lengthen your spine and stand tall with shoulders relaxed.",
                "hi": "अपनी रीढ़ को लंबा करें और कंधों को आराम देते हुए सीधे खड़े हों।",
                "bn": "মেরুদণ্ড সোজা করে বুক টানটান করে দাঁড়ান এবং কাঁধ শিথিল রাখুন।"
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
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Map language code to full language name to instruct the LLM
            lang_names = {"en": "English", "hi": "Hindi", "bn": "Bengali"}
            target_lang_name = lang_names.get(language, "English")
            
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a professional yoga instructor. Translate or paraphrase the following instruction: '{correction_text}' into {target_lang_name}. Do NOT suggest stretching further or pushing deeper. Keep it safe, concise (under 15 words), and respond ONLY in the {target_lang_name} language."
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
        except Exception as e:
            logger.warning(f"Groq correction call failed, falling back to template: {e}")
            
    return correction_text, is_safe
