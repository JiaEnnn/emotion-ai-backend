import os
import json
import base64
import numpy as np
import cv2
import tensorflow as tf
# Force TensorFlow to use as little memory as possible
tf.config.set_visible_devices([], 'GPU')
import mediapipe as mp
import requests
import wave
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. INITIALIZE FLASK
app = Flask(__name__)
CORS(app)

# --- Voice Mapping for Gemini TTS ---
VOICE_PROFILES = {
    "Young Cutesy Girl": "Aoede",
    "Young Cutesy Boy": "Puck",
    "Teen Girl": "Kore",
    "Teen Boy": "Zephyr",
    "Mature Woman": "Charon",
    "Mature Man": "Fenrir",
    "Old Woman": "Leda",
    "Old Man": "Enceladus"
}

PERSONALITIES = {
    "Young Cutesy Girl": "You are 'Aoede'. Use light, cheerful language. Use emojis like ✨ and 🌸. You are like an energetic younger sister who wants to cheer everyone up.",
    "Young Cutesy Boy": "You are 'Puck'. You are playful, a bit mischievous, and very high energy. You use words like 'Cool!' and 'Awesome!'",
    "Teen Girl": "You are 'Kore'. You are relatable, uses modern slang sparingly, and focuses on being a supportive best friend. You're empathetic but keep it casual.",
    "Teen Boy": "You are 'Zephyr'. You are chill, down-to-earth, and use relaxed language. You act like a supportive brother who's got the user's back.",
    "Mature Woman": "You are 'Charon'. You are calm, sophisticated, and deeply therapeutic. You speak with wisdom and poise, like a professional mentor.",
    "Mature Man": "You are 'Fenrir'. You have a deep, steady presence. You are protective, calm, and provide a 'rock' for the user to lean on.",
    "Old Woman": "You are 'Leda'. You are maternal, gentle, and slow-paced. You call the user 'dear' or 'my child' and offer comfort like a grandmother.",
    "Old Man": "You are 'Enceladus'. You are a wise sage. You use metaphors and provide perspective, like an old professor who has seen it all."
}

# 2. INITIALIZE SPECIALISTS
print("🔄 Starting System Initialization...")

cnn_model = None
model_path = 'cnn_master_global_personal.h5'

def load_cnn():
    global cnn_model
    try:
        cnn_model = tf.keras.models.load_model(model_path, compile=False)
        if hasattr(cnn_model, 'optimizer'):
            del cnn_model.optimizer
            
        return "✅ CNN Texture Expert: Online"
    except Exception as e:
        return f"❌ CNN CRITICAL LOAD ERROR: {e}"    
    #     return "✅ CNN Texture Expert: Online (via tf.keras)"
    # except Exception as e:
    #     return f"❌ CNN CRITICAL LOAD ERROR: {e}"

print(load_cnn())

mlp_data = None
try:
    with open('mlp_v3_master.json', 'r') as f:
        mlp_data = json.load(f)
    print("✅ MLP Geometry Data: Loaded.")
except Exception as e:
    print(f"❌ MLP Data Error: {e}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)
print("✅ MediaPipe FaceMesh: Initialized.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

# --- AUDIO UTILITIES ---
def pcm_to_wav(pcm_data, sample_rate=24000):
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()

def generate_empathetic_audio(text, profile_name):
    voice_name = VOICE_PROFILES.get(profile_name, "Kore")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_API_KEY}"
    prompt = f"Say this as a very empathetic, warm, and therapeutic {profile_name}: {text}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            }
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()
        inline_data = result['candidates'][0]['content']['parts'][0]['inlineData']
        pcm_bytes = base64.b64decode(inline_data['data'])
        wav_bytes = pcm_to_wav(pcm_bytes)
        return base64.b64encode(wav_bytes).decode('utf-8')
    except Exception as e:
        print(f"⚠️ TTS Generation Failed: {e}")
        return None

def run_mlp_inference(features):
    if mlp_data is None: return np.array([0, 0, 0, 0, 0, 0, 1.0])
    res = (np.array(features) - np.array(mlp_data['scaler_mean'])) / np.array(mlp_data['scaler_scale'])
    for i in range(len(mlp_data['weights'])):
        res = np.dot(res, np.array(mlp_data['weights'][i])) + np.array(mlp_data['biases'][i])
        if i < len(mlp_data['weights']) - 1: 
            res = np.maximum(0, res) 
    e_x = np.exp(res - np.max(res))
    return e_x / e_x.sum()

def process_single_frame(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_bgr.shape
        results = face_mesh.process(img_rgb)
        
        m_probs = {e: 0.0 for e in EMOTIONS}
        c_probs = {e: 0.0 for e in EMOTIONS}
        m_probs['Neutral'] = 1.0
        c_probs['Neutral'] = 1.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = np.array([[lm.x, lm.y] for lm in landmarks])
            centered = (coords - coords[1]).flatten()
            m_out = run_mlp_inference(centered)
            m_probs = {EMOTIONS[i]: float(m_out[i]) for i in range(len(EMOTIONS))}

            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            mx, my = int((x2-x1)*0.25), int((y2-y1)*0.25)
            face_crop_bgr = img_bgr[max(0, y1-my):min(h, y2+my), max(0, x1-mx):min(w, x2+mx)]
            
            if face_crop_bgr.size > 0:
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_crop_rgb, (112, 112))
                reshaped = resized.reshape(1, 112, 112, 3).astype('float32')
                c_out = cnn_model.predict(reshaped, verbose=0)[0]
                c_probs = {EMOTIONS[i]: float(c_out[i]) for i in range(len(EMOTIONS))}

        return {"cnn": c_probs, "mlp": m_probs}
    except Exception as e:
        print(f"❌ Frame Processing Error: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        user_text = data.get('text', "")
        history_frames = data.get('history', [])
        selected_voice = data.get('voice', "Teen Girl")
        
        timeline_summary = []
        for idx, frame_b64 in enumerate(history_frames):
            res = process_single_frame(frame_b64)
            if res:
                top_c = max(res['cnn'], key=res['cnn'].get)
                top_m = max(res['mlp'], key=res['mlp'].get)
                timeline_summary.append({
                    "time": f"T-{(len(history_frames)-idx)*3}s",
                    "cnn": top_c, "mlp": top_m
                })

        final_res = process_single_frame(data['image'])
        if not final_res:
            return jsonify({"error": "Failed to process final image"}), 400

        top_cnn = max(final_res['cnn'], key=final_res['cnn'].get)
        top_mlp = max(final_res['mlp'], key=final_res['mlp'].get)

        # --- RESTORED: FULL LIVE DASHBOARD (THE BARS) ---
        print("\n" + "🔥" * 20)
        print(f"💬 USER: '{user_text}'")
        print("-" * 40)
        print("📈 EMOTIONAL TIMELINE (Every 3s):")
        if not timeline_summary: 
            print("  (No history frames received yet)")
        for t in timeline_summary:
            print(f"  {t['time']} | CNN: {t['cnn']:8} | MLP: {t['mlp']}")
        
        print("-" * 40)
        print("🧬 FINAL CNN (TEXTURE) BREAKDOWN:")
        for emo in EMOTIONS:
            bar = "█" * int(final_res['cnn'][emo] * 20)
            print(f"  {emo:10} | {final_res['cnn'][emo]:.1%} {bar}")
        
        print("\n📐 FINAL MLP (GEOMETRY) BREAKDOWN:")
        for emo in EMOTIONS:
            bar = "█" * int(final_res['mlp'][emo] * 20)
            print(f"  {emo:10} | {final_res['mlp'][emo]:.1%} {bar}")
        print("🔥" * 20 + "\n")

        # 3. THE ARBITER
        try:
            current_personality = PERSONALITIES.get(selected_voice, PERSONALITIES["Teen Girl"])

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""{current_personality}

                        You are an empathetic AI therapist. 
                        You analyze user text vs their emotional timeline while typing.
                        - If CNN and MLP disagree, prioritize the one with higher confidence.
                        - If they are smiling (Happy in MLP) but CNN says Sad, trust the MLP more.
                        - If the user says they are fine but looks 'Sad', address the contradiction gently.
                        - ALWAYS stay in character based on your assigned personality.
                        - Keep responses concise (2-3 sentences) so the TTS doesn't take too long.
                        
                        Respond in JSON: {{'chatbot_reply': '...', 'detected_emotion': '...'}}"""
                    },
                    {"role": "user", "content": f"Text: '{user_text}'\nFinal Visuals: CNN:{final_res['cnn']}, MLP:{final_res['mlp']}\nHistory: {timeline_summary}"}
                ],
                response_format={ "type": "json_object" }
            )
            arbiter_data = json.loads(response.choices[0].message.content)
            bot_reply = arbiter_data.get("chatbot_reply")
            
            audio_b64 = generate_empathetic_audio(bot_reply, selected_voice)
            
            return jsonify({
                "chatbot_reply": bot_reply,
                "detected_emotion": arbiter_data.get("detected_emotion"),
                "audio": audio_b64,
                "cnn_top": top_cnn, "cnn_conf": final_res['cnn'][top_cnn],
                "mlp_top": top_mlp, "mlp_conf": final_res['mlp'][top_mlp]
            })
        except Exception as api_err:
            final_v = top_cnn if final_res['cnn'][top_cnn] > final_res['mlp'][top_mlp] else top_mlp
            fallback_text = f"I've been watching you. You seem to be feeling {final_v.lower()}."
            audio_b64 = generate_empathetic_audio(fallback_text, selected_voice)
            return jsonify({
                "chatbot_reply": fallback_text,
                "detected_emotion": final_v,
                "audio": audio_b64,
                "cnn_top": top_cnn, "cnn_conf": final_res['cnn'][top_cnn],
                "mlp_top": top_mlp, "mlp_conf": final_res['mlp'][top_mlp]
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)