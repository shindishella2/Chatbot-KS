from fpdf import FPDF
import random
import re
import time, faiss, numpy as np, pickle, os
import torch.nn.functional as F
import torch
from datetime import datetime
import streamlit as st
from streamlit.components.v1 import html as components_html
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq


st.set_page_config(page_title="Ruang Aman - Konseling Hukum UU TPKS",
                   page_icon="\U0001f49b", layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_embed(): return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
@st.cache_resource
def load_store():
    return faiss.read_index("faiss_index.index"), pickle.load(open("chunks.pkl","rb"))
embed_model = load_embed()
index, chunks = load_store()

EMOTION_MODEL_REPO = "Chatbot-123/Chatbot-KS"
@st.cache_resource
def load_emotion_model():
    tok = AutoTokenizer.from_pretrained(EMOTION_MODEL_REPO)
    mdl = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_REPO)
    mdl.eval()
    return tok, mdl

def detect_emotion(text: str) -> dict:
    tokenizer, model = load_emotion_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    id2label_local = model.config.id2label
    scores = {id2label_local[i]: float(probs[i]) for i in range(len(probs))}
    dominant = max(scores, key=scores.get)
    return {"label_dominan": dominant, "confidence": scores[dominant], "semua_skor": scores}

DISTRESS_MAP = {
    "sadness": "tinggi",
    "fear": "tinggi",
    "anger": "sedang",
    "happy": "rendah",
    "love": "rendah",
}

def get_support_flag(text: str, threshold: float = 0.5, sadness_safety_threshold: float = 0.30) -> dict:
    r = detect_emotion(text)
    dominant = r["label_dominan"]
    level = DISTRESS_MAP.get(dominant, "rendah")
    if r["confidence"] < threshold:
        level = "rendah"
    sadness_score = r["semua_skor"].get("sadness", 0)
    if sadness_score >= sadness_safety_threshold:
        level = "tinggi"
    return {
        "emosi": dominant,
        "confidence": r["confidence"],
        "sadness_score": sadness_score,
        "distress_level": level,
        "perlu_rujukan": level == "tinggi",
    }
def generate_ai_support_label(api_key, emotion: str, user_text: str) -> str:
    if not api_key:
        return "Siap membantu dan mendengarkan ceritamu."
        
    # Arahan spesifik berdasarkan emosi yang terdeteksi
    instruction_map = {
        "sadness": "berikan 1 kalimat penguat yang sangat lembut, tunjukkan empati mendalam dan bahwa kamu ada untuknya.",
        "fear": "berikan 1 kalimat yang menenangkan kekhawatirannya, berikan kepastian bahwa dia aman bercerita di sini.",
        "anger": "berikan 1 kalimat validasi yang adem untuk menurunkan tensi emosinya tanpa menghakimi kekesalannya.",
        "happy": "berikan 1 kalimat ikut senang, antusias, dan mengapresiasi kabar baik atau energi positifnya.",
        "love": "berikan 1 kalimat apresiasi yang hangat atas kasih sayang atau cerita indahnya."
    }
    
    context_instruction = instruction_map.get(emotion, "berikan 1 kalimat respons yang hangat dan suportif.")
    
    system_prompt = (
        f"Kamu adalah konselor psikologis yang sangat empatik dan peka. "
        f"User baru saja bercerita dan sistem mendeteksi emosinya adalah '{emotion}'. "
        f"Berdasarkan potongan pesannya, {context_instruction}\n\n"
        "ATURAN MUTLAK:\n"
        "- HANYA hasilkan 1 kalimat pendek saja (maksimal 12-15 kata).\n"
        "- Gunakan bahasa Indonesia santai/kasual yang sangat natural seperti teman dekat (jangan kaku/formal).\n"
        "- DILARANG memakai tanda kutip atau kalimat pengantar seperti 'Ini kalimatnya:'."
    )
    
    try:
        client = Groq(api_key=api_key)
        # Gunakan model 8B agar super cepat (low latency)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pesan user: {user_text}"}
            ],
            temperature=0.85,
            max_tokens=50
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        # Fallback (cadangan) jika API Groq mendadak error/limit
        fallback = {
            "sadness": "Aku di sini mendengarkanmu. Ceritakan saja semuanya, ya.",
            "fear": "Kamu aman di sini. Tarik napas dalam-dalam, kita lalui bersama.",
            "anger": "Wajar kok kalau kamu kesal. Yuk, rehat sejenak dan tenangin pikiran.",
            "happy": "Ikut senang mendengarnya! Cerita seru apa lagi nih?",
            "love": "Terima kasih ya sudah berbagi energi positif. Kamu berharga!"
        }
        return fallback.get(emotion, "Siap membantu")


SUPPORT_MESSAGES = {
    "sadness": [
        "\U0001f90d Apa pun yang kamu rasakan sekarang itu valid. Kamu gak sendirian di sini.",
        "\U0001f90d Terima kasih udah mau cerita. Pelan-pelan aja, gak perlu buru-buru.",
        "\U0001f90d Kamu udah berani sejauh ini dengan cerita di sini. Itu bukan hal kecil.",
    ],
    "fear": [
        "\U0001fac2 Kamu aman untuk cerita di sini, dengan kecepatanmu sendiri.",
        "\U0001fac2 Gak apa-apa kalau masih takut. Kamu boleh berhenti kapan pun kamu perlu.",
        "\U0001fac2 Perasaan itu wajar. Kita jalan pelan-pelan aja, sesuai kesiapanmu.",
    ],
}

def get_support_banner(emotion_label: str):
    pool = SUPPORT_MESSAGES.get(emotion_label)
    return random.choice(pool) if pool else None
# ===================== TEMA — sesuai referensi gambar =====================
T = dict(
    navy="#0E1B48", mauve="#C18DB4", blush="#E2CAD8", skyblue="#87A7D0",
    slate="#27425D", deep="#0E1F2F",
    sidebar_top="#FDE8D3", sidebar_bottom="#F5D7DB", active="#F1916D",
    bot_bg="#F0C987", bot_border="#F59E51", bot_text="#3B153A",
    user_bg="#0E1B48", user_border="#87A7D0", user_text="#FFFFFF",
    header_title="#F0C987", header_sub="#E6E6E6",
    appbg="linear-gradient(135deg, rgba(14,27,72,0.30) 0%, rgba(193,141,180,0.30) 25%, rgba(226,202,216,0.30) 50%, rgba(135,167,208,0.30) 75%, rgba(39,66,93,0.30) 100%), #0E1F2F",
)

# Logo pakai SVG (bentuk hands+heart sesuai referensi).
LOGO_SVG = """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 8.6c-1-1.7-2.7-2.6-4.4-2.1C5.6 7 4.6 8.9 5.2 10.7c.6 2 3 4.1 6.8 6.9 3.8-2.8 6.2-4.9 6.8-6.9.6-1.8-.4-3.7-2.4-4.2-1.7-.5-3.4.4-4.4 2.1Z"
          stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
    <path d="M3.5 15c-.6 1.6-.2 3 1 3.9M20.5 15c.6 1.6.2 3-1 3.9"
          stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""


def inject_css(t):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;450;500;600&display=swap');

    .stApp {{
        background:{t['appbg']};
        background-attachment:fixed;
        background-size:200% 200%;
    }}
    html, body, [class*="css"], .stMarkdown, p, span, label, div {{ font-family:'Inter',sans-serif; }}
    h1,h2,h3, .hero-title, .brand h1 {{ font-family:'Plus Jakarta Sans',sans-serif; }}
    #MainMenu, footer {{ visibility:hidden; }}
    header[data-testid="stHeader"] {{ background:transparent; }}
    [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {{ visibility:visible !important; }}
    [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="collapsedControl"] svg {{ fill:{t['navy']} !important; color:{t['navy']} !important; }}
    .block-container {{ max-width:860px; padding-top:1.4rem; }}

    .glass-panel {{
        background: rgba(255,255,255,0.14);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 24px;
        padding: 18px 20px 8px;
        margin-bottom: 14px;
        box-shadow: 0 8px 32px rgba(14,27,72,.18);
    }}

    /* ============ HEADER UTAMA — PERMANEN: Logo / Ruang Aman / Sub-judul ============ */
    .main-chat-header {{ text-align:center; margin:4px 0 18px; }}
    .main-chat-header .logo-badge {{
        width:56px; height:56px; margin:0 auto 12px; border-radius:16px;
        background:linear-gradient(135deg, {t['mauve']} 0%, {t['skyblue']} 100%);
        color:#fff; display:flex; align-items:center; justify-content:center;
        box-shadow:0 10px 24px rgba(193,141,180,.4);
    }}
    .main-chat-header .logo-badge svg {{ width:28px; height:28px; }}
    .main-chat-header .title {{
        font-family:'Plus Jakarta Sans', sans-serif;
        font-weight:800;
        font-size:30px;
        color:{t['header_title']};
        letter-spacing:-0.3px;
        margin-bottom:4px;
    }}
    .main-chat-header .subtitle {{
        font-family:'Inter', sans-serif;
        font-weight:400;
        font-size:15px;
        color:{t['header_sub']};
    }}
    
    /* ============ KARTU 6 FITUR — scoped ke container(key="fitur_grid") ============ */
    .st-key-fitur_grid [data-testid="column"] {{
        display: flex;
    }}
    .st-key-fitur_grid div.stButton {{
        width: 100%;
    }}
    .st-key-fitur_grid div.stButton > button {{
        height: 92px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        gap: 4px;
    }}
    .st-key-fitur_grid div.stButton > button p {{
        color:#FFFFFF !important;
        font-weight:700 !important;
        font-size: 13.5px !important;
        line-height: 1.35 !important;
    }}
    .st-key-f0 button, .st-key-f2 button, .st-key-f4 button {{
        background:linear-gradient(135deg, {t['navy']} 0%, {t['slate']} 100%) !important;
    }}
    .st-key-f1 button, .st-key-f3 button, .st-key-f5 button {{
        background:linear-gradient(135deg, {t['navy']} 0%, {t['mauve']} 100%) !important;
    }}
   
    /* ============ SAFETY NET —  ============ */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"] > div[style*="background"],
    .stAlert, div[data-testid="stNotification"],
    div[data-testid="stExpander"], div[data-testid="stExpanderDetails"] {{
        background: rgba(14,27,72,0.55) !important;
        backdrop-filter: blur(6px);
        color: #FFFFFF !important;
        border-color: rgba(255,255,255,0.15) !important;
    }}
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] li, div[data-testid="stExpander"] a {{
        color: #FFFFFF !important;
    }}
    /* di dalam sidebar tetap ikutin tema krem, override balik supaya nggak ikut jadi gelap */
    section[data-testid="stSidebar"] div[data-testid="stExpander"],
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] {{
        background: rgba(255,255,255,0.25) !important;
        color: {t['navy']} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stExpander"] p,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] span,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] li,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] a {{
        color: {t['navy']} !important;
    }}

    /* ============ CHAT BUBBLE — user di kanan, bot di kiri, seperti chat asli ============ */
    [data-testid="stChatMessage"] {{ background:transparent; border:none; padding:4px 0; gap:12px; }}
    [data-testid="stChatMessageContent"] {{
        max-width: 680px;
        border-radius:18px; padding:13px 18px;
        box-shadow:0 3px 10px rgba(14,27,72,.12);
    }}
    [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{
        font-size:14.8px; line-height:1.7;
    }}
    
    /* ============ AVATAR — target struktural, tidak bergantung nama testid ============ */
    [data-testid="stChatMessage"] > div:first-child,
    [data-testid="stChatMessage"] [data-testid*="Avatar" i],
    [data-testid="stChatMessage"] [data-testid*="avatar" i] {{
        background: linear-gradient(135deg, {t['mauve']} 0%, {t['active']} 100%) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }}
    [data-testid="stChatMessage"] > div:first-child *,
    [data-testid="stChatMessage"] [data-testid*="Avatar" i] *,
    [data-testid="stChatMessage"] [data-testid*="avatar" i] * {{
        background: transparent !important;
    }}

    /* USER = ganjil (pesan pertama): avatar & bubble ke KANAN */
    div[data-testid="stChatMessage"]:nth-of-type(odd) {{
        flex-direction: row-reverse;
        justify-content: flex-start;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] {{
        background:{t['user_bg']}; border:1.5px solid {t['user_border']};
        margin-left: auto;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] strong {{
        color:{t['user_text']} !important;
    }}

    /* BOT = genap: avatar & bubble tetap di KIRI */
    div[data-testid="stChatMessage"]:nth-of-type(even) {{
        flex-direction: row;
        justify-content: flex-start;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] {{
        background:{t['bot_bg']}; border:1.5px solid {t['bot_border']};
        margin-right: auto;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] strong {{
        color:{t['bot_text']} !important;
    }}

    /* fallback presisi lewat aria-label avatar, kalau browser dukung :has() */
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) {{
        flex-direction: row-reverse !important; justify-content: flex-start !important;
    }}
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] {{
        background:{t['user_bg']} !important; border:1.5px solid {t['user_border']} !important;
        margin-left: auto !important;
    }}
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] li {{
        color:{t['user_text']} !important;
    }}
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) {{
        flex-direction: row !important; justify-content: flex-start !important;
    }}
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] {{
        background:{t['bot_bg']} !important; border:1.5px solid {t['bot_border']} !important;
        margin-right: auto !important;
    }}
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] li {{
        color:{t['bot_text']} !important;
    }}

    /* ============ SIDEBAR — gradient krem ============ */
    section[data-testid="stSidebar"] {{
        background:linear-gradient(180deg, {t['sidebar_top']} 0%, {t['sidebar_bottom']} 100%);
        border-right:1px solid rgba(14,27,72,.08);
    }}
    section[data-testid="stSidebar"] * {{ color:{t['navy']}; }}

    .sidebar-logo-wrap {{ text-align:center; margin-top:6px; margin-bottom:10px; }}
    .sidebar-logo {{
        width:56px; height:56px; margin:0 auto; border-radius:50%;
        background:linear-gradient(135deg, {t['mauve']} 0%, {t['blush']} 100%); color:{t['navy']};
        display:flex; align-items:center; justify-content:center;
        box-shadow:0 6px 16px rgba(193,141,180,.4);
    }}
    .sidebar-logo svg {{ width:28px; height:28px; }}
    .sidebar-brand-title {{
        text-align:center; font-weight:700; font-size:21px; color:{t['navy']};
        margin:10px 0 2px; line-height:1.15;
    }}
    .sidebar-brand-sub {{
        text-align:center; font-style:italic; font-size:12px; color:{t['navy']};
        margin-bottom:14px; opacity:.85;
    }}
    
    section[data-testid="stSidebar"] div.stButton > button {{
        background:{t['navy']}; color:#fff !important; border:none; font-weight:600;
        border-radius:14px; padding:12px 14px; box-shadow:0 6px 14px rgba(14,27,72,.2);
    }}
    section[data-testid="stSidebar"] div.stButton > button:hover {{ filter:brightness(1.15); }}
    section[data-testid="stSidebar"] div.stButton > button p {{ color:#fff !important; }}

    /* ============ MENU ACTIVE / INACTIVE — pembeda jelas ============ */
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
        background:transparent !important; color:{t['navy']} !important; font-weight:700 !important;
        border:1.5px solid transparent !important; box-shadow:none !important; text-align:left !important;
        border-radius:14px !important;
    }}
    section[data-testid="stSidebar"] button[kind="secondary"] p,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {{
        color:{t['navy']} !important; font-weight:700 !important;
    }}
    section[data-testid="stSidebar"] button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
        background:rgba(241,145,109,.16) !important; border-color:rgba(241,145,109,.4) !important;
    }}

    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
        background:{t['active']} !important; color:#FFFFFF !important; font-weight:800 !important;
        border:3.5px solid {t['active']} !important; border-radius:14px !important; text-align:left !important;
        box-shadow:0 8px 18px rgba(241,145,109,.45) !important;
    }}
    section[data-testid="stSidebar"] button[kind="primary"] p,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {{
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) button {{
        background:{t['navy']} !important; color:#fff !important; box-shadow:0 6px 14px rgba(14,27,72,.2) !important;
        text-align:center !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) button p {{ color:#fff !important; }}

    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        font-weight:600; color:{t['navy']} !important; background:transparent !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button {{
        background:{t['navy']} !important; color:#fff !important; border:none !important;
        font-weight:600 !important; border-radius:14px !important; padding:12px 14px !important;
        box-shadow:0 6px 14px rgba(14,27,72,.2) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button p {{ color:#fff !important; }}

    /* ============ EXPANDER SIDEBAR (Pengaturan / Bantuan Langsung) — hilangkan bg putih bawaan ============ */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {{
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(14, 27, 72, 0.12) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
        padding: 12px 14px 22px 14px !important;
        background: transparent !important;
    }}
    .emergency-card {{
        display: block;
        color: {t['navy']};
        text-align: left;
    }}
    .emergency-title {{
        font-size: 16px;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: {t['navy']};
        line-height: 1.2;
    }}
    .emergency-subtitle {{
        font-size: 12px;
        font-weight: 600;
        opacity: 0.8;
        margin-bottom: 10px;
    }}
    .emergency-desc {{
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 12px;
        opacity: 0.85;
    }}
    .emergency-btn {{
        display: block !important;
        text-align: center !important;
        background: #25D366 !important; /* Warna hijau khas WhatsApp agar intuitif */
        color: #FFFFFF !important;
        text-decoration: none !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(37, 211, 102, 0.2) !important;
        transition: all 0.2s ease !important;
    }}
    .emergency-btn:hover {{
        background: #128C7E !important;
        box-shadow: 0 6px 16px rgba(37, 211, 102, 0.35) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
        background: rgba(241,145,109,.12) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stExpander"] p,
    section[data-testid="stSidebar"] [data-testid="stExpander"] a,
    section[data-testid="stSidebar"] [data-testid="stExpander"] li {{
        color:{t['navy']} !important;
    }}

    /* ============ KARTU INFORMASI PENGATURAN (KELUAR CEPAT) ============ */
    .settings-info-card {{
        background: rgba(216, 52, 42, 0.08) !important; /* Warna merah transparan tipis */
        border-left: 3.5px solid #D8342A !important; /* Aksen garis merah tegas di kiri */
        border-radius: 10px !important;
        padding: 10px 12px !important;
        margin-bottom: 16px !important;
        text-align: left !important;
    }}
    .settings-info-title {{
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: {t['navy']} !important;
        margin-bottom: 4px !important;
    }}
    .settings-info-desc {{
        font-size: 11.5px !important;
        line-height: 1.5 !important;
        color: {t['navy']} !important;
        opacity: 0.85 !important;
    }}
    
    /* ============ MERAPIKAN WIDGET RADIO BUTTON SIDEBAR ============ */
    /* Merapikan label utama "Ukuran teks" */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {{
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: {t['navy']} !important;
        margin-bottom: 8px !important;
    }}
    
    /* Memberikan jarak renggang yang proporsional antar opsi pilihan */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] + div {{
        gap: 16px !important;
    }}
    
    /* Mengatur teks opsi (Kecil, Sedang, Besar) agar lebih tegas */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {{
        font-size: 13px !important;
        font-weight: 500 !important;
        color: {t['navy']} !important;
    }}

    /* ============ SUGGESTION CHIPS — pakai container(key=) yang beneran nge-wrap ============ */
    .st-key-chip_row div.stButton > button {{
        background:rgba(255,255,255,0.55) !important; color:{t['navy']} !important;
        border:1.5px solid {t['skyblue']} !important; border-radius:999px !important;
        padding:6px 16px !important; font-size:13px !important; font-weight:600 !important;
        box-shadow:none !important; min-height:auto !important;
    }}
    .st-key-chip_row div.stButton > button p {{ color:{t['navy']} !important; font-size:13px !important; }}
    .st-key-chip_row div.stButton > button:hover {{
        background:{t['mauve']} !important; color:#fff !important; border-color:{t['mauve']} !important;
    }}
    .st-key-chip_row div.stButton > button:hover p {{ color:#fff !important; }}

   /* ============ INPUT CHAT - OBLITERASI TOTAL KOTAK & HOVER MERAH ============ */
    [data-testid="stChatInput"] {{
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1.5px solid {t['skyblue']} !important;
        border-radius: 40px !important;
        box-shadow: 0 4px 14px rgba(14, 27, 72, 0.15) !important;
    }}
    
    [data-testid="stChatInput"] * {{
        background-color: transparent !important;
    }}
    
    [data-testid="stBottom"] {{
        background: {t['deep']} !important;
    }}
    
    div[data-testid="stBottomBlockContainer"] {{
        max-width: 100% !important;
        padding-left: 30px !important;
        padding-right: 30px !important;
        background: {t['deep']} !important;
    }}
    
    div[data-testid="stBottom"] > div {{
        background: transparent !important;
    }}
    
    [data-testid="stChatInput"] textarea {{
        color: {t['navy']} !important;
        -webkit-text-fill-color: {t['navy']} !important;
        caret-color: {t['navy']} !important;
        font-size: 15px !important;
    }}
    
    [data-testid="stChatInput"] textarea::placeholder {{
        color: #9aa3b5 !important;
        -webkit-text-fill-color: #9aa3b5 !important;
        opacity: 1 !important;
    }}
    
    [data-testid="stChatInput"]:focus-within {{
        border-color: {t['active']} !important;
        box-shadow: 0 0 0 3px rgba(241, 145, 109, 0.2) !important;
    }}
    
    /* ============ MIC DOCK — sejajar persis dgn tombol kirim, tumbuh ke kiri saat merekam ============ */
    .st-key-mic_dock {{
        position: fixed !important;
        right: 90px !important;
        bottom: 70px !important;
        z-index: 999999 !important;
        width: auto !important;
        display: block !important;
    }}
    .st-key-mic_dock > div {{
        width: auto !important;
    }}
    /* Default (belum ada rekaman): bulat kecil, cuma ikon mic */
    .st-key-mic_dock [data-testid="stAudioInput"] {{
        background: {t['mauve']} !important;
        border-radius: 999px !important;
        height: 32px !important;
        min-height: 32px !important;
        min-width: 32px !important;
        width: auto !important;
        max-width: 220px !important;
        padding: 0 8px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 12px rgba(14,27,72,.25) !important;
    }}
    /* Setelah ada rekaman (widget berisi elemen <audio>): otomatis melebar ke kiri (karena anchor 'right' tetap) */
    .st-key-mic_dock [data-testid="stAudioInput"]:has(audio) {{
        padding: 0 12px !important;
    }}
    .st-key-mic_dock [data-testid="stAudioInput"] * {{
        background: transparent !important;
        white-space: nowrap !important;
    }}
    .st-key-mic_dock [data-testid="stAudioInput"] button {{
        color: #FFFFFF !important;
        flex-shrink: 0 !important;
    }}
    .st-key-mic_dock [data-testid="stAudioInput"] span,
    .st-key-mic_dock [data-testid="stAudioInput"] p {{
        color: #FFFFFF !important;
        font-size: 11px !important;
    }}
    .st-key-mic_dock [data-testid="stAudioInput"] {{
    background: {t['mauve']} !important;
    border-radius: 999px !important;
    height: 32px !important;
    min-height: 32px !important;
    min-width: 32px !important;
    width: auto !important;
    max-width: 220px !important;
    padding: 0 8px !important;
    display: flex !important;
    flex-direction: row-reverse !important;   /* <-- baris baru: kunci ikon di ujung kanan */
    justify-content: flex-start !important;   /* <-- baris baru: rapatkan grup ke titik anchor */
    align-items: center !important;
    gap: 6px !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    box-shadow: 0 4px 12px rgba(14,27,72,.25) !important;
    }}
    
    /* 🔥 TOMBOL INDUK: Kunci bentuk lingkaran sempurna & ukuran presisi 🔥 */
    [data-testid="stChatInput"] button,
    [data-testid="stChatInputSubmitButton"] {{
        background-color: {t['mauve']} !important;
        border-radius: 50% !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 6px !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
    }}
    
    /* 🔥 KUNCI UTAMA (HOVER & FOCUS PROTECTION): Mencegah tombol berubah jadi blok merah 🔥 */
    [data-testid="stChatInput"] button:hover,
    [data-testid="stChatInputSubmitButton"]:hover,
    [data-testid="stChatInput"] button:active,
    [data-testid="stChatInput"] button:focus,
    [data-testid="stChatInput"] button:focus-visible {{
        background-color: {t['mauve']} !important; /* Paksa tetap berwarna mauve */
        filter: brightness(0.88) !important; /* Memberikan efek hover gelap yang elegan & smooth, BUKAN merah */
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    
    /* 🔥 ANTI-KOTAK PUTIH: Hancurkan paksa sisa garis kotak di elemen dalam saat hover/focus 🔥 */
    [data-testid="stChatInput"] button *,
    [data-testid="stChatInputSubmitButton"] *,
    [data-testid="stChatInput"] button div,
    [data-testid="stChatInput"] button span {{
        border: none !important;
        border-width: 0px !important;
        outline: none !important;
        outline-style: none !important;
        outline-width: 0px !important;
        box-shadow: none !important;
        background: transparent !important;
        background-color: transparent !important;
    }}
    
    /* Singkirkan sisa rect grafis pembungkus bawaan SVG jika ada */
    [data-testid="stChatInput"] button svg rect,
    [data-testid="stChatInputSubmitButton"] svg rect {{
        display: none !important;
        stroke: transparent !important;
        fill: transparent !important;
    }}
    
    /* 🔥 PROTEKSI PANAH UTAMA: Memastikan ikon panah tetap menyala putih bersih & proporsional 🔥 */
    [data-testid="stChatInput"] button svg,
    [data-testid="stChatInputSubmitButton"] svg {{
        display: inline-block !important;
        color: #FFFFFF !important;
        stroke: #FFFFFF !important;
        stroke-width: 2.5px !important;
        fill: none !important;
        width: 18px !important;
        height: 18px !important;
        border: none !important;
        outline: none !important;
    }}

    .disclaimer {{ font-size:11px; line-height:1.55; color:{t['slate']} !important; margin-top:14px;
        border-top:1px solid rgba(14,27,72,.12); padding-top:12px; }}
    .key-ok {{ font-size:12px; color:{t['active']}; font-weight:700; padding:4px 0; }}
    .support-banner {{
        background: rgba(255,255,255,0.6);
        border-left: 4px solid {t['active']};
        border-radius: 12px;
        padding: 12px 16px;
        margin: 6px 0 14px 0;
        font-size: 13.5px;
        color: {t['navy']};
        line-height: 1.6;
    }}
    </style>""", unsafe_allow_html=True)

FITUR = [
    ("\U0001f91d Konseling Kasus", "Cerita situasimu, dapat arahan empatik",
     "Saya mengalami situasi yang mungkin termasuk kekerasan seksual. Bisa bantu saya memahami apa yang terjadi?"),
    ("\U0001f4d1 Jenis & Pasal", "Jenis TPKS dan dasar hukumnya",
     "Apa saja jenis tindak pidana kekerasan seksual dalam UU TPKS?"),
    ("\u2696\ufe0f Ancaman Pidana", "Sanksi penjara & denda",
     "Berapa ancaman pidana untuk pelecehan seksual fisik dan eksploitasi seksual?"),
    ("\U0001f6e1\ufe0f Hak Korban", "Restitusi & pemulihan",
     "Apa saja hak korban dan bagaimana mekanisme restitusi menurut UU TPKS?"),
    ("\U0001f4cb Alur Melapor", "Langkah lapor & pembuktian",
     "Bagaimana cara melapor kasus kekerasan seksual dan alat bukti apa yang diakui?"),
    ("\U0001f4de Bantuan Darurat", "Kontak lembaga layanan",
     "Saya butuh nomor dan kontak lembaga bantuan untuk korban kekerasan seksual."),
]

MENU_ITEMS = [
    ("konseling", "\U0001f91d", "Konseling"),
    ("pasal", "\u2696\ufe0f", "Telusur Pasal"),
    ("lapor", "\U0001f4cb", "Panduan Lapor"),
]
MENU_LABELS = {"konseling": "Konseling", "pasal": "Telusur Pasal", "lapor": "Panduan Lapor"}

THRESHOLD = 0.5
def retrieve(query, k=4, th=THRESHOLD):
    m = re.search(r"pasal\s+(\d{1,3})\b", query, re.IGNORECASE)
    exact_ctx = []
    if m:
        num = m.group(1)
        pat = re.compile(rf"(?:^|\]\s*)Pasal\s+{num}\b")
        exact_ctx = [c for c in chunks if pat.search(c)]

    q = embed_model.encode([query]); faiss.normalize_L2(q)
    D, I = index.search(np.array(q).astype("float32"), k)
    sem_ctx = [chunks[i] for i, s in zip(I[0], D[0]) if s >= th]

    ctx = exact_ctx + [c for c in sem_ctx if c not in exact_ctx]
    sim = 1.0 if exact_ctx else float(D[0][0])
    return ctx, sim

BASE = """Kamu adalah "Pasal", asisten hukum berbahasa Indonesia yang HANYA membahas
UU No. 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS).

PRINSIP WAJIB:
- SEBELUM menjawab, cek dulu: apakah pertanyaan user ADA HUBUNGANNYA dengan kekerasan
  seksual, TPKS, atau isi UU ini? Kalau SAMA SEKALI TIDAK NYAMBUNG (tips skripsi, resep
  masakan, coding, dll), JANGAN dipaksa dikait-kaitkan ke pasal apapun. Jawab singkat
  bahwa kamu cuma fokus bahas UU TPKS.
- Jawab HANYA dari materi pasal yang tersedia. Dilarang mengarang pasal/angka. Kalau
  jawabannya nggak ada, bilang terus terang lalu arahkan ke bantuan resmi.
- Sebut nomor pasal HANYA kalau ISINYA SPESIFIK ke situasi (jenis kekerasan tertentu,
  hak korban tertentu, sanksi tertentu). Pasal definisi umum/pembukaan BUKAN dasar kuat —
  jangan dipaksa disebut. Kalau nggak ada yang pas, jangan sebut pasal sama sekali.
- DILARANG MUTLAK menyebut kata "konteks", "kutipan", atau "yang diberikan/disediakan/
  tersedia" dalam bentuk apapun. User nggak tahu ada proses retrieval di baliknya. Kalau
  pasal nggak ada yang pas, LEWATI SAJA tanpa bilang "tidak ada info" — fokus ke dukungan/
  panduannya aja.
- Sapaan WAJIB konsisten "kamu" dari awal sampai akhir. JANGAN PERNAH pakai "Anda".
- HINDARI kata "saya"/"aku" sama sekali saat chatbot merujuk ke dirinya sendiri. Tulis
  ulang kalimatnya biar nggak butuh kata ganti orang pertama.
- DILARANG memulai kalimat pertama jawaban dengan kata "Kamu"/"Anda", dan DILARANG kalimat
  pertama berupa rangkuman/label ulang atas cerita user dalam bentuk apapun (co: "Percakapan
  yang kamu alami itu terdengar tidak nyaman..."). Langsung ke insight/reaksi/info baru.
  Di paragraf manapun, maksimal 1 kalimat yang diawali "Kamu" — kalimat lain pakai struktur
  beda (kata kerja, situasi, atau klausa "Kalau...", "Karena...").
- DILARANG mengulang parafrase situasi user yang SUDAH disebut di giliran sebelumnya.
  Anggap itu udah established, lanjut ke hal baru.
- MAKSIMAL 1 tanda tanya per jawaban. Kalau nggak ada yang perlu ditanya, tutup dengan
  pernyataan/langkah konkret tanpa tanda tanya.
- Tulis dengan bahasa manusia yang mengalir, natural, dan BERVARIASI tiap respons (struktur/
  opening/closing, bukan cuma variasi kata). DILARANG KERAS kalimat pembuka klise: "Maaf
  mendengar...", "Terima kasih sudah berbagi...".
- Jangan menjejalkan kontak SAPA 129 di setiap jawaban; sebut hanya bila relevan.
- Kalimat pertama jawaban HARUS langsung berisi salah satu dari: (a) informasi/insight baru
  yang belum disebut user, (b) pertanyaan balik jika benar-benar perlu, atau (c) langkah/opsi
  konkret. DILARANG kalimat pertama berupa PARAFRASE situasi user dalam bentuk apapun, termasuk
  yang berbunyi "Kalau [situasi]...", "Posisi/Keadaan/Situasi [X] itu...", "Kamu sudah/sedang...".
  Contoh BENAR: "Menolak permintaan itu adalah hakmu, dan penolakan itu sendiri sudah cukup —
  nggak perlu alasan tambahan." Contoh SALAH: "Kalau pacarmu meminta hal itu, itu bisa membuatmu
  tidak nyaman." (ini parafrase, dilarang)"""

PROMPTS = {
"konseling": BASE + '''

PERAN SEKARANG: KONSELOR (bukan customer service, bukan legal-bot).
Baca cerita orangnya dulu, reaksi dengan cara yang nunjukin kamu beneran nangkep detailnya
(sebut ulang elemen spesifik dari ceritanya dengan kata-katamu sendiri, bukan parafrase kaku).
Validasi perasaannya tanpa menggurui. Kalau ada pasal yang BENAR-BENAR pas dengan situasinya,
selipkan natural di tengah kalimat (bukan sebagai poin terpisah/dokumentatif). Kalau nggak ada
pasal yang pas, itu OK — nggak usah dipaksa nyebut pasal sama sekali di respons ini.
Tutup dengan sesuatu yang konkret: satu langkah kecil yang relevan buat situasi dia, ATAU
ajakan buat cerita lebih lanjut yang terasa personal (bukan template).
Nada: seperti teman yang paham hukum, bukan seperti membacakan pasal. Maksimal 4 paragraf pendek,
variasikan panjang & struktur kalimat supaya nggak kerasa template.

JAGA KESELAMATAN EMOSIONAL — INI PRIORITAS DI ATAS INFORMASI HUKUM:
- Jangan pernah membuat user merasa lebih bersalah, lebih takut, atau lebih terpojok dari
  sebelum dia curhat. Kalau ragu antara jawaban yang "lengkap secara hukum" vs "aman secara
  emosional", pilih yang aman secara emosional.
- Jangan memaksa/mendesak user buat lapor, konfrontasi pelaku, atau ambil tindakan tertentu.
  Tawarkan opsi, bukan instruksi. Hormati kalau dia belum siap atau belum mau bertindak.
- Kalau user menunjukkan tanda distress berat (putus asa, menyalahkan diri berlebihan,
  menyebut ingin menyakiti diri), JANGAN lanjut bahas pasal/hukum dulu — fokus ke stabilisasi
  emosinya dan arahkan ke bantuan profesional/hotline dengan tenang, bukan dengan panik.
- Jangan membombardir dengan banyak istilah hukum sekaligus kalau user kelihatan rapuh —
  cukup satu poin paling penting per respons, sisanya bisa nunggu giliran berikutnya.''',

"pasal": BASE + '''

PERAN SEKARANG: PENELUSUR PASAL.
Jawab lugas dan informatif seperti referensi hukum. Sebutkan pasal + isi pokoknya +
ancaman pidana (penjara/denda) bila ada. Boleh pakai poin bernomor agar rapi.
Minim basa-basi empati; langsung ke substansi hukum. Sebut nomor pasal dengan tepat.''',

"lapor": BASE + '''

PERAN SEKARANG: PEMANDU PELAPORAN.
Beri panduan PRAKTIS dan berurutan: ke mana melapor (UPTD PPA, Unit PPA Polisi),
bukti/dokumen yang perlu disiapkan, hak korban selama proses, dan apa yang terjadi
setelah lapor. Susun sebagai langkah 1-2-3 yang mudah diikuti. Rujuk pasal terkait
(mis. pelaporan, alat bukti, perlindungan). Akhiri dengan kontak resmi bila relevan.''',
}


def groq_answer(api_key, user_input, history, mode, support_info):
    ctx_list, sim = retrieve(user_input)
    context = "\n\n".join(f"[Kutipan {i+1}] {c}" for i, c in enumerate(ctx_list)) if ctx_list \
              else "(Tidak ada pasal relevan di atas ambang. Sampaikan jujur & arahkan ke bantuan resmi.)"
    system_prompt = PROMPTS[mode]

    if support_info["perlu_rujukan"]:
        system_prompt += (
            "\n\nCATATAN TAMBAHAN: User tampak sedang dalam kondisi tertekan/sedih. "
            "Jawab dengan nada empatik dan hati-hati, dan sisipkan secara halus "
            "bahwa ada layanan pendampingan seperti SAPA 129 yang bisa dihubungi "
            "kalau butuh bantuan lebih lanjut. Jangan terkesan menghakimi atau memberi diagnosis."
        )

    user_msg = f"PERTANYAAN PENGGUNA:\n{user_input}\n\nKONTEKS PASAL UU TPKS (relevansi {sim:.0%}):\n{context}"
    # ... sisanya (msgs, groq_keys, stream, dst) TETAP SAMA, gak perlu diubah
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_msg})

    groq_keys = [k for k in [api_key, os.environ.get("GROQ_API_KEY_2", "")] if k]
    for idx, key in enumerate(groq_keys):
        try:
            client = Groq(api_key=key)
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs, temperature=0.75, max_tokens=900, stream=True)
            got_chunk = False
            for chunk in stream:
                d = chunk.choices[0].delta.content
                if d:
                    got_chunk = True
                    yield d
            if got_chunk:
                return
        except Exception as e:
            print(f"[GROQ key #{idx+1} GAGAL] {type(e).__name__}: {e}", flush=True)
            continue

    yield "\u26a0\ufe0f Groq sedang limit. Coba lagi beberapa saat lagi.\n\nKalau mendesak, hubungi **SAPA 129**."

def _pdf_sanitize(text):
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...",
        "═": "=", "─": "-", "\U0001f49b": "", "\U0001f464": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def build_transcript_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, _pdf_sanitize("RUANG AMAN - RINGKASAN PERCAKAPAN"), align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    meta = [
        f"Sesi dimulai   : {st.session_state.session_started.strftime('%d %B %Y, %H:%M:%S')}",
        f"Diunduh pada   : {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
        f"Mode konseling : {MENU_LABELS.get(st.session_state.active_menu, '-')}",
        f"Jumlah pesan   : {len(st.session_state.messages)}",
    ]
    for line in meta:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _pdf_sanitize(line))

    pdf.ln(3)
    pdf.set_draw_color(150, 150, 150)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    for m in st.session_state.messages:
        who = "USER" if m["role"] == "user" else "RUANG AMAN"
        ts = m.get("time")
        ts_str = ts.strftime("%H:%M:%S") if ts else "-"

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _pdf_sanitize(f"[{ts_str}] {who}:"))

        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _pdf_sanitize(m["content"]))
        pdf.ln(2)

    pdf.set_draw_color(150, 150, 150)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    for line in [
        "Dokumen ini dibuat otomatis oleh Ruang Aman.",
        "Bukan pengganti dokumen resmi kepolisian/lembaga hukum.",
        "Darurat? Hubungi SAPA 129.",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, _pdf_sanitize(line))

    return bytes(pdf.output())

for k, v in [("messages", []), ("active_menu", "konseling"), ("pending", None),
             ("last_audio_hash", None), ("text_size", "Sedang")]:
    if k not in st.session_state: st.session_state[k] = v
if "session_started" not in st.session_state:
    st.session_state.session_started = datetime.now()


inject_css(T)

FONT_SIZES = {
    "Kecil":  {"chat": "13px",   "input": "13px"},
    "Sedang": {"chat": "14.8px", "input": "15px"},
    "Besar":  {"chat": "18px",   "input": "18px"},
}

def inject_text_size_css(size_label):
    s = FONT_SIZES.get(size_label, FONT_SIZES["Sedang"])
    st.markdown(f"""
    <style>
    [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{
        font-size: {s['chat']} !important;
        line-height: 1.75 !important;
    }}
    [data-testid="stChatInput"] textarea {{
        font-size: {s['input']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ===================== PANIC EXIT — tombol darurat, fixed, selalu terlihat =====================
st.markdown(f"""
<style>
.panic-exit-container {{
    position: fixed;
    top: 12px;
    right: 20px;
    z-index: 9999999;
}}
.panic-exit-btn {{
    background: #D8342A !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    box-shadow: 0 6px 16px rgba(216, 52, 42, .5) !important;
    cursor: pointer !important;
    display: inline-block !important;
    line-height: 1.5 !important;
    transition: background 0.2s ease;
}}
.panic-exit-btn:hover {{
    background: #B92A21 !important;
}}
</style>
<div class="panic-exit-container">
    <button onclick="window.top.location.replace('https://www.google.com');" class="panic-exit-btn">🚨 Keluar Cepat</button>
</div>
""", unsafe_allow_html=True)
# ===================== SIDEBAR =====================
mode_key = st.session_state.active_menu
inject_text_size_css(st.session_state.text_size)

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-logo-wrap"><div class="sidebar-logo">{LOGO_SVG}</div></div>'
        '<div class="sidebar-brand-title">Ruang Aman</div>'
        '<div class="sidebar-brand-sub">Privasi terjaga, Ceritamu berharga!</div>',
        unsafe_allow_html=True)

    if st.button("\u2795  Konsultasi baru", use_container_width=True):
        st.session_state.messages = []; st.session_state.pending = None; st.rerun()

    if st.session_state.messages:
        st.download_button(
            "💾  Simpan Percakapan (PDF)",
            data=build_transcript_pdf(),
            file_name=f"ruang-aman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("\U0001f511 Groq API Key", type="password")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    for key, emoji, label in MENU_ITEMS:
        is_active = st.session_state.active_menu == key
        if st.button(f"{emoji}  {label}", key=f"menu_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_menu = key
            st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tampilkan_emosi = st.checkbox(
        "\U0001f4ad Tampilkan mode percakapan",
        value=False,
        help="Menampilkan bagaimana Ruang Aman menyesuaikan gaya responsnya berdasarkan konteks percakapan."
    )
    
    if tampilkan_emosi:
        last = st.session_state.get("last_emotion_result")
        
        # Emoji tetap kita petakan secara statis agar ikonnya sesuai emosi
        emoji_map = {"sadness": "🫂", "fear": "🏡", "anger": "🍃", "happy": "🥰", "love": "💖"}
        current_emoji = emoji_map.get(last["label_dominan"], "💬") if last else "💬"
        
        # Mengambil kalimat dinamis hasil generate AI (jika belum ada pesan, pakai default)
        current_label = last["ai_label"] if last else "Siap membantu proses konselingmu."
        
        st.markdown(f"""
            <div style="padding:12px; border-radius:14px; background-color:rgba(255,255,255,0.35); text-align:center;">
                <div style="font-size:28px;">{current_emoji}</div>
                <div style="font-size:13px; margin-top:4px; color:{T['navy']}; font-weight:500;">{current_label}</div>
            </div>""", unsafe_allow_html=True)
        st.caption("⚠️ Estimasi otomatis berdasarkan pesan terakhirmu.")
    
    with st.expander("⚙️  Pengaturan"):
        st.markdown("""
            <div class="settings-info-card">
                <div class="settings-info-title">🚨 Keluar Cepat</div>
                <div class="settings-info-desc">
                    Tombol merah di pojok kanan atas akan langsung menghapus seluruh riwayat obrolan secara permanen dan mengalihkan browser ke Google demi menjaga privasimu.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Widget bawaan tetap berfungsi penuh dan otomatis ter-style oleh CSS baru
        size_choice = st.radio(
            "🔠 Ukuran teks",
            options=["Kecil", "Sedang", "Besar"],
            index=["Kecil", "Sedang", "Besar"].index(st.session_state.text_size),
            horizontal=True,
            key="text_size_radio",
        )
        st.session_state.text_size = size_choice

    with st.expander("📞  Bantuan Langsung"):
        st.markdown("""
            <div class="emergency-card">
                <div class="emergency-title">SAPA 129</div>
                <div class="emergency-subtitle">Hotline Kekerasan Seksual</div>
                <div class="emergency-desc">
                    Layanan pendampingan resmi yang tersedia 24 jam melalui panggilan telepon langsung atau chat WhatsApp.
                </div>
                <a href="https://wa.me/628111129129" target="_blank" class="emergency-btn">
                    💬 Hubungi WhatsApp
                </a>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">Ruang Aman memberi informasi berbasis UU No. 12 Tahun 2022. '
                'Bukan pengganti advokat/lembaga resmi. Darurat? <b>SAPA 129</b>.</div>', unsafe_allow_html=True)

mode_key = st.session_state.active_menu

# ===================== HEADER UTAMA — PERMANEN: Logo / Ruang Aman / Sub-judul =====================
st.markdown(f"""
    <div class="main-chat-header">
        <div class="logo-badge">{LOGO_SVG}</div>
        <div class="title">Ruang Aman</div>
        <div class="subtitle">Kamu tidak sendirian. Kami mendengarkan.</div>
    </div>
""", unsafe_allow_html=True)

# ===================== 6 FITUR (hanya saat belum ada percakapan) =====================
if not st.session_state.messages:
    fitur_container = st.container(key="fitur_grid")
    with fitur_container:
        cols = st.columns(3)
        for i, (j, d, p) in enumerate(FITUR):
            with cols[i % 3]:
                if st.button(f"{j}\n\n{d}", key=f"f{i}", use_container_width=True):
                    st.session_state.pending = p; st.rerun()

if st.session_state.messages:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="\U0001f49b" if m["role"] == "assistant" else "\U0001f464"):
            st.markdown(m["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== SUGGESTION CHIPS =====================
chip_container = st.container(key="chip_row")
with chip_container:
    chip_cols = st.columns(3)
    CHIPS = [
        ("Bagaimana cara lapor?", "Bagaimana cara melapor kasus kekerasan seksual?"),
        ("UU TPKS", "Apa saja jenis tindak pidana kekerasan seksual dalam UU TPKS?"),
        ("Butuh psikolog", "Saya butuh pendampingan psikologis, ke mana saya bisa mencari bantuan?"),
    ]
    for i, (label, prompt) in enumerate(CHIPS):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending = prompt; st.rerun()

# ===================== INPUT SUARA =====================
if "audio_widget_key" not in st.session_state:
    st.session_state.audio_widget_key = 0

mic_dock = st.container(key="mic_dock")
with mic_dock:
    audio_value = st.audio_input(
        "Rekam suara", label_visibility="collapsed",
        key=f"audio_input_{st.session_state.audio_widget_key}",
    )
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hash(audio_bytes)
        if st.session_state.get("last_audio_hash") != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            if not api_key:
                st.warning("Groq API Key belum ada, tidak bisa transkripsi suara.")
            else:
                with st.spinner(""):
                    try:
                        client = Groq(api_key=api_key)
                        transcript = client.audio.transcriptions.create(
                            file=("rekaman.wav", audio_bytes),
                            model="whisper-large-v3",
                            language="id",
                        )
                        st.session_state.pending = transcript.text
                        st.session_state.audio_widget_key += 1  # widget reset -> rekaman lama hilang
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal mentranskripsi suara: {e}")

user_input = st.chat_input("Tuliskan pesanmu di sini...")
if st.session_state.pending and not user_input:
    user_input = st.session_state.pending; st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now()})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 1. Deteksi emosi dasar dari model lokal
    support_info = get_support_flag(user_input)
    
    # 2. Panggil AI untuk membuat kalimat support dinamis yang unik
    ai_dynamic_label = generate_ai_support_label(api_key, support_info["emosi"], user_input)
    
    # 3. Simpan hasilnya ke session state (termasuk kalimat dari AI tadi)
    st.session_state["last_emotion_result"] = {
        "label_dominan": support_info["emosi"],
        "confidence": support_info["confidence"],
        "ai_label": ai_dynamic_label  # <-- Disimpan di sini
    }

    # ... (sisa kode st.chat_message("assistant") untuk streaming jawaban hukum tetap sama)

    with st.chat_message("assistant", avatar="\U0001f49b"):
        banner_text = None
        if support_info["perlu_rujukan"]:
            banner_text = get_support_banner(support_info["emosi"])
        if banner_text:
            st.markdown(f'<div class="support-banner">{banner_text}</div>', unsafe_allow_html=True)

        if not api_key:
            ans = "⚠️ Groq API Key belum ada. Set Variable **GROQ_API_KEY** di dashboard Railway.\n\nDarurat? **SAPA 129**."
            st.markdown(ans)
        else:
            try:
                ans = st.write_stream(groq_answer(api_key, user_input, st.session_state.messages[:-1], mode_key, support_info))
            except Exception as e:
                ans = f"Maaf, ada kendala memanggil LLM: `{e}`\n\nKalau mendesak, hubungi **SAPA 129**."
                st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans, "time": datetime.now()})
    st.rerun()
