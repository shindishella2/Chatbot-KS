import random
import re
import time, faiss, numpy as np, pickle, os
from datetime import datetime
import streamlit as st
from streamlit.components.v1 import html as components_html
import json
from google import genai
from google.genai import types
import psutil

def print_ram(stage):
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024 / 1024
    print(f"[RAM] {stage}: {ram:.1f} MB")

st.set_page_config(page_title="Ruang Aman - Konseling Hukum UU TPKS",
                   page_icon="\U0001f49b", layout="wide",
                   initial_sidebar_state="expanded")

@st.cache_resource
def load_embed():
    print_ram("Sebelum load embedding")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

    print_ram("Sesudah load embedding")

    return model
@st.cache_resource
@st.cache_resource
def load_store():
    print_ram("Sebelum load faiss")

    index = faiss.read_index("faiss_index.index")
    chunks = pickle.load(open("chunks.pkl","rb"))

    print_ram("Sesudah load faiss")

    return index,chunks

FALLBACK_SUPPORT_SENTENCE = {
    "sadness": "Aku di sini mendengarkanmu. Ceritakan saja semuanya, ya.",
    "fear": "Kamu aman di sini. Tarik napas dalam-dalam, kita lalui bersama.",
    "anger": "Wajar kok kalau kamu kesal. Yuk, rehat sejenak dan tenangin pikiran.",
    "happy": "Ikut senang mendengarnya! Cerita seru apa lagi nih?",
    "love": "Terima kasih ya sudah berbagi energi positif. Kamu berharga!",
}

def analyze_emotion_and_support(text: str, api_key: str) -> dict:
    """1 Gemini call yang sekaligus: (1) klasifikasi emosi 5 kelas, dan
    (2) bikin 1 kalimat support dinamis berdasarkan emosi dominan itu sendiri.
    Menggantikan 2 call terpisah (detect_emotion + generate_ai_support_label)."""
    labels = ["sadness", "fear", "anger", "happy", "love"]
    default_scores = {l: 0.0 for l in labels}
    default_fallback = "Siap membantu dan mendengarkan ceritamu."
    if not api_key:
        return {"label_dominan": "sadness", "confidence": 0.0, "semua_skor": default_scores, "ai_label": default_fallback}
    system_prompt = (
        "Kamu adalah asisten yang menganalisis emosi teks Bahasa Indonesia SEKALIGUS "
        "membuat 1 kalimat respons suportif singkat, dalam satu langkah.\n\n"
        "LANGKAH 1 - KLASIFIKASI EMOSI:\n"
        "Klasifikasikan pesan user ke 5 emosi: sadness, fear, anger, happy, love. "
        "Beri skor keyakinan 0-1 untuk masing-masing (total sekitar 1.0).\n\n"
        "LANGKAH 2 - KALIMAT SUPPORT (berdasarkan emosi dominan hasil Langkah 1):\n"
        "- sadness: 1 kalimat sangat lembut, empati mendalam, tunjukkan kamu ada untuknya.\n"
        "- fear: 1 kalimat menenangkan, berikan kepastian dia aman bercerita di sini.\n"
        "- anger: 1 kalimat validasi yang adem, turunkan tensi tanpa menghakimi.\n"
        "- happy: 1 kalimat ikut senang, antusias, apresiasi energi positifnya.\n"
        "- love: 1 kalimat apresiasi hangat atas kasih sayang/cerita indahnya.\n\n"
        "ATURAN KALIMAT SUPPORT:\n"
        "- Maksimal 12-15 kata.\n"
        "- Bahasa Indonesia santai/kasual, natural seperti teman dekat (jangan kaku/formal).\n"
        "- DILARANG pakai tanda kutip atau kalimat pengantar seperti 'Ini kalimatnya:'.\n\n"
        "Balas HANYA dengan JSON valid tanpa teks/markdown lain, format PERSIS:\n"
        '{"sadness": 0.0, "fear": 0.0, "anger": 0.0, "happy": 0.0, "love": 0.0, "support_sentence": "..."}'
    )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=150,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        scores = {l: float(data.get(l, 0.0)) for l in labels}
        dominant = max(scores, key=scores.get)
        support_sentence = str(data.get("support_sentence", "")).strip()
        if not support_sentence:
            support_sentence = FALLBACK_SUPPORT_SENTENCE.get(dominant, default_fallback)
        return {"label_dominan": dominant, "confidence": scores[dominant], "semua_skor": scores, "ai_label": support_sentence}
    except Exception:
        return {"label_dominan": "sadness", "confidence": 0.0, "semua_skor": default_scores,
                 "ai_label": FALLBACK_SUPPORT_SENTENCE["sadness"]}


DISTRESS_MAP = {
    "sadness": "tinggi",
    "fear": "tinggi",
    "anger": "sedang",
    "happy": "rendah",
    "love": "rendah",
}

def get_support_flag(text: str, api_key: str, threshold: float = 0.5, sadness_safety_threshold: float = 0.30) -> dict:
    r = analyze_emotion_and_support(text, api_key)
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
        "ai_label": r["ai_label"],
    }


TYPING_INDICATOR_HTML = """
<div class="typing-indicator"><span></span><span></span><span></span></div>
"""


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

    /* ============================================================ */
    /* INPUT CHAT — dibangun ulang total (versi lama punya bug:      */
    /* kotak persegi mengintip di belakang tombol bulat, dan cursor  */
    /* berubah jadi ikon "dilarang" / lingkaran merah saat hover).   */
    /* ============================================================ */
    [data-testid="stChatInput"] {{
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1.5px solid {t['skyblue']} !important;
        border-radius: 40px !important;
        box-shadow: 0 4px 14px rgba(14, 27, 72, 0.15) !important;
    }}

    [data-testid="stChatInput"] * {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        filter: none !important;
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

    /* 1. Wrapper di sekitar tombol kirim — pakai flush ke semua level
          div di dalam stChatInput agar tidak ada bg/kotak sisa yang
          mengintip di belakang lingkaran. Ini jauh lebih robust tanpa
          :has() yang rentan mismatch antar versi browser/Streamlit. */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div,
    [data-testid="stChatInput"] > div > div > div > div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    /* Kembalikan background container utama agar tetap terlihat */
    [data-testid="stChatInput"] {{
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1.5px solid {t['skyblue']} !important;
    }}

    /* 2. Tombol bulat itu sendiri */
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
        min-width: 32px !important;
        min-height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        -webkit-appearance: none !important;
        appearance: none !important;
        transition: filter 0.2s ease, background-color 0.2s ease !important;
    }}

    /* 3. Cursor & hover — kunci fix untuk "block merah" saat hover.
          Browser menampilkan cursor "not-allowed" (lingkaran dicoret,
          sering kelihatan kemerahan di Windows) kalau tombolnya dalam
          state disabled (mis. textarea masih kosong). Di sini kita
          bedakan tegas: aktif = pointer, nonaktif = not-allowed tapi
          dengan tampilan pudar yang jelas, bukan tombol yang kelihatan
          normal tapi cursor-nya nyasar. */
    [data-testid="stChatInput"] button:not(:disabled),
    [data-testid="stChatInputSubmitButton"]:not(:disabled) {{
        cursor: pointer !important;
    }}
    [data-testid="stChatInput"] button:disabled,
    [data-testid="stChatInputSubmitButton"]:disabled,
    [data-testid="stChatInput"] button[disabled],
    [data-testid="stChatInputSubmitButton"][disabled] {{
        cursor: not-allowed !important;
        background-color: rgba(193, 141, 180, 0.45) !important;
    }}

    [data-testid="stChatInput"] button:hover:not(:disabled),
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled),
    [data-testid="stChatInput"] button:active:not(:disabled),
    [data-testid="stChatInputSubmitButton"]:active:not(:disabled) {{
        background-color: {t['mauve']} !important;
        filter: brightness(0.88) !important;
    }}
    [data-testid="stChatInput"] button:focus,
    [data-testid="stChatInputSubmitButton"]:focus,
    [data-testid="stChatInput"] button:focus-visible,
    [data-testid="stChatInputSubmitButton"]:focus-visible {{
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(193, 141, 180, 0.35) !important;
    }}

    /* 4. Bersihkan elemen anak (svg wrapper, span, dll) dari border/bg sisa */
    [data-testid="stChatInput"] button *,
    [data-testid="stChatInputSubmitButton"] * {{
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }}
    [data-testid="stChatInput"] button svg rect,
    [data-testid="stChatInputSubmitButton"] svg rect {{
        display: none !important;
    }}

    /* 5. Ikon tombol kirim — sama kayak mic, SVG Material Icon dgn bounding-box
          path (fill="none" bawaan). fill putih + stroke none biar bounding-box
          tetap invisible dan panah-nya keliatan solid putih. */
    [data-testid="stChatInputSubmitButton"] svg {{
        display: inline-block !important;
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: none !important;
        width: 18px !important;
        height: 18px !important;
    }}

    /* 6. Ikon mic — SVG-nya Google Material Icon: path ke-1 cuma bounding-box
          placeholder (fill="none" bawaan, HARUS tetap invisible), path ke-2 baru
          bentuk mic beneran, digambar pakai fill bukan stroke. Rule lama maksa
          stroke:white ke-inherit ke path ke-1 -> kotak invisible-nya jadi keliatan.
          Fix: fill putih (buat mic-nya), stroke none (biar kotak placeholder
          tetap invisible, cuma path ke-1 yang punya fill="none" sendiri jadi
          tetap ketutup, path ke-2 ikutan fill putih dari inherit). */
    [data-testid="stChatInputMicButton"] svg {{
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: none !important;
        width: 18px !important;
        height: 18px !important;
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

    /* ============================================================ */
    /* RESPONSIVE MOBILE — layar sempit (hp/tablet portrait)         */
    /* ============================================================ */
    @media (max-width: 640px) {{
        .block-container {{
            padding-left: 12px !important;
            padding-right: 12px !important;
            padding-top: 0.8rem !important;
        }}
        .glass-panel {{
            padding: 14px 12px 6px !important;
            border-radius: 18px !important;
        }}
        /* Header utama — kecilkan logo & judul */
        .main-chat-header {{ margin: 2px 0 12px; }}
        .main-chat-header .logo-badge {{
            width: 44px !important; height: 44px !important; margin-bottom: 8px !important;
        }}
        .main-chat-header .logo-badge svg {{ width: 22px !important; height: 22px !important; }}
        .main-chat-header .title {{ font-size: 22px !important; }}
        .main-chat-header .subtitle {{ font-size: 12.5px !important; }}
        /* Grid 6 fitur — 2 kolom lebih enak digenggam daripada 3 sempit */
        .st-key-fitur_grid [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
        }}
        .st-key-fitur_grid [data-testid="column"] {{
            flex: 1 1 46% !important;
            min-width: 46% !important;
        }}
        .st-key-fitur_grid div.stButton > button {{
            height: 78px !important;
        }}
        .st-key-fitur_grid div.stButton > button p {{
            font-size: 12px !important;
        }}
        /* Bubble chat — full width & teks lebih rapat */
        [data-testid="stChatMessageContent"] {{
            max-width: 88vw !important;
            padding: 11px 14px !important;
        }}
        [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{
            font-size: 13.5px !important;
        }}
        /* Chip saran — biar wrap rapi, nggak kepotong */
        .st-key-chip_row [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 8px !important;
        }}
        .st-key-chip_row [data-testid="column"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }}
        .st-key-chip_row div.stButton > button {{
            padding: 8px 14px !important;
            font-size: 12.5px !important;
        }}
        /* Tombol darurat — perkecil & pas di layar kecil */
        div.st-key-panic_component {{
            top: 8px !important;
            right: 10px !important;
            width: 128px !important;
            height: 38px !important;
        }}
        /* Input chat & container bawah — padding lebih hemat */
        div[data-testid="stBottomBlockContainer"] {{
            padding-left: 12px !important;
            padding-right: 12px !important;
        }}
        [data-testid="stChatInput"] textarea {{
            font-size: 16px !important; /* cegah auto-zoom Safari/Chrome iOS saat fokus input */
        }}
        /* Sidebar — kecilkan logo & judul brand */
        .sidebar-logo {{ width: 44px !important; height: 44px !important; }}
        .sidebar-logo svg {{ width: 22px !important; height: 22px !important; }}
        .sidebar-brand-title {{ font-size: 17px !important; }}
        .sidebar-brand-sub {{ font-size: 11px !important; }}
    }}

    /* ============ TYPING INDICATOR — animasi "sedang mengetik" ============ */
    .typing-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 6px 2px;
    }}
    .typing-indicator span {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: {t['bot_text']};
        opacity: 0.35;
        animation: typing-bounce 1.1s infinite ease-in-out;
    }}
    .typing-indicator span:nth-child(1) {{ animation-delay: 0s; }}
    .typing-indicator span:nth-child(2) {{ animation-delay: 0.15s; }}
    .typing-indicator span:nth-child(3) {{ animation-delay: 0.3s; }}
    @keyframes typing-bounce {{
        0%, 60%, 100% {{ transform: translateY(0); opacity: 0.35; }}
        30% {{ transform: translateY(-6px); opacity: 1; }}
    }}
    </style>""", unsafe_allow_html=True)

FITUR = [
    ("\U0001f91d Konseling Kasus", "Cerita situasimu, dapat arahan empatik",
     "Saya mengalami situasi yang mungkin termasuk kekerasan seksual. Bisa bantu saya memahami apa yang terjadi?"),
    ("\U0001f4d1 Jenis & Pasal", "Jenis TPKS dan dasar hukumnya",
     "Apa saja jenis tindak pidana kekerasan seksual dalam UU TPKS?"),
    ("⚖️ Ancaman Pidana", "Sanksi penjara & denda",
     "Berapa ancaman pidana untuk pelecehan seksual fisik dan eksploitasi seksual?"),
    ("\U0001f6e1️ Hak Korban", "Restitusi & pemulihan",
     "Apa saja hak korban dan bagaimana mekanisme restitusi menurut UU TPKS?"),
    ("\U0001f4cb Alur Melapor", "Langkah lapor & pembuktian",
     "Bagaimana cara melapor kasus kekerasan seksual dan alat bukti apa yang diakui?"),
    ("\U0001f4de Bantuan Darurat", "Kontak lembaga layanan",
     "Saya butuh nomor dan kontak lembaga bantuan untuk korban kekerasan seksual."),
]

MENU_ITEMS = [
    ("konseling", "\U0001f91d", "Konseling"),
    ("pasal", "⚖️", "Telusur Pasal"),
    ("lapor", "\U0001f4cb", "Panduan Lapor"),
]
MENU_LABELS = {"konseling": "Konseling", "pasal": "Telusur Pasal", "lapor": "Panduan Lapor"}

THRESHOLD = 0.5
print_ram("Sebelum encode")
def retrieve(query, k=4, th=THRESHOLD):
    embed_model = load_embed()
    index, chunks = load_store()
print_ram("Sesudah encode")

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
  pasal nggak ada yang pas, LEWATI SAJA tanpa billing "tidak ada info" — fokus ke dukungan/
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
  tidak nyaman." (ini parafrase, dilarang)
- WAJIB menyelipkan emoji/emoticon yang relevan, hangat, dan menenangkan di setiap respons (misalnya: 🫂, 💛, 🛡️, ✨) di dalam bubble chat agar terasa suportif dan ramah.
"""
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
  emosinya dan arahkan ke bantuan profesional/hotline dengan tenang, bukan dengan nada
  panik atau menghakimi.
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


def gemini_answer(api_key, user_input, history, mode, support_info):
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

    # Gemini pakai role "user"/"model" (bukan "assistant"), system prompt terpisah dari contents
    gemini_history = []
    for h in history[-5:]:
        role = "model" if h["role"] == "assistant" else "user"
        gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=h["content"])]))
    gemini_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))

    gemini_keys = [k for k in [api_key, os.environ.get("GEMINI_API_KEY_2", "")] if k]
    for idx, key in enumerate(gemini_keys):
        try:
            client = genai.Client(api_key=key)
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.75,
                    max_output_tokens=900,
                ),
            )
            got_chunk = False
            for chunk in stream:
                if chunk.text:
                    got_chunk = True
                    yield chunk.text
            if got_chunk:
                return
        except Exception as e:
            print(f"[GEMINI key #{idx+1} GAGAL] {type(e).__name__}: {e}", flush=True)
            continue
        except Exception as e:
            print(f"[GEMINI ERROR] {type(e).__name__}: {e}", flush=True)
        
    yield "⚠️ Chatbot sedang limit. Coba lagi beberapa saat lagi.\n\nKalau mendesak, hubungi **SAPA 129**."
    
def transcribe_audio(audio_bytes_io, api_key, model_name="gemini-2.5-flash"):
    """Kirim audio ke Gemini, kembalikan teks hasil transkripsi (Bahasa Indonesia)."""
    audio_bytes_io.seek(0)
    audio_bytes = audio_bytes_io.read()
    client = genai.Client(api_key=api_key)
    print_ram("Sebelum Gemini")
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Transkripsikan audio ini ke teks Bahasa Indonesia. Balas HANYA dengan teks "
            "transkripsinya saja, tanpa embel-embel atau penjelasan tambahan.",
        ],
    )
    print_ram("Sesudah Gemini")
    transcription = response.text
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()

def _pdf_sanitize(text):
    from fpdf import FPDF
    replacements = {
        "—": "-", "–": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "…": "...",
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

for k, v in [("messages", []), ("active_menu", "konseling"), ("pending", None), ("text_size", "Sedang"), ("last_emotion_result", None)]:
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
panic_exit_html = """
<!DOCTYPE html>
<html>
<head>
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    body {
        background: transparent;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        overflow: hidden;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        height: 100%;
    }
    .panic-btn {
        background-color: #D8342A;
        color: #FFFFFF;
        font-weight: 800;
        border: none;
        border-radius: 999px;
        padding: 8px 18px;
        font-size: 13px;
        cursor: pointer;
        box-shadow: 0 4px 14px rgba(216, 52, 42, .4);
        white-space: nowrap; /* Mencegah teks terlipat jadi 2 baris */
        transition: background 0.2s ease, transform 0.1s ease;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .panic-btn:hover {
        background-color: #B92A21;
    }
    .panic-btn:active {
        transform: scale(0.96);
    }
</style>
</head>
<body>
    <button onclick="emergencyExit()" class="panic-btn">🚨 Keluar Cepat</button>

    <script>
    function emergencyExit() {
        try {
            window.top.location.replace("https://www.google.com");
        } catch (e) {
            window.open("https://www.google.com", "_blank");
            window.location.href = "https://www.google.com";
        }
    }
    </script>
</body>
</html>
"""

# Berikan wadah fixed dengan lebar 180px dan tinggi 45px agar tombol muat sempurna
st.markdown("""
<style>
div.st-key-panic_component {
    position: fixed !important;
    top: 10px !important;
    right: 20px !important;
    z-index: 9999999 !important;
    width: 180px !important;
    height: 45px !important;
}
div.st-key-panic_component iframe {
    width: 100% !important;
    height: 100% !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

with st.container(key="panic_component"):
    components_html(panic_exit_html, height=42, scrolling=False)
# ===================== SIDEBAR =====================
mode_key = st.session_state.active_menu
inject_text_size_css(st.session_state.text_size)

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-logo-wrap"><div class="sidebar-logo">{LOGO_SVG}</div></div>'
        '<div class="sidebar-brand-title">Ruang Aman</div>'
        '<div class="sidebar-brand-sub">Privasi terjaga, Ceritamu berharga!</div>',
        unsafe_allow_html=True)

    if st.button("➕  Konsultasi baru", use_container_width=True):
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

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("\U0001f511 Gemini API Key", type="password")

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

        emoji_map = {"sadness": "🫂", "fear": "🏡", "anger": "🍃", "happy": "🥰", "love": "💖"}
        current_emoji = emoji_map.get(last["label_dominan"], "💬") if last else "💬"

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
                    Tombol merah di pojok kiri atas akan langsung menghapus seluruh riwayat obrolan secara permanen dan mengalihkan browser ke Google demi menjaga privasimu.
                </div>
            </div>
        """, unsafe_allow_html=True)

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

# ===================== INPUT BAR — mic native, satu widget sama teks =====================
# accept_audio=True bikin tombol mic jadi bagian dari chat_input itu sendiri, jadi otomatis
# ke-pin di posisi yang sama (chat_input SELALU dirender Streamlit di container fixed bawah,
# sedangkan audio_input terpisah dulu nggak ikut ke-pin -> itu penyebab mic "geser" posisinya).
prompt = st.chat_input(
    "Tuliskan pesanmu di sini...",
    accept_audio=True,
    audio_sample_rate=16000,
)
user_input = None
if prompt:
    if prompt.audio is not None:
        with st.spinner("Mentranskripsi suara..."):
            try:
                if not api_key:
                    st.warning("🔑 GEMINI_API_KEY belum diatur — transkripsi audio tidak tersedia.")
                else:
                    user_input = transcribe_audio(prompt.audio, api_key)
            except Exception as e:
                st.warning(f"Gagal mentranskripsi audio: {e}")
    elif prompt.text:
        user_input = prompt.text

if st.session_state.pending and not user_input:
    user_input = st.session_state.pending; st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now()})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="\U0001f49b"):
        typing_ph = st.empty()
        typing_ph.markdown(TYPING_INDICATOR_HTML, unsafe_allow_html=True)

        # 1. Deteksi emosi + kalimat support dinamis sekaligus (1 Gemini call)
        support_info = get_support_flag(user_input, api_key)

        # 2. Simpan hasilnya ke session state
        st.session_state["last_emotion_result"] = {
            "label_dominan": support_info["emosi"],
            "confidence": support_info["confidence"],
            "ai_label": support_info["ai_label"]
        }

        typing_ph.empty()

        banner_text = None
        if support_info["perlu_rujukan"]:
            banner_text = get_support_banner(support_info["emosi"])
        if banner_text:
            st.markdown(f'<div class="support-banner">{banner_text}</div>', unsafe_allow_html=True)

        if not api_key:
            ans = "⚠️ Gemini API Key belum ada. Set Variable **GEMINI_API_KEY** di dashboard Railway.\n\nDarurat? **SAPA 129**."
            st.markdown(ans)
        else:
            try:
                ans = st.write_stream(gemini_answer(api_key, user_input, st.session_state.messages[:-1], mode_key, support_info))
            except Exception as e:
                ans = f"Maaf, ada kendala memanggil LLM: `{e}`\n\nKalau mendesak, hubungi **SAPA 129**."
                st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans, "time": datetime.now()})
    st.rerun()
