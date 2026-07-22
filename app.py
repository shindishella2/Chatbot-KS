import os
import random
import re
import pickle
from datetime import datetime
from fpdf import FPDF
import faiss
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as components_html
from sentence_transformers import SentenceTransformer
from groq import Groq

# Set Page Config
st.set_page_config(
    page_title="Ruang Aman - Konseling Hukum UU TPKS",
    page_icon="💛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== LOAD VECTOR DATABASE =====================
@st.cache_resource
def load_embed(): 
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_store():
    return faiss.read_index("faiss_index.index"), pickle.load(open("chunks.pkl","rb"))

embed_model = load_embed()
index, chunks = load_store()

# ===================== DYNAMIC EMOTION VIA GROQ AI =====================
def analyze_emotion_and_label_via_groq(api_key, user_text: str) -> dict:
    """Menggunakan Groq AI untuk mendeteksi emosi sekaligus membuat pesan penguat dinamis."""
    if not api_key:
        return {
            "emosi": "neutral",
            "perlu_rujukan": False,
            "ai_label": "Siap membantu dan mendengarkan ceritamu."
        }
    
    system_prompt = (
        "Kamu adalah sistem pengolah emosi untuk chatbot konseling kekerasan seksual.\n"
        "Tugasmu menganalisis pesan user dan menghasilkan output dalam format JSON MURNI (tanpa markdown/penjelasan tambahan):\n"
        "{\n"
        '  "emosi": "sadness" | "fear" | "anger" | "happy" | "love" | "neutral",\n'
        '  "perlu_rujukan": true | false,\n'
        '  "ai_label": "1 kalimat penguat yang sangat hangat, lembut, dan natural (maks 12 kata)"\n'
        "}\n\n"
        "ATURAN:\n"
        "- Set 'perlu_rujukan' = true HANYA jika emosi user tergolong 'sadness', 'fear', atau distress berat.\n"
        "- Bahasa pada 'ai_label' harus santai, empati, tanpa tanda kutip, seperti teman dekat."
    )
    
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pesan user: {user_text}"}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        import json
        res = json.loads(completion.choices[0].message.content)
        return {
            "emosi": res.get("emosi", "neutral"),
            "perlu_rujukan": res.get("perlu_rujukan", False),
            "ai_label": res.get("ai_label", "Siap membantu proses konselingmu.")
        }
    except Exception:
        return {
            "emosi": "neutral",
            "perlu_rujukan": False,
            "ai_label": "Aku di sini mendengarkanmu. Ceritakan saja, ya."
        }

SUPPORT_MESSAGES = {
    "sadness": [
        "🤍 Apa pun yang kamu rasakan sekarang itu valid. Kamu gak sendirian di sini.",
        "🤍 Terima kasih udah mau cerita. Pelan-pelan aja, gak perlu buru-buru.",
        "🤍 Kamu udah berani sejauh ini dengan cerita di sini. Itu bukan hal kecil.",
    ],
    "fear": [
        "🫂 Kamu aman untuk cerita di sini, dengan kecepatanmu sendiri.",
        "🫂 Gak apa-apa kalau masih takut. Kamu boleh berhenti kapan pun kamu perlu.",
        "🫂 Perasaan itu wajar. Kita jalan pelan-pelan aja, sesuai kesiapanmu.",
    ],
    "anger": [
        "🍃 Luapkan saja rasa kesalmu. Perasaanmu sangat berhak untuk didengar.",
    ]
}

def get_support_banner(emotion_label: str):
    pool = SUPPORT_MESSAGES.get(emotion_label)
    return random.choice(pool) if pool else None

# ===================== TEMA CSS =====================
T = dict(
    navy="#0E1B48", mauve="#C18DB4", blush="#E2CAD8", skyblue="#87A7D0",
    slate="#27425D", deep="#0E1F2F",
    sidebar_top="#FDE8D3", sidebar_bottom="#F5D7DB", active="#F1916D",
    bot_bg="#F0C987", bot_border="#F59E51", bot_text="#3B153A",
    user_bg="#0E1B48", user_border="#87A7D0", user_text="#FFFFFF",
    header_title="#F0C987", header_sub="#E6E6E6",
    appbg="linear-gradient(135deg, rgba(14,27,72,0.30) 0%, rgba(193,141,180,0.30) 25%, rgba(226,202,216,0.30) 50%, rgba(135,167,208,0.30) 75%, rgba(39,66,93,0.30) 100%), #0E1F2F",
)

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
    header[data-testid="stHeader"] {{ display:none !important; }}
    .block-container {{ max-width:860px; padding-top:2rem !important; }}

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

    .st-key-fitur_grid [data-testid="column"] {{ display: flex; }}
    .st-key-fitur_grid div.stButton {{ width: 100%; }}
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

    [data-testid="stChatMessage"] {{ background:transparent; border:none; padding:4px 0; gap:12px; }}
    [data-testid="stChatMessageContent"] {{
        max-width: 680px;
        border-radius:18px; padding:13px 18px;
        box-shadow:0 3px 10px rgba(14,27,72,.12);
    }}

    div[data-testid="stChatMessage"]:nth-of-type(odd) {{
        flex-direction: row-reverse;
        justify-content: flex-start;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] {{
        background:{t['user_bg']}; border:1.5px solid {t['user_border']};
        margin-left: auto;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(odd) [data-testid="stChatMessageContent"] p {{
        color:{t['user_text']} !important;
    }}

    div[data-testid="stChatMessage"]:nth-of-type(even) {{
        flex-direction: row;
        justify-content: flex-start;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] {{
        background:{t['bot_bg']}; border:1.5px solid {t['bot_border']};
        margin-right: auto;
    }}
    div[data-testid="stChatMessage"]:nth-of-type(even) [data-testid="stChatMessageContent"] p {{
        color:{t['bot_text']} !important;
    }}

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

    section[data-testid="stSidebar"] button[kind="primary"] {{
        background:{t['active']} !important; color:#FFFFFF !important; font-weight:800 !important;
        border:3.5px solid {t['active']} !important; border-radius:14px !important;
    }}

    .st-key-chip_row div.stButton > button {{
        background:rgba(255,255,255,0.55) !important; color:{t['navy']} !important;
        border:1.5px solid {t['skyblue']} !important; border-radius:999px !important;
        padding:6px 16px !important; font-size:13px !important; font-weight:600 !important;
    }}

    [data-testid="stChatInput"] {{
        background: rgba(255, 255, 255, 0.96) !important;
        border: 1.5px solid {t['skyblue']} !important;
        border-radius: 40px !important;
    }}

    [data-testid="stBottom"] {{ background: {t['deep']} !important; }}
    div[data-testid="stBottomBlockContainer"] {{ background: {t['deep']} !important; }}

    .disclaimer {{ font-size:11px; line-height:1.55; color:{t['slate']} !important; margin-top:14px; border-top:1px solid rgba(14,27,72,.12); padding-top:12px; }}
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
    ("🤝 Konseling Kasus", "Cerita situasimu, dapat arahan empatik", "Saya mengalami situasi yang mungkin termasuk kekerasan seksual. Bisa bantu saya memahami apa yang terjadi?"),
    ("📑 Jenis & Pasal", "Jenis TPKS dan dasar hukumnya", "Apa saja jenis tindak pidana kekerasan seksual dalam UU TPKS?"),
    ("⚖️ Ancaman Pidana", "Sanksi penjara & denda", "Berapa ancaman pidana untuk pelecehan seksual fisik dan eksploitasi seksual?"),
    ("🛡️ Hak Korban", "Restitusi & pemulihan", "Apa saja hak korban dan bagaimana mekanisme restitusi menurut UU TPKS?"),
    ("📋 Alur Melapor", "Langkah lapor & pembuktian", "Bagaimana cara melapor kasus kekerasan seksual dan alat bukti apa yang diakui?"),
    ("📞 Bantuan Darurat", "Kontak lembaga layanan", "Saya butuh nomor dan kontak lembaga bantuan untuk korban kekerasan seksual."),
]

MENU_ITEMS = [
    ("konseling", "🤝", "Konseling"),
    ("pasal", "⚖️", "Telusur Pasal"),
    ("lapor", "📋", "Panduan Lapor"),
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

BASE = """Kamu adalah "Pasal", asisten hukum berbahasa Indonesia yang HANYA membahas UU No. 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS).

PRINSIP WAJIB:
- SEBELUM menjawab, cek dulu: apakah pertanyaan user ADA HUBUNGANNYA dengan kekerasan seksual, TPKS, atau isi UU ini? Kalau SAMA SEKALI TIDAK NYAMBUNG, JANGAN dipaksa dikaitkan. Jawab singkat bahwa kamu cuma fokus bahas UU TPKS.
- Jawab HANYA dari materi pasal yang tersedia. Dilarang mengarang pasal.
- Sapaan WAJIB konsisten "kamu". JANGAN PERNAH pakai "Anda".
- HINDARI kata "saya"/"aku" saat chatbot merujuk ke dirinya sendiri.
- DILARANG memulai kalimat pertama jawaban dengan kata "Kamu"/"Anda", dan DILARANG parafrase kaku atas cerita user.
- WAJIB menyelipkan emoji/emoticon yang relevan dan hangat di setiap respons (misalnya: 🫂, 💛, 🛡️, ✨).
"""

PROMPTS = {
    "konseling": BASE + "\nPERAN SEKARANG: KONSELOR empatik. Validasi perasaan user, selipkan pasal jika sangat relevan. Maksimal 4 paragraf pendek.",
    "pasal": BASE + "\nPERAN SEKARANG: PENELUSUR PASAL. Jawab lugas, sebutkan pasal + sanksi pidana secara rinci.",
    "lapor": BASE + "\nPERAN SEKARANG: PEMANDU PELAPORAN. Beri langkah praktis 1-2-3 ke lembaga terkait & pembuktian.",
}

def groq_answer(api_key, user_input, history, mode, support_info):
    ctx_list, sim = retrieve(user_input)
    context = "\n\n".join(f"[Kutipan {i+1}] {c}" for i, c in enumerate(ctx_list)) if ctx_list else "(Tidak ada pasal relevan di atas ambang.)"
    system_prompt = PROMPTS[mode]

    if support_info.get("perlu_rujukan"):
        system_prompt += "\n\nCATATAN TAMBAHAN: User tampak tertekan. Jawab dengan sangat empatik dan sisipkan secara halus layanan SAPA 129."

    user_msg = f"PERTANYAAN PENGGUNA:\n{user_input}\n\nKONTEKS PASAL UU TPKS (relevansi {sim:.0%}):\n{context}"
    msgs = [{"role": "system", "content": system_prompt}]
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_msg})

    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=msgs, temperature=0.75, max_tokens=900, stream=True
        )
        for chunk in stream:
            d = chunk.choices[0].delta.content
            if d:
                yield d
    except Exception as e:
        yield "⚠️ Groq sedang limit. Coba lagi beberapa saat lagi.\n\nKalau mendesak, hubungi **SAPA 129**."

def transcribe_audio(audio_bytes_io, groq_client):
    audio_bytes_io.seek(0)
    transcription = groq_client.audio.transcriptions.create(
        file=("input.wav", audio_bytes_io.read()),
        model="whisper-large-v3-turbo",
        language="id",
        response_format="text",
        temperature=0.0,
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()

def _pdf_sanitize(text):
    replacements = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."}
    for k, v in replacements.items(): text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def build_transcript_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, _pdf_sanitize("RUANG AMAN - RINGKASAN PERCAKAPAN"), align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 10)
    for m in st.session_state.messages:
        who = "USER" if m["role"] == "user" else "RUANG AMAN"
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, _pdf_sanitize(f"{who}:"))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_sanitize(m["content"]))
        pdf.ln(2)
    return bytes(pdf.output())

# Initialize State
for k, v in [("messages", []), ("active_menu", "konseling"), ("pending", None), ("text_size", "Sedang"), ("last_emotion_result", None)]:
    if k not in st.session_state: st.session_state[k] = v

inject_css(T)

# ===================== PANIC EXIT =====================
panic_exit_html = """
<!DOCTYPE html>
<html>
<head>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: transparent; overflow: hidden; display: flex; justify-content: flex-end; align-items: center; height: 100%; }
    .panic-btn {
        background-color: #D8342A; color: #FFFFFF; font-weight: 800; border: none;
        border-radius: 999px; padding: 8px 18px; font-size: 13px; cursor: pointer;
        box-shadow: 0 4px 14px rgba(216, 52, 42, .4); white-space: nowrap;
    }
    .panic-btn:hover { background-color: #B92A21; }
</style>
</head>
<body>
    <button onclick="window.top.location.replace('https://www.google.com');" class="panic-btn">🚨 Keluar Cepat</button>
</body>
</html>
"""

st.markdown("""
<style>
div.st-key-panic_component { position: fixed !important; top: 12px !important; right: 20px !important; z-index: 9999999 !important; width: 180px !important; height: 48px !important; }
div.st-key-panic_component iframe { width: 100% !important; height: 100% !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

with st.container(key="panic_component"):
    components_html(panic_exit_html, height=45, scrolling=False)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown(f'<div class="sidebar-logo-wrap"><div class="sidebar-logo">{LOGO_SVG}</div></div><div class="sidebar-brand-title">Ruang Aman</div><div class="sidebar-brand-sub">Privasi terjaga, Ceritamu berharga!</div>', unsafe_allow_html=True)

    if st.button("➕ Konsultasi baru", use_container_width=True):
        st.session_state.messages = []; st.session_state.pending = None; st.rerun()

    if st.session_state.messages:
        st.download_button("💾 Simpan Percakapan (PDF)", data=build_transcript_pdf(), file_name="ruang-aman.pdf", mime="application/pdf", use_container_width=True)

    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("🔑 Groq API Key", type="password")

    for key, emoji, label in MENU_ITEMS:
        is_active = st.session_state.active_menu == key
        if st.button(f"{emoji} {label}", key=f"menu_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.active_menu = key
            st.rerun()

    tampilkan_emosi = st.checkbox("💭 Tampilkan mode percakapan", value=False)
    if tampilkan_emosi:
        last = st.session_state.get("last_emotion_result")
        emoji_map = {"sadness": "🫂", "fear": "🏡", "anger": "🍃", "happy": "🥰", "love": "💖"}
        current_emoji = emoji_map.get(last["emosi"], "💬") if last else "💬"
        current_label = last["ai_label"] if last else "Siap membantu proses konselingmu."
        st.markdown(f'<div style="padding:12px; border-radius:14px; background-color:rgba(255,255,255,0.35); text-align:center;"><div style="font-size:28px;">{current_emoji}</div><div style="font-size:13px; margin-top:4px; color:{T["navy"]}; font-weight:500;">{current_label}</div></div>', unsafe_allow_html=True)

groq_client = Groq(api_key=api_key) if api_key else None
mode_key = st.session_state.active_menu

# Header Utama
st.markdown(f'<div class="main-chat-header"><div class="logo-badge">{LOGO_SVG}</div><div class="title">Ruang Aman</div><div class="subtitle">Kamu tidak sendirian. Kami mendengarkan.</div></div>', unsafe_allow_html=True)

# Grid Fitur Utama (Hanya Tampil Jika Belum Ada Chat)
if not st.session_state.messages:
    fitur_container = st.container(key="fitur_grid")
    with fitur_container:
        cols = st.columns(3)
        for i, (j, d, p) in enumerate(FITUR):
            with cols[i % 3]:
                if st.button(f"{j}\n\n{d}", key=f"f{i}", use_container_width=True):
                    st.session_state.pending = p; st.rerun()

# Display Chat Messages
if st.session_state.messages:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="💛" if m["role"] == "assistant" else "👤"):
            st.markdown(m["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# Input Bar & Transkripsi Audio
prompt = st.chat_input("Tuliskan pesanmu di sini...", accept_audio=True, audio_sample_rate=16000)
user_input = None

if prompt:
    if prompt.audio is not None:
        with st.spinner("Mentranskripsi suara..."):
            try:
                user_input = transcribe_audio(prompt.audio, groq_client) if groq_client else None
            except Exception as e:
                st.warning(f"Gagal transkripsi audio: {e}")
    elif prompt.text:
        user_input = prompt.text

if st.session_state.pending and not user_input:
    user_input = st.session_state.pending; st.session_state.pending = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "time": datetime.now()})
    
    # 1. Analisis Emosi via Groq AI (Cepat & Tanpa Model Lokal)
    support_info = analyze_emotion_and_label_via_groq(api_key, user_input)
    st.session_state["last_emotion_result"] = support_info

    # 2. Render Chat Assistant
    with st.chat_message("assistant", avatar="💛"):
        if support_info.get("perlu_rujukan"):
            banner = get_support_banner(support_info["emosi"])
            if banner:
                st.markdown(f'<div class="support-banner">{banner}</div>', unsafe_allow_html=True)

        if not api_key:
            ans = "⚠️ Groq API Key belum ada. Silakan atur Secrets GROQ_API_KEY di Streamlit Cloud."
            st.markdown(ans)
        else:
            ans = st.write_stream(groq_answer(api_key, user_input, st.session_state.messages[:-1], mode_key, support_info))

    st.session_state.messages.append({"role": "assistant", "content": ans, "time": datetime.now()})
    st.rerun()
