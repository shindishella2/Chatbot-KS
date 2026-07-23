# ⚖️ Chatbot Hukum TPKS (UU No. 12 Tahun 2022)

> AI-powered legal chatbot berbasis **Retrieval-Augmented Generation (RAG)** untuk membantu pengguna memahami Undang-Undang Tindak Pidana Kekerasan Seksual (UU TPKS). Sistem menggabungkan pencarian semantik menggunakan **FAISS**, embedding menggunakan **Sentence Transformers**, klasifikasi emosi pengguna, dan Large Language Model (LLM) melalui **Groq API** sehingga mampu menghasilkan jawaban yang relevan, cepat, dan kontekstual.

---

# Daftar Isi

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [LLM & AI Models](#llm--ai-models)
- [Getting Started](#getting-started)
- [Clone Repository](#1-clone-repository)
- [Create Virtual Environment](#2-create-virtual-environment)
- [Install Dependencies](#3-install-dependencies-development)
- [Run Notebook](#4-run-notebook)
- [Run Streamlit App](#5-run-streamlit)
- [Deploy to Railway](#deploy-to-railway)
- [Environment Variables](#environment-variables)
- [Generated Files](#generated-files)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

# Project Overview

Chatbot ini dikembangkan untuk membantu masyarakat memperoleh informasi mengenai **Undang-Undang Nomor 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS)** secara lebih mudah melalui percakapan alami (Natural Language).

Alih-alih hanya mengirimkan jawaban dari Large Language Model (LLM), sistem terlebih dahulu melakukan pencarian informasi pada dokumen UU yang telah diproses menjadi knowledge base menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**.

Dengan pendekatan tersebut, jawaban yang dihasilkan menjadi:

- lebih akurat
- mengurangi hallucination
- berdasarkan isi dokumen hukum
- dapat menjelaskan pasal yang relevan
- mampu menyesuaikan gaya bahasa berdasarkan emosi pengguna

---

# System Architecture

```
                  PDF UU TPKS
                       │
                       ▼
                OCR / Text Extraction
                       │
                       ▼
                 Text Cleaning
                       │
                       ▼
              Text Chunking
                       │
                       ▼
         Sentence Transformer Embedding
                       │
                       ▼
                 FAISS Index
                       │
──────────────────────────────────────────

                 User Question
                       │
                       ▼
          Emotion Classification
                       │
                       ▼
         Semantic Search (FAISS)
                       │
                       ▼
      Retrieved Relevant Chunks
                       │
                       ▼
             Prompt Engineering
                       │
                       ▼
               Groq LLM API
                       │
                       ▼
                Final Response
```

---

# Repository Structure

```
Chatbot-KS/
│
├── app.py
├── chatbot_hukum_TPKS_v15.ipynb
├── requirements.txt
├── Procfile
│
├── emotion_model_final/
│   ├── config.json
│   ├── tokenizer/
│   ├── model/
│   └── ...
│
├── faiss_index.index
├── chunks.pkl
│
├── poppler-26.02.0/
│
├── README.md
│
└── .devcontainer/
```

---

# Technology Stack

## Backend

- Python
- Streamlit
- Groq API

## NLP

- Transformers
- Sentence Transformers
- FAISS
- HuggingFace

## OCR

- PaddleOCR
- Poppler
- PDF2Image

## Machine Learning

- PyTorch (CPU)
- Scikit-learn

---

# LLM & AI Models

Project ini terdiri dari beberapa model AI yang memiliki fungsi berbeda.

---

## 1. Sentence Transformer

Digunakan untuk mengubah setiap potongan dokumen hukum menjadi embedding vector.

Fungsi:

- Semantic Search
- Similarity Search
- Vector Database

Output:

```
Text
↓

768-dimensional vector
```

Model ini **tidak menghasilkan jawaban**, melainkan hanya mengubah teks menjadi representasi numerik agar dapat dicari menggunakan FAISS.

---

## 2. FAISS

FAISS bukan merupakan model AI, melainkan vector database yang digunakan untuk melakukan pencarian embedding secara sangat cepat.

Fungsi:

- menyimpan embedding
- mencari chunk paling relevan
- nearest neighbor search

---

## 3. Emotion Classification Model

Folder

```
emotion_model_final/
```

berisi model klasifikasi emosi pengguna.

Model ini digunakan untuk mengenali kondisi emosional pertanyaan pengguna, misalnya:

- marah
- sedih
- takut
- netral
- bingung

Hasil klasifikasi digunakan untuk mengatur gaya bahasa jawaban sehingga chatbot menjadi lebih empatik.

---

## 4. Groq Large Language Model

LLM digunakan pada tahap akhir setelah retrieval selesai.

Alur:

```
User Question

↓

Relevant Chunks

↓

Prompt

↓

Groq API

↓

Generated Answer
```

Groq digunakan sebagai inference engine yang menjalankan model bahasa besar dengan latency yang sangat rendah.

LLM bertugas untuk:

- memahami pertanyaan
- menggabungkan informasi hasil retrieval
- menyusun jawaban natural
- menjelaskan pasal hukum
- menghasilkan respon yang mudah dipahami

---

# Getting Started

Project ini terdiri dari dua bagian:

1. Notebook untuk preprocessing data
2. Streamlit untuk deployment chatbot

Notebook hanya dijalankan ketika ingin:

- membuat knowledge base baru
- mengganti dokumen hukum
- melatih ulang model

Sedangkan aplikasi Streamlit hanya menggunakan hasil preprocessing.

---

# 1. Clone Repository

```bash
git clone https://github.com/USERNAME/Chatbot-KS.git

cd Chatbot-KS
```

---

# 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# 3. Install Dependencies (Development)

> **Catatan Penting**
>
> `requirements.txt` pada repository **hanya digunakan untuk deployment Railway**, bukan untuk menjalankan notebook dari awal.

Install library berikut.

## Install Torch CPU

Disarankan menggunakan Torch CPU karena lebih stabil dibandingkan beberapa versi CUDA yang sering menyebabkan crash.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Install Library

```bash
pip install \
numpy \
pandas \
matplotlib \
seaborn \
scikit-learn \
streamlit \
transformers \
sentence-transformers \
faiss-cpu \
groq \
python-dotenv \
pdf2image \
paddleocr \
paddlepaddle \
opencv-python \
Pillow \
tqdm \
joblib \
ipywidgets \
notebook
```

Jika menggunakan Jupyter Lab:

```bash
pip install jupyterlab
```

---

# Poppler Installation

### Windows

Download Poppler lalu ekstrak.

Tambahkan folder berikut ke PATH.

```
poppler/bin
```

Atau gunakan folder Poppler yang sudah tersedia di repository.

---

# 4. Run Notebook

Jalankan

```
chatbot_hukum_TPKS_v15.ipynb
```

Notebook akan melakukan proses:

- OCR
- Cleaning
- Chunking
- Embedding
- FAISS Indexing
- Training Emotion Model
- Testing

Output yang dihasilkan:

```
faiss_index.index

chunks.pkl

emotion_model_final/
```

File-file tersebut digunakan oleh aplikasi Streamlit.

---

# 5. Run Streamlit

Setelah seluruh file hasil preprocessing tersedia.

Buat file

```
.env
```

Isi:

```
GROQ_API_KEY=YOUR_API_KEY
```

Kemudian jalankan

```bash
streamlit run app.py
```

Secara default aplikasi berjalan di

```
http://localhost:8501
```

---

# Deploy to Railway

Repository ini telah disiapkan untuk deployment menggunakan Railway.

## Step 1

Push project ke GitHub.

---

## Step 2

Login ke Railway.

---

## Step 3

Klik

```
New Project
```

---

## Step 4

Pilih

```
Deploy from GitHub Repo
```

---

## Step 5

Pilih repository.

Railway akan otomatis membaca

```
Procfile
```

dan

```
requirements.txt
```

---

## Step 6

Tambahkan Environment Variables.

---

# Environment Variables

Buat Environment Variable berikut.

```
GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

Apabila menggunakan API tambahan, tambahkan sesuai kebutuhan.

---

# Generated Files

Notebook akan menghasilkan file berikut.

## faiss_index.index

Vector database untuk semantic search.

---

## chunks.pkl

Daftar seluruh chunk dokumen.

---

## emotion_model_final/

Berisi model klasifikasi emosi yang digunakan saat chatbot berjalan.

---

# Workflow

```
PDF

↓

OCR

↓

Cleaning

↓

Chunking

↓

Embedding

↓

FAISS

↓

Deployment

↓

User Question

↓

Emotion Detection

↓

Semantic Search

↓

Prompt

↓

Groq LLM

↓

Answer
```

---

# Troubleshooting

## Torch sering crash

Gunakan versi CPU.

```bash
pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ModuleNotFoundError

Install library yang belum tersedia.

Contoh:

```bash
pip install sentence-transformers
```

atau

```bash
pip install paddleocr
```

---

## OCR tidak berjalan

Pastikan:

- Poppler sudah terinstall
- PATH Poppler sudah benar
- PaddleOCR berhasil terinstall

---

## FAISS Error

Gunakan

```bash
pip install faiss-cpu
```

---

## Railway Build Failed

Pastikan:

- requirements.txt sesuai deployment
- Procfile tersedia
- Environment Variable telah ditambahkan
- Python version sesuai dengan yang digunakan saat pengembangan

---

# Notes

- Notebook digunakan untuk preprocessing dan pelatihan model.
- Streamlit hanya digunakan sebagai aplikasi chatbot.
- `requirements.txt` difokuskan untuk deployment di Railway agar proses build lebih ringan.
- Untuk pengembangan lokal, gunakan panduan instalasi library pada README ini, bukan `requirements.txt`.

---

# License

Project ini dikembangkan untuk keperluan penelitian dan edukasi mengenai implementasi **Retrieval-Augmented Generation (RAG)** pada chatbot hukum berbasis **UU No. 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS)**.

Silakan menyesuaikan dan mengembangkan repository ini sesuai kebutuhan dengan tetap memperhatikan lisensi dari library dan model pihak ketiga yang digunakan.
