# ⚖️ Chatbot Hukum TPKS
### AI-Powered Legal Assistant using Retrieval-Augmented Generation (RAG)

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00599C?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/Sentence-Transformers-yellow?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM_API-black?style=for-the-badge)
![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E?style=for-the-badge&logo=railway)

</p>

---

## 📖 Overview

Chatbot Hukum TPKS merupakan chatbot berbasis Artificial Intelligence yang dirancang untuk membantu masyarakat memahami isi **Undang-Undang Nomor 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS)** secara lebih mudah melalui percakapan alami (Natural Language).

Berbeda dengan chatbot berbasis Large Language Model (LLM) yang hanya mengandalkan pengetahuan bawaan model, sistem ini mengimplementasikan pendekatan **Retrieval-Augmented Generation (RAG)** sehingga setiap jawaban didasarkan pada isi dokumen hukum yang telah diproses sebelumnya.

Selain itu, chatbot juga memiliki modul **Emotion Classification** sehingga jawaban dapat disampaikan dengan gaya bahasa yang lebih empatik sesuai kondisi emosional pengguna.

---

## 🎯 Objectives

Project ini bertujuan untuk:

- Membantu masyarakat memahami isi UU TPKS.
- Mengurangi kesalahan interpretasi terhadap pasal-pasal hukum.
- Menyediakan chatbot hukum yang lebih akurat melalui Retrieval-Augmented Generation (RAG).
- Mengurangi hallucination pada Large Language Model.
- Menghasilkan jawaban yang lebih empatik melalui emotion detection.
- Menjadi contoh implementasi AI pada domain legal.

---

# ✨ Features

- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic Search menggunakan FAISS
- 🧠 Sentence Transformer Embedding
- ❤️ Emotion Classification
- 🤖 Large Language Model melalui Groq API
- 📄 OCR PDF menggunakan PaddleOCR
- 💬 Streamlit Web Application
- 🚀 Railway Deployment Ready
- ⚡ Fast Inference
- 📖 Knowledge Base berbasis Dokumen UU

---

# 🖼️ Demo

Tambahkan screenshot aplikasi pada bagian berikut.

```
assets/

├── homepage.png
├── chat.png
├── retrieval.png
└── architecture.png
```

Contoh:

```markdown
![Homepage](assets/homepage.png)

![Chat](assets/chat.png)
```

---

# 🏗️ System Architecture

```
                    UU TPKS (PDF)
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
                   FAISS Vector Index

========================================================

                    User Question
                           │
                           ▼
                Emotion Classification
                           │
                           ▼
               Semantic Search (FAISS)
                           │
                           ▼
             Top-K Relevant Document Chunks
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

# 📂 Repository Structure

```
Chatbot-KS/
│
├── app.py
├── chatbot_hukum_TPKS_v15.ipynb
│
├── requirements.txt
├── requirements-dev.txt
│
├── Procfile
├── README.md
│
├── faiss_index.index
├── chunks.pkl
│
├── emotion_model_final/
│
├── poppler-26.02.0/
│
├── .devcontainer/
│
└── assets/
```

---

# 📁 Folder Explanation

## app.py

Merupakan aplikasi utama berbasis **Streamlit**.

File ini bertanggung jawab untuk:

- menerima pertanyaan pengguna
- melakukan semantic search
- mengirim prompt ke Groq
- menampilkan jawaban

Notebook **tidak dijalankan** ketika aplikasi berjalan.

---

## chatbot_hukum_TPKS_v15.ipynb

Notebook preprocessing.

Notebook digunakan untuk:

- OCR
- Cleaning
- Chunking
- Embedding
- FAISS Indexing
- Emotion Model Training

Notebook hanya dijalankan ketika ingin membangun ulang knowledge base.

---

## emotion_model_final/

Berisi model hasil training emotion classifier.

Model ini digunakan ketika aplikasi berjalan.

---

## faiss_index.index

Vector database yang digunakan untuk semantic retrieval.

---

## chunks.pkl

Berisi seluruh potongan dokumen hasil preprocessing.

---

## Procfile

Digunakan Railway untuk menentukan command ketika deployment.

---

## requirements.txt

Digunakan **khusus deployment Railway**.

---

## requirements-dev.txt

Digunakan untuk menjalankan notebook dari awal.

---

# 🧠 AI Models

Project ini terdiri dari beberapa model AI yang bekerja bersama.

---

# 1. Sentence Transformer

Sentence Transformer digunakan untuk mengubah setiap potongan dokumen menjadi embedding vector.

Alur:

```
Text

↓

Embedding Vector (Dense Representation)
```

Embedding digunakan agar komputer dapat memahami kemiripan makna antar kalimat.

Contoh:

```
Apa hak korban?

↓

[0.283, 0.192, ..., 0.441]
```

Embedding inilah yang nantinya disimpan ke dalam FAISS.

### Fungsi

- Semantic Search
- Similarity Search
- Dense Retrieval

### Library

```
sentence-transformers
```

---

# 2. FAISS

FAISS (Facebook AI Similarity Search) merupakan vector database.

Tugas FAISS adalah mencari embedding yang paling mirip dengan pertanyaan pengguna.

```
Question Embedding

↓

Nearest Neighbor Search

↓

Top K Chunks
```

Tanpa FAISS, chatbot harus membandingkan seluruh dokumen satu per satu sehingga proses menjadi sangat lambat.

### Fungsi

- Vector Indexing
- Semantic Retrieval
- Fast Similarity Search

---

# 3. Emotion Classification

Model ini digunakan untuk mengenali kondisi emosional pengguna.

Misalnya:

```
Saya takut melapor.

↓

Fear
```

atau

```
Saya marah kepada pelaku.

↓

Anger
```

atau

```
Apa isi Pasal 5?

↓

Neutral
```

Informasi tersebut digunakan untuk mengatur gaya bahasa jawaban chatbot.

Contoh:

Neutral

> Berikut isi Pasal 5...

Fear

> Saya memahami kekhawatiran Anda. Berdasarkan UU TPKS...

Dengan demikian chatbot tidak hanya menjawab pertanyaan tetapi juga lebih empatik.

---

# 4. Groq Large Language Model

Setelah retrieval selesai, seluruh konteks yang relevan dikirim ke Large Language Model melalui Groq API.

LLM bertugas untuk:

- memahami pertanyaan
- memahami konteks hukum
- menggabungkan beberapa pasal
- menyusun jawaban natural
- menghindari hallucination

Keuntungan menggunakan Groq:

- inference sangat cepat
- latency rendah
- kompatibel dengan model open-source modern
- mudah diintegrasikan melalui API

---

# 🔄 Complete AI Pipeline

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

FAISS Index

──────────────────────────────

Question

↓

Emotion Detection

↓

Embedding

↓

FAISS Search

↓

Top-K Context

↓

Prompt Engineering

↓

Groq LLM

↓

Final Answer
```

---

# 🛠️ Technology Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | Python |
| OCR | PaddleOCR |
| PDF Processing | pdf2image |
| Embedding | Sentence Transformers |
| Vector Database | FAISS |
| Deep Learning | PyTorch |
| Emotion Model | Transformers |
| LLM | Groq API |
| Deployment | Railway |

---

# 🚀 Getting Started

Repository ini memiliki dua mode penggunaan.

## Development

Digunakan ketika ingin:

- melakukan OCR ulang
- membuat embedding baru
- membuat FAISS baru
- melatih ulang emotion model

Mode ini menggunakan:

```
requirements-dev.txt
```

---

## Deployment

Digunakan ketika chatbot sudah siap digunakan.

Mode ini menggunakan:

```
requirements.txt
```

yang jauh lebih ringan karena hanya memuat library yang dibutuhkan oleh aplikasi Streamlit.

---

# 💻 Development Installation

## 1. Clone Repository

```bash
git clone https://github.com/USERNAME/Chatbot-KS.git

cd Chatbot-KS
```

---

## 2. Create Virtual Environment

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

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Torch (CPU)

Project ini **sangat disarankan menggunakan Torch CPU**.

Beberapa versi CUDA sering menyebabkan crash ketika menjalankan notebook, terutama saat menggunakan kombinasi `transformers`, `sentence-transformers`, dan `PaddleOCR`.

Install menggunakan:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

---

# 📦 Library Explanation

| Library | Function |
|----------|----------|
| torch | Deep Learning Framework |
| transformers | NLP Model |
| sentence-transformers | Text Embedding |
| faiss-cpu | Vector Database |
| paddleocr | OCR Engine |
| paddlepaddle | Backend PaddleOCR |
| pdf2image | Convert PDF menjadi gambar |
| opencv-python | Image Processing |
| Pillow | Image Utilities |
| scikit-learn | Machine Learning |
| streamlit | Web Application |
| groq | Groq API Client |
| python-dotenv | Environment Variable |
| matplotlib | Visualisasi |
| seaborn | Visualisasi |
| pandas | Data Processing |
| numpy | Numerical Computing |
| tqdm | Progress Bar |
| joblib | Model Serialization |

---

# 📄 Poppler Installation

OCR memerlukan Poppler untuk mengubah file PDF menjadi gambar.

Repository telah menyediakan folder:

```
poppler-26.02.0/
```

Apabila menggunakan Windows, tambahkan folder berikut ke PATH:

```
poppler/bin
```

Atau gunakan path tersebut langsung pada notebook sesuai kebutuhan.

---

# ▶️ Running Notebook

```
chatbot_hukum_TPKS_v15.ipynb
```

Notebook terdiri dari beberapa tahapan preprocessing yang akan menghasilkan knowledge base dan model yang digunakan oleh aplikasi chatbot.

# 📒 Notebook Pipeline

Notebook `chatbot_hukum_TPKS_v15.ipynb` merupakan inti dari proses pembangunan knowledge base. Seluruh proses preprocessing dilakukan pada notebook ini sebelum aplikasi Streamlit dijalankan.

---

# Step 1 — Load PDF

Tahap pertama adalah membaca dokumen **UU No. 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual**.

Input:

```
UU TPKS.pdf
```

Output:

```
List of PDF Pages
```

Tujuan tahap ini adalah mempersiapkan dokumen agar dapat diproses oleh OCR.

---

# Step 2 — OCR (Optical Character Recognition)

Dokumen PDF dikonversi menjadi gambar menggunakan **pdf2image**, kemudian setiap halaman diproses menggunakan **PaddleOCR**.

Pipeline:

```
PDF

↓

Image

↓

PaddleOCR

↓

Raw Text
```

Library yang digunakan:

```
pdf2image

paddleocr

opencv-python

Pillow
```

Output:

```
Raw Text
```

---

# Step 3 — Text Cleaning

Hasil OCR biasanya masih mengandung karakter yang tidak diperlukan, seperti:

- nomor halaman
- karakter asing
- spasi berlebih
- baris kosong
- simbol OCR

Contoh:

```
Pasal   5

....

```

menjadi

```
Pasal 5
```

Tahap ini sangat penting karena kualitas embedding sangat bergantung pada kualitas teks.

---

# Step 4 — Text Chunking

Dokumen hukum memiliki panjang yang sangat besar sehingga tidak dapat langsung diberikan kepada LLM.

Oleh karena itu dokumen dibagi menjadi beberapa chunk.

Contoh:

```
Chunk 1

Pasal 1

...

Chunk 2

Pasal 2

...

Chunk 3

Pasal 3
```

Chunking membuat proses retrieval menjadi jauh lebih efektif.

Output:

```
chunks.pkl
```

---

# Step 5 — Sentence Embedding

Setiap chunk diubah menjadi embedding vector menggunakan Sentence Transformer.

```
Chunk

↓

Embedding

↓

768-dimensional Vector
```

Embedding ini akan digunakan sebagai representasi numerik setiap dokumen.

---

# Step 6 — FAISS Indexing

Seluruh embedding disimpan ke dalam FAISS.

```
Embedding

↓

FAISS

↓

Vector Index
```

Output:

```
faiss_index.index
```

Ketika pengguna bertanya, embedding pertanyaan akan dibandingkan dengan seluruh embedding yang ada di FAISS.

---

# Step 7 — Emotion Model Training

Notebook juga melakukan proses training emotion classifier.

Pipeline:

```
Dataset

↓

Tokenizer

↓

Transformer

↓

Training

↓

Evaluation

↓

emotion_model_final/
```

Model hasil training disimpan sehingga tidak perlu dilatih ulang setiap kali aplikasi dijalankan.

---

# Step 8 — Evaluation

Notebook melakukan evaluasi model untuk memastikan:

- embedding berjalan dengan benar
- retrieval menghasilkan chunk yang relevan
- emotion classifier dapat melakukan prediksi
- model berhasil disimpan

---

# Output Notebook

Notebook menghasilkan beberapa file penting.

```
faiss_index.index

chunks.pkl

emotion_model_final/
```

Ketiga file tersebut digunakan langsung oleh aplikasi Streamlit.

---

# 🚀 Running Streamlit

Setelah notebook selesai dijalankan, aplikasi dapat dijalankan menggunakan Streamlit.

---

## 1. Buat Environment Variable

Buat file

```
.env
```

Isi dengan API Key Groq.

```
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

Secara default aplikasi dapat diakses pada:

```
http://localhost:8501
```

---

# 🔄 Runtime Workflow

Ketika aplikasi dijalankan, alur sistem menjadi sebagai berikut.

```
User

↓

Input Question

↓

Emotion Detection

↓

Sentence Embedding

↓

FAISS Retrieval

↓

Top-K Context

↓

Prompt Engineering

↓

Groq API

↓

Generated Response

↓

Streamlit Interface
```

---

# 📝 Prompt Engineering

Prompt yang dikirim ke LLM terdiri dari tiga bagian utama.

```
System Prompt

+

Retrieved Context

+

User Question
```

Dengan pendekatan ini model tidak menjawab berdasarkan pengetahuan umum saja, tetapi juga mempertimbangkan isi dokumen hukum yang relevan.

---

# 🌐 Railway Deployment

Repository ini telah disiapkan untuk deployment menggunakan Railway.

---

## Step 1

Push repository ke GitHub.

```bash
git add .

git commit -m "Initial Commit"

git push origin main
```

---

## Step 2

Masuk ke Railway.

https://railway.app

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
Deploy from GitHub Repository
```

---

## Step 5

Hubungkan repository.

Railway akan membaca

```
Procfile

requirements.txt
```

---

## Step 6

Tambahkan Environment Variable.

```
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Step 7

Klik Deploy.

Railway akan melakukan proses:

```
Clone Repository

↓

Install requirements.txt

↓

Build Project

↓

Run Streamlit

↓

Deployment Success
```

---

# ⚙️ Environment Variables

| Variable | Description |
|-----------|-------------|
| GROQ_API_KEY | API Key Groq |

---

# 📦 Generated Files

## chunks.pkl

Berisi seluruh hasil chunking dokumen.

Digunakan oleh aplikasi ketika retrieval dilakukan.

---

## faiss_index.index

Berisi vector database.

Digunakan untuk semantic search.

---

## emotion_model_final/

Model hasil fine-tuning emotion classifier.

Digunakan ketika chatbot berjalan.

---

# 🔁 Complete Project Workflow

```
UU TPKS PDF

↓

OCR

↓

Cleaning

↓

Chunking

↓

Sentence Transformer

↓

Embedding

↓

FAISS

↓

Knowledge Base

================================================

User Question

↓

Emotion Classification

↓

Embedding

↓

Semantic Search

↓

Relevant Chunks

↓

Prompt Engineering

↓

Groq LLM

↓

Generated Response

↓

Streamlit
```

---

# 📊 Advantages of This Architecture

Dibandingkan chatbot biasa, pendekatan yang digunakan pada project ini memiliki beberapa keunggulan.

- Jawaban berdasarkan dokumen hukum.
- Mengurangi hallucination.
- Semantic Search lebih baik dibanding keyword search.
- Respon lebih empatik melalui emotion classification.
- Mudah memperbarui knowledge base tanpa melatih ulang LLM.
- Deployment ringan karena preprocessing dilakukan sebelumnya.

---

# ❓ Frequently Asked Questions (FAQ)

### Mengapa menggunakan RAG?

Karena LLM dapat menghasilkan jawaban yang kurang akurat jika hanya mengandalkan pengetahuan bawaan. RAG memastikan jawaban didukung oleh dokumen hukum yang relevan.

---

### Mengapa menggunakan FAISS?

FAISS memungkinkan pencarian semantik yang sangat cepat pada ribuan embedding tanpa perlu membandingkan dokumen satu per satu.

---

### Mengapa menggunakan Sentence Transformer?

Sentence Transformer menghasilkan embedding yang menangkap makna kalimat, sehingga pencarian tidak hanya bergantung pada kecocokan kata, tetapi juga pada kesamaan konteks.

---

### Mengapa menggunakan Groq?

Groq menyediakan layanan inference dengan latensi rendah untuk berbagai model LLM open-source, sehingga chatbot dapat memberikan respons lebih cepat.

---

### Mengapa ada Emotion Classification?

Emotion Classification membantu chatbot menyesuaikan gaya bahasa agar lebih sesuai dengan kondisi emosional pengguna, terutama pada topik sensitif seperti kekerasan seksual.

---

### Apakah chatbot dapat menjawab pertanyaan di luar UU TPKS?

Chatbot dirancang untuk berfokus pada dokumen UU TPKS. Pertanyaan di luar ruang lingkup tersebut mungkin tidak menghasilkan jawaban yang optimal.

---

### Apakah knowledge base dapat diperbarui?

Ya. Jalankan kembali notebook untuk melakukan OCR, preprocessing, embedding, dan pembangunan ulang indeks FAISS.

---

# 🛠️ Troubleshooting

## Torch Crash

Gunakan Torch CPU.

```bash
pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ModuleNotFoundError

Pastikan seluruh library development telah diinstall.

```bash
pip install -r requirements-dev.txt
```

---

## PaddleOCR Error

Pastikan:

- PaddleOCR berhasil terinstall.
- PaddlePaddle sesuai dengan sistem operasi.
- Poppler tersedia.

---

## Poppler Error

Pastikan folder Poppler telah ditambahkan ke PATH atau path Poppler telah dikonfigurasi pada notebook.

---

## FAISS Error

Install ulang FAISS CPU.

```bash
pip install --upgrade faiss-cpu
```

---

## Streamlit Error

Pastikan aplikasi dijalankan menggunakan:

```bash
streamlit run app.py
```

---

## Environment Variable Not Found

Pastikan file `.env` tersedia dan berisi:

```
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## Railway Build Failed

Periksa:

- `requirements.txt`
- `Procfile`
- Python version
- Environment Variables

---

## OCR Sangat Lambat

OCR merupakan proses yang paling memakan waktu karena seluruh halaman PDF diproses satu per satu. Setelah knowledge base selesai dibuat, OCR tidak perlu dijalankan kembali kecuali dokumen diperbarui.

---

# 🚀 Future Improvements

Beberapa pengembangan yang dapat dilakukan pada versi berikutnya:

- Hybrid Search (BM25 + Dense Retrieval)
- Cross Encoder Reranker
- Citation per Pasal
- Multi-document Knowledge Base
- Conversation Memory
- Multi-turn Retrieval
- Feedback Learning
- Voice Input
- PDF Upload oleh pengguna
- Dashboard monitoring penggunaan
- Logging dan analytics
- Docker support
- Kubernetes deployment
- REST API menggunakan FastAPI
- Authentication pengguna
- Riwayat percakapan

---

# 📚 References

- Johnson, J., Douze, M., & Jégou, H. (2019). *FAISS: A Library for Efficient Similarity Search*.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT Networks*.
- Hugging Face Transformers Documentation.
- PaddleOCR Documentation.
- Streamlit Documentation.
- Groq API Documentation.
- PyTorch Documentation.

---

# 🤝 Contributing

Kontribusi sangat terbuka.

Langkah umum:

1. Fork repository.
2. Buat branch baru.
3. Implementasikan perubahan.
4. Commit perubahan.
5. Push ke repository pribadi.
6. Ajukan Pull Request.

---

# 📄 License

Project ini dikembangkan untuk tujuan penelitian dan edukasi mengenai implementasi **Retrieval-Augmented Generation (RAG)** pada chatbot hukum berbasis **Undang-Undang Nomor 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (UU TPKS)**.

Silakan menggunakan, mempelajari, dan mengembangkan repository ini sesuai kebutuhan dengan tetap memperhatikan lisensi dari seluruh library, model, dan layanan pihak ketiga yang digunakan.

---

<p align="center">
Made with ❤️ using Python, Streamlit, FAISS, Sentence Transformers, PaddleOCR, and Groq LLM.
</p>
