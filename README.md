# 🦙 Llama RAG Chatbot

Proyek ini merupakan sistem chatbot yang dirancang khusus untuk layanan perpustakaan. Sistem ini dibangun menggunakan _Large Languange Model_ LLaMA 2 dan diperkuat dengan pendekatan RAG _(Retrieval-Augmented Generation)_ untuk memberikan jawaban yang lebih akurat dan sesuai konteks berdasarkan data perpustakaan yang nyata.

Untuk mencapai hal ini, sistem menggunakan FAISS sebagai basis data vektor untuk menyimpan dan mencari dokumen. Ketika pengguna mengajukan pertanyaan, sistem akan mengambil informasi yang relevan dari database dan mengirimkannya ke model untuk menghasilkan jawaban yang lebih tepat.

## Arsitektur
### Mengapa sistem dibangun dengan pendekatan ini?
   Sistem chatbot perpustakaan ini dirancang untuk mempermudah mahasiswa dan sivitas akademika Universitas Brawijaya dalam mencari informasi secara cepat dan efisien tanpa perlu datang langsung ke petugas. Beberapa alasan utama pemilihan arsitektur ini adalah:
   - Modularitas & Skalabilitas: Dengan memisahkan antar komponen (UI, backend, database, LLM), sistem dapat dengan mudah dikembangkan atau ditingkatkan tanpa memengaruhi keseluruhan arsitektur.
   - Responsif & Real-time: FastAPI dipilih karena performanya yang tinggi untuk REST API, sehingga cocok untuk menangani permintaan real-time dari chatbot.
   - Kemampuan Pemahaman Bahasa Alami: Integrasi dengan model LLM seperti LLaMA memungkinkan sistem memahami dan menjawab pertanyaan dalam bahasa alami dengan konteks yang lebih baik.
   - Efisiensi Pencarian Dokumen: Menggunakan FAISS sebagai vector database memungkinkan pencarian berbasis semantic similarity, lebih relevan daripada pencarian berbasis kata kunci biasa.
   - Portabilitas & Reproduksibilitas: Dengan Docker, sistem bisa dijalankan di berbagai lingkungan tanpa konfigurasi ulang, sedangkan MLflow memungkinkan pengawasan dan evaluasi eksperimen model secara otomatis.

### Bagaimana sistem ini bekerja?

1. Pengguna berinteraksi melalui antarmuka chatbot yang dibangun dengan Streamlit.
2. Permintaan dikirim ke backend melalui REST API yang dibuat menggunakan FastAPI.
3. FastAPI akan:
   - Jika perlu pencarian data, akan mengambil embedding dari pertanyaan pengguna.
   - Melakukan pencocokan dengan dokumen yang relevan di FAISS Vector Database.
5. Setelah dokumen relevan ditemukan, sistem menyusun prompt yang berisi:
   - Pertanyaan pengguna
   - Dokumen terkait
6. Prompt tersebut dikirim ke model LLaMA, yang akan menghasilkan jawaban berdasarkan konteks tersebut.
7. Jawaban dari model dikembalikan ke FastAPI dan diteruskan ke UI chatbot Streamlit.
8. Sementara itu, proses pelatihan model, pengujian, dan pengelolaan eksperimen dilakukan dengan MLflow, dan semua komponen dijalankan dalam Docker container untuk menjaga konsistensi dan kemudahan deployment.

## Quick Start
### Setup

1. **Create and Activate Conda Environment**
   
   Pastikan Anda telah menginstal Anaconda atau Miniconda.

   Buka terminal dan jalankan:
   ```bash
   conda create -n llama-chat python=3.10 -y
   conda activate llama-chat
   ```
   
3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

### Config Model
Buka file config.py untuk mengatur model:
```bash
USE_GPU = False  # True if GPU
```

# Model settings 
```bash
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_FILE = "llama-2-7b-chat.Q5_K_S.gguf"
```
📌 Catatan: Pastikan file model llama-2-7b-chat.Q5_K_S.gguf telah diunduh dan ditempatkan di direktori yang benar.

### Testing

1. **Test document loading**
   ```bash
   python -c "from src.document_loader import load_all_documents; print(load_all_documents())"
   ```

2. **Test RAG pipeline**
   ```bash
   python -c "from utils import setup_rag_pipeline; success, message, vectorstore = setup_rag_pipeline(); print(message)"
   ```

3. **Run the chatbot**
   ```bash
   # CLI interface
   python app.py
   
   # Web interface without bakcend
   streamlit run app.py
   ```
3. **Running with Docker + FastAPI Backend**

   Proyek ini telah dikontainerisasi menggunakan Docker untuk memudahkan proses deployment dan terintegrasi dengan backend FastAPI untuk inferensi berbasis API.

Langkah 1: Jalankan backend dengan Docker
   ```bash
   docker run -p 8000:8000 backend
   ```

      Langkah 2: Buka antarmuka frontend
   ```bash
   streamlit run frontend/streamlit_app.py --server.port 8501 &
   ```
   Alternatif (Tanpa Docker): Menjalankan Backend dengan Uvicorn
   Jika tidak ingin menggunakan Docker, backend FastAPI juga dapat dijalankan secara manual:

   Langkah 1: Jalankan backend dengan uvicorn
      ```bash
      uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
      ```
      Langkah 2: Buka antarmuka frontend
      ```bash
      streamlit run frontend/streamlit_app.py --server.port 8501 &
      ```
## Features

- Memuat dan memproses dokumen JSON dan teks
- Menggunakan FAISS untuk penyimpanan dan pencarian
- Menjawab dalam Bahasa Indonesia
- Mengurangi kemungkinan jawaban halusinasi ketika informasi tidak ditemukan
- Mendukung antarmuka CLI dan web

## Troubleshooting

- **Akses Hugging Face**: Pastikan model tersedia dalam format `.cache`
- **Performa**: Aktifkan GPU dengan mengatur `USE_CUDA=True`

## Structure

```
llama-rag-chatbot/
├── app.py                  # Main application entry point
├── requirements.txt        # Project dependencies
├── config.py               # Configuration settings
├── .env                    # Environment variables (create this file)
├── data/
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading utilities
│   ├── embedding.py        # Embedding generation
│   ├── indexing.py         # Vector indexing with FAISS
│   ├── chunking.py         # Text splitting logic
│   ├── retriever.py        # Document retrieval
│   ├── llm.py              # LLM interface
│   └── chatbot.py          # Chatbot implementation
└── utils/
    ├── __init__.py
    └── helpers.py          # Helper functions
```



> 🦙 **LLaMA 2 Chatbot for Library Services**  
> A lightweight AI assistant for library services using **LLaMA 2** with **RAG (Retrieval-Augmented Generation)**.  
> Built for the **Capstone Project** course by Arache, Arion, Danna, Dayang, Ryan.
