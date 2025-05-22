"""
Script untuk mengecek dan mendownload model Llama yang benar
"""
import os
from huggingface_hub import snapshot_download, list_repo_files
from pathlib import Path

def check_and_fix_model():
    """Cek model yang ada dan download yang benar jika perlu"""
    
    model_id = "TheBloke/Llama-2-7B-Chat-GGUF"
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    print("🔍 Mengecek file model yang tersedia...")
    
    try:
        # List semua file di repository
        files = list_repo_files(model_id)
        gguf_files = [f for f in files if f.endswith('.gguf')]
        
        print(f"📁 File GGUF yang tersedia di repository:")
        for i, file in enumerate(gguf_files, 1):
            print(f"  {i}. {file}")
        
        # Cek folder cache lokal
        model_cache_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
        
        if os.path.exists(model_cache_path):
            print(f"\n📂 Model cache ditemukan di: {model_cache_path}")
            
            # Cari snapshot folder
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_folders = os.listdir(snapshots_dir)
                if snapshot_folders:
                    latest_snapshot = os.path.join(snapshots_dir, snapshot_folders[0])
                    print(f"📁 Snapshot folder: {latest_snapshot}")
                    
                    # List file yang ada di snapshot
                    existing_files = os.listdir(latest_snapshot)
                    existing_gguf = [f for f in existing_files if f.endswith('.gguf')]
                    
                    print(f"\n💾 File GGUF yang sudah ada di cache:")
                    for file in existing_gguf:
                        file_path = os.path.join(latest_snapshot, file)
                        file_size = os.path.getsize(file_path) / (1024*1024*1024)
                        print(f"  - {file} ({file_size:.2f} GB)")
                    
                    if existing_gguf:
                        print(f"\n✅ Gunakan file ini dalam config.py:")
                        print(f"MODEL_FILE = \"{existing_gguf[0]}\"")
                        return existing_gguf[0]
        
        print(f"\n⬇️ Mendownload model yang direkomendasikan...")
        
        # Download file yang direkomendasikan untuk CPU
        recommended_files = [
            "llama-2-7b-chat.Q4_K_M.gguf",  # Kualitas bagus, ukuran sedang
            "llama-2-7b-chat.Q4_0.gguf",    # Lebih kecil
            "llama-2-7b-chat.Q5_K_M.gguf"   # Kualitas lebih bagus
        ]
        
        for filename in recommended_files:
            if filename in gguf_files:
                print(f"📥 Mendownload {filename}...")
                snapshot_download(
                    repo_id=model_id,
                    allow_patterns=[filename],
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                print(f"✅ {filename} berhasil didownload!")
                print(f"\n🔧 Update config.py dengan:")
                print(f"MODEL_FILE = \"{filename}\"")
                return filename
        
        print("❌ Tidak ada file yang cocok ditemukan")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def create_download_script():
    """Buat script untuk download manual"""
    script_content = '''
import os
from huggingface_hub import snapshot_download

def download_llama_model():
    """Download model Llama GGUF"""
    model_id = "TheBloke/Llama-2-7B-Chat-GGUF"
    
    # Pilih salah satu file ini (uncomment yang diinginkan):
    
    # Untuk CPU dengan RAM terbatas (< 8GB)
    filename = "llama-2-7b-chat.Q4_0.gguf"  # ~3.5 GB
    
    # Untuk CPU dengan RAM sedang (8-16GB) - RECOMMENDED
    # filename = "llama-2-7b-chat.Q4_K_M.gguf"  # ~4.1 GB
    
    # Untuk CPU dengan RAM besar (>16GB)
    # filename = "llama-2-7b-chat.Q5_K_M.gguf"  # ~4.8 GB
    
    print(f"Downloading {filename}...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[filename],
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            local_files_only=False
        )
        print(f"✅ {filename} downloaded successfully!")
        print(f"\\n🔧 Update your config.py with:")
        print(f'MODEL_FILE = "{filename}"')
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")

if __name__ == "__main__":
    download_llama_model()
'''
    
    with open("download_model.py", "w") as f:
        f.write(script_content)
    
    print("📝 Script download_model.py telah dibuat!")
    print("   Jalankan dengan: python download_model.py")

if __name__ == "__main__":
    result = check_and_fix_model()
    
    if not result:
        print("\n📝 Membuat script download manual...")
        create_download_script()
        print("\n💡 Langkah selanjutnya:")
        print("1. Jalankan: python download_model.py")
        print("2. Update MODEL_FILE di config.py sesuai output")
        print("3. Jalankan aplikasi Anda lagi")