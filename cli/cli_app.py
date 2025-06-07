import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import LlamaRagChatbot
from utils.helpers import setup_rag_pipeline, time_function
import config

def run_cli_chatbot(language: str):
    """Run the chatbot in CLI interactive mode."""
    # Setup pipeline
    success, message, vectorstore = setup_rag_pipeline()
    print(f"\n[INFO] {message}")
    if not success:
        sys.exit(1)

    # Initialize chatbot
    chatbot = LlamaRagChatbot(vectorstore)
    chatbot.set_language(language)
    print(f"[INFO] Bahasa output disetel ke: {language}\n")

    # CLI interface
    print("🦙  Llama RAG Chatbot - CLI Mode")
    print("Ketik 'exit' atau 'quit' untuk keluar\n")

    while True:
        try:
            user_input = input("Anda: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Terima kasih telah menggunakan Binky. Sampai jumpa!")
                break
            if not user_input:
                continue
            print("\n⏳ Memproses pertanyaan Anda...")
            result = time_function(chatbot.process_query)(user_input)
            print("\n🦙 Binky:", result["response"])
            print(f"📄 ({result['num_docs_retrieved']} dokumen relevan ditemukan)")
            print("\n---")
        except KeyboardInterrupt:
            print("\n👋 Keluar dari sesi. Sampai jumpa!")
            break

def main():
    parser = argparse.ArgumentParser(description="Llama RAG Chatbot CLI")
    parser.add_argument(
        "--language",
        choices=["bahasa_indonesia", "english"],
        default="bahasa_indonesia",
        help="Pilih bahasa output (default: bahasa_indonesia)"
    )
    args = parser.parse_args()
    run_cli_chatbot(args.language)

if __name__ == "__main__":
    main()