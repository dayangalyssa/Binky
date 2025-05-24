# run_evaluation.py

import sys
import os
# Tambahkan path ke folder evaluate
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Tambahkan path ke root project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import LibraryBLEUEvaluator
from config import MODEL_CONFIG
from src.chatbot import LlamaRagChatbot
from utils.helpers import setup_rag_pipeline

def run_current_config(chatbot):
    """
    Run evaluasi dengan config saat ini
    """
    print(f" Running evaluation with config: {MODEL_CONFIG}")
    
    # Run evaluation dengan MLflow tracking
    evaluator = LibraryBLEUEvaluator(use_mlflow=True)
    results = evaluator.run_bleu_evaluation(chatbot, MODEL_CONFIG)
    evaluator.print_evaluation_report(results)
    
    # Print summary
    bleu_4 = results["summary"]["overall"]["avg_bleu_4"]
    print(f"\n RESULT: BLEU-4 = {bleu_4:.3f}")
    
    return results

# Usage
if __name__ == "__main__":
    print("Setting up RAG pipeline for evaluation...")
    success, message, vectorstore = setup_rag_pipeline()
    if not success:
        print("Error setting up RAG pipeline:", message)
        sys.exit(1)
    
    print("Creating chatbot instance...")
    chatbot = LlamaRagChatbot(vectorstore)
    
    print("Starting evaluation...")
    results = run_current_config(chatbot)
    print("Evaluation complete!")