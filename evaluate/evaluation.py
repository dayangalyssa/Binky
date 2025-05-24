"""
Implementasi evaluasi BLEU untuk chatbot library dengan pengukuran CPU usage dan latensi
"""
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import mlflow
import os
import json
import time
import psutil
import numpy as np


class LibraryBLEUEvaluator:
    def __init__(self, use_mlflow=True):
        self.smoothing = SmoothingFunction()
        self.test_dataset = self.create_library_test_dataset()
        self.use_mlflow = use_mlflow

        if self.use_mlflow:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("Library_Chatbot_CPU")
            print("MLflow tracking enabled with URI: http://localhost:5000")

    def create_library_test_dataset(self) -> List[Dict]:
        return [
            {
                "query": "Apa saja fasilitas yang terdapat di perpustakaan Universitas Brawijaya?",
                "reference_answer": "Perpustakaan Universitas Brawijaya memiliki berbagai fasilitas sepertinya 1. Ruang Baca 2. Ruang Diskusi Kelompok 3. Komputer Akses Internet 4. Sistem Penelusuran Koleksi Digital 5. WiFi 6. Mesin Fotokopi Mandiri 7. Scanner 8. Printer 9. Loker Penyimpanan Barang dan banyak lagi fasilitas yang dapat digunakan mahasiswa Universitas Brawijaya.",
            },
            {
                "query": "Bagaimana aturan atau SOP peminjaman buku di Universitas Brawijaya?",
                "reference_answer": "SOP peminjaman buku 1. Melakukan Scan KTM 2. Cek status data peminjam di APiklasi Inlislite 3. Scan bacode yang terdapat pada buku yan ingin dipinjam",
            },
            {
                "query": "Apa saja layanan referensi jurnal yang terdapat pada universitas brawijaya?",
                "reference_answer": "Perpustakaan UB menyediakan layanan seperti 1. ProQuest, 2. IEEE Xplore, 3. Science Direct, 4. Springer, 5. Nature, 6. Nature, 7. CAJ dan masih banyak lagi",
            },
            {
                "query": "Dimana kah letak rak buku yang berjudul 'Diabetes : terapi dan pencegahannya'?",
                "reference_answer": "Buku berjudul 'Diabetes : terapi dan pencegahannya' dapat ditemukan di rak buku 1a",
            },
            {
                "query": "Apakah saya dapat memperpanjang masa peminjaman buku saya ? jika iya, bagaimana cara dan tahapannya?",
                "reference_answer": "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut. Silahkan hubungi helpdesk kami di https://lib.ub.ac.id/.",
            },
            {
                "query": "Jelaskan apa itu layanan 'Klinik Journal' dalam Universitas Brawijaya?",
                "reference_answer": "Perpustakaan Universitas Brawijaya menyediakan layanan pendampingan eksklusif bagi mahasiswa S1, S2, dan S3 untuk mendukung proses penyelesaian artikel jurnal. Layanan ini berlokasi di Ruang Scholar Lounge, tempat yang dirancang khusus untuk memberikan bimbingan dari tahap awal hingga penyelesaian karya ilmiah. Dengan pendampingan yang profesional dan personal, layanan ini hadir untuk memastikan kualitas karya akademik pemustaka mencapai standar terbaik.",
            },
            {
                "query": "Apakah terdapat fasilitas 'UB Sport' dalam perpustakaan Universitas Brawijaya??",
                "reference_answer": "Berdasarkan dokumen yang tersedia, saya tidak dapat menemukan jawaban untuk pertanyaan Anda.",
            },
            {
                "query": "Bagaimana menghubungi Admin Help desk universitas brawijaya?",
                "reference_answer": "Untuk menghubungi Admin Help desk Universitas Brawijaya, Anda dapat mengunjungi situs web resmi perpustakaan di https://lib.ub.ac.id/ dan mencari informasi kontak yang tersedia.",
            }
        ]

    def run_bleu_evaluation(self, chatbot, config_params=None):
        results = {
            "individual_scores": [],
            "overall_scores": {
                "bleu_1": [],
                "bleu_2": [],
                "bleu_3": [],
                "bleu_4": [],
                "cpu_usage": [],
                "latency_ms": [],
            }
        }

        print("BLEU EVALUATION - LIBRARY CHATBOT")
        print("=" * 50)

        for i, test_case in enumerate(self.test_dataset):
            query = test_case["query"]
            reference = test_case["reference_answer"]

            print(f"\n Test {i+1}: '{query}'")

            # Mulai monitor CPU dan waktu
            start_cpu = psutil.cpu_percent(interval=None)
            start_time = time.time()

            # Dapatkan respon chatbot
            chatbot_result = chatbot.process_query(query)
            generated_response = chatbot_result["response"]

            # Hitung waktu dan CPU usage
            latency_ms = (time.time() - start_time) * 1000
            cpu_after = psutil.cpu_percent(interval=None)
            cpu_used = (start_cpu + cpu_after) / 2

            # Tokenize
            ref_tokens = reference.lower().split()
            gen_tokens = generated_response.lower().split()

            bleu_1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0))
            bleu_2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0))
            bleu_3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=self.smoothing.method1)

            individual_result = {
                "query": query,
                "reference": reference,
                "generated": generated_response,
                "bleu_1": bleu_1,
                "bleu_2": bleu_2,
                "bleu_3": bleu_3,
                "bleu_4": bleu_4,
                "cpu_usage": cpu_used,
                "latency_ms": latency_ms,
                "num_docs_retrieved": chatbot_result.get("num_docs_retrieved", 0)
            }

            results["individual_scores"].append(individual_result)

            for metric, value in zip(["bleu_1", "bleu_2", "bleu_3", "bleu_4"], [bleu_1, bleu_2, bleu_3, bleu_4]):
                results["overall_scores"][metric].append(value)

            results["overall_scores"]["cpu_usage"].append(cpu_used)
            results["overall_scores"]["latency_ms"].append(latency_ms)

            print(f"   BLEU-1: {bleu_1:.3f}")
            print(f"   BLEU-2: {bleu_2:.3f}")
            print(f"   BLEU-3: {bleu_3:.3f}")
            print(f"   BLEU-4: {bleu_4:.3f}")
            print(f"   CPU usage: {cpu_used:.2f}%")
            print(f"   Latency: {latency_ms:.2f} ms")
            print(f"   Response preview: {generated_response[:100]}...")

        results["summary"] = self.calculate_summary_scores(results)

        if self.use_mlflow and config_params:
            self._log_to_mlflow(results, config_params)

        return results

    def calculate_summary_scores(self, results: Dict) -> Dict:
        summary = {}

        for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "cpu_usage", "latency_ms"]:
            summary[f"avg_{key}"] = np.mean(results["overall_scores"][key])
            summary[f"std_{key}"] = np.std(results["overall_scores"][key])

        return {
            "overall": summary,
            "total_tests": len(results["individual_scores"])
        }

    def print_evaluation_report(self, results: Dict):
        print("\n BLEU EVALUATION SUMMARY")
        print("=" * 40)

        summary = results["summary"]["overall"]
        print(f"   Avg BLEU-1: {summary['avg_bleu_1']:.3f} (±{summary['std_bleu_1']:.3f})")
        print(f"   Avg BLEU-2: {summary['avg_bleu_2']:.3f} (±{summary['std_bleu_2']:.3f})")
        print(f"   Avg BLEU-3: {summary['avg_bleu_3']:.3f} (±{summary['std_bleu_3']:.3f})")
        print(f"   Avg BLEU-4: {summary['avg_bleu_4']:.3f} (±{summary['std_bleu_4']:.3f})")
        print(f"   Avg CPU usage: {summary['avg_cpu_usage']:.2f}% (±{summary['std_cpu_usage']:.2f})")
        print(f"   Avg Latency: {summary['avg_latency_ms']:.2f} ms (±{summary['std_latency_ms']:.2f})")

        avg_bleu_4 = summary["avg_bleu_4"]
        if avg_bleu_4 >= 0.5:
            interpretation = "EXCELLENT - Response quality sangat baik"
        elif avg_bleu_4 >= 0.3:
            interpretation = "GOOD - Response quality baik"
        elif avg_bleu_4 >= 0.2:
            interpretation = "FAIR - Response quality cukup, perlu improvement"
        else:
            interpretation = "POOR - Response quality perlu perbaikan signifikan"

        print(f"\n🎯 Overall Interpretation: {interpretation}")

    def _log_to_mlflow(self, results, config_params):
        """Log hasil evaluasi ke MLflow"""
        with mlflow.start_run():
            # Log BLEU metrics
            summary = results["summary"]["overall"]
            mlflow.log_metric("bleu_1", summary["avg_bleu_1"])
            mlflow.log_metric("bleu_2", summary["avg_bleu_2"])
            mlflow.log_metric("bleu_3", summary["avg_bleu_3"])
            mlflow.log_metric("bleu_4", summary["avg_bleu_4"])
            mlflow.log_metric("avg_cpu_usage", summary["avg_cpu_usage"])
            mlflow.log_metric("avg_latency_ms", summary["avg_latency_ms"])
            
            # Log model parameters
            for param_name, param_value in config_params.items():
                mlflow.log_param(param_name.lower(), param_value)
            
            # Save results to JSON and log as artifact
            evaluation_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results")
            os.makedirs(evaluation_dir, exist_ok=True)
            results_path = os.path.join(evaluation_dir, "bleu_evaluation_results.json")
            
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n Results saved to: {results_path}")
            
            # Log artifact
            mlflow.log_artifact(results_path)
            
            print("Results logged to MLflow!")
