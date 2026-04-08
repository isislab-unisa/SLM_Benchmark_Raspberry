import time
import psutil
import ollama
import threading
import os
import glob
import subprocess
import csv

# Configurazione

MODELS = ["phi:2.7b", "llama3.1:8b", "qwen3.5:2b","deepseek-r1:8b","alibayram/smollm3:latest"]  # Aggiungi qui i modelli che vuoi testare
PROMPT = "You are being used as part of a benchmark to evaluate Small Language Models (SLMs). Your task is to answer the prompt as accurately and clearly as possible. Rules: use a maximum of 120 words, write in clear and concise language, do not include explanations about your reasoning process, do not ask questions, and do not include any text before or after the answer. Answer format: 1. Definition 2. Main Uses 3. Hardware Features. Prompt: Explain briefly what a Raspberry Pi 5 is, what it is used for, and its main hardware features."
CSV_FILE = "benchmark_results.csv"

class StatsMonitor:
    def __init__(self):
        self.stop_flag = False
        self.cpu_usage = []
        self.mem_usage = []
        self.power_usage = []

    def get_power(self):
        # Percorsi specifici ed utility per verificare il consumo energetico su Raspberry Pi 5
        try:
            # 1. Prova vcgencmd pmic_read_adc (Specifico per Raspberry Pi 5)
            result = subprocess.run(['vcgencmd', 'pmic_read_adc'], capture_output=True, text=True)
            if result.returncode == 0:
                currents = {}
                volts = {}
                for line in result.stdout.splitlines():
                    if "=" not in line: continue
                    left, right = line.split("=", 1)
                    name_part = left.split()[0]
                    val_str = right.replace("A", "").replace("V", "").strip()
                    try:
                        val = float(val_str)
                        if name_part.endswith("_V"):
                            base_name = name_part[:-2]
                            volts[base_name] = val
                        elif name_part.endswith("_A") or name_part.endswith("_I"):
                            base_name = name_part[:-2]
                            currents[base_name] = val
                    except:
                        pass
                
                power_w = 0.0
                for base in currents:
                    if base in volts:
                        power_w += currents[base] * volts[base]
                if power_w > 0:
                    return power_w

            # 2. Prova hwmon (power1_input) - base kernel standard uW
            paths = glob.glob("/sys/class/hwmon/hwmon*/power1_input")
            if paths:
                with open(paths[0], "r") as f:
                    return int(f.read()) / 1_000_000  # Converte in Watt
                    
            # 3. Prova power_supply (es. rpi-sys-vdd-5v)
            paths = glob.glob("/sys/class/power_supply/*/power_now")
            if paths:
                with open(paths[0], "r") as f:
                    return int(f.read()) / 1_000_000

        except:
            pass
            
        return 0.0

    def monitor(self):
        while not self.stop_flag:
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
            self.mem_usage.append(psutil.virtual_memory().used / (1024 * 1024)) # MB
            self.power_usage.append(self.get_power())

def get_val(obj, key, default):
    """Estrae un valore in modo sicuro sia da dict che da oggetti pydantic."""
    if isinstance(obj, dict):
        val = obj.get(key)
    else:
        val = getattr(obj, key, None)
    return default if val is None else val

def run_benchmark(model_name):
    print(f"\n>>> Avvio Benchmark per: {model_name}")
    
    monitor = StatsMonitor()
    monitor_thread = threading.Thread(target=monitor.monitor)
    
    start_time = time.time()
    monitor_thread.start()

    try:
        # keep_alive=0 scarica il modello immediatamente dopo la risposta
        response = ollama.generate(
            model=model_name,
            prompt=PROMPT,
            think=False,
            keep_alive=0
        )
        
        end_time = time.time()
        monitor.stop_flag = True
        monitor_thread.join()

        # Elaborazione Dati
        total_time = end_time - start_time
        load_duration = get_val(response, 'load_duration', 0) / 1_000_000_000
        eval_count = get_val(response, 'eval_count', 0)
        eval_duration = get_val(response, 'eval_duration', 1) / 1_000_000_000
        tps = eval_count / eval_duration if eval_duration > 0 else 0

        # Statistiche monitoraggio
        cpu_avg = sum(monitor.cpu_usage) / len(monitor.cpu_usage) if monitor.cpu_usage else 0.0
        cpu_max = max(monitor.cpu_usage) if monitor.cpu_usage else 0.0
        mem_max = max(monitor.mem_usage) if monitor.mem_usage else 0.0
        pwr_avg = sum(monitor.power_usage) / len(monitor.power_usage) if monitor.power_usage else 0.0

        print(f"Completato: {tps:.2f} t/s, Memoria Max: {mem_max:.2f} MB")

        return {
            "Modello": model_name,
            "Tempo Caricamento (s)": round(load_duration, 2),
            "Tempo Totale (s)": round(total_time, 2),
            "Token Generati": eval_count,
            "TPS (Token/s)": round(tps, 2),
            "CPU Media (%)": round(cpu_avg, 1),
            "CPU Max (%)": round(cpu_max, 1),
            "Memoria Max (MB)": round(mem_max, 2),
            "Potenza Media (W)": round(pwr_avg, 2)
        }

    except Exception as e:
        print(f"Errore durante il benchmark di {model_name}: {e}")
        monitor.stop_flag = True
        if monitor_thread.is_alive():
            monitor_thread.join()
        return None

if __name__ == "__main__":
    # Intestazioni per il file CSV
    fieldnames = [
        "Modello", "Tempo Caricamento (s)", "Tempo Totale (s)", 
        "Token Generati", "TPS (Token/s)", "CPU Media (%)", 
        "CPU Max (%)", "Memoria Max (MB)", "Potenza Media (W)"
    ]

    # Verifica se il file esiste già per gestire l'header
    file_exists = os.path.isfile(CSV_FILE)

    for model in MODELS:
        result = run_benchmark(model)
        if result:
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(result)
            
            # Pausa tecnica tra un modello e l'altro
            time.sleep(2)

    print(f"\n>>> Benchmark completato. Risultati salvati in: {CSV_FILE}")
