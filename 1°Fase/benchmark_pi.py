import time
import psutil
import ollama
import threading
import os
import glob
import subprocess

# Configurazione
MODEL_NAME = "phi:2.7b"  # Cambia con il modello che vuoi testare
PROMPT = "You are being used as part of a benchmark to evaluate Small Language Models (SLMs). Your task is to answer the prompt as accurately and clearly as possible. Rules: use a maximum of 120 words, write in clear and concise language, do not include explanations about your reasoning process, do not ask questions, and do not include any text before or after the answer. Answer format: 1. Definition 2. Main Uses 3. Hardware Features. Prompt: Explain briefly what a Raspberry Pi 5 is, what it is used for, and its main hardware features."

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
            # Questo restituisce il consumo dei vari rami del PMIC, li sommiamo per stimare il totale.
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
    """Estrae un valore in modo sicuro sia da dict che da oggetti pydantic (nuove versioni lib ollama)."""
    if isinstance(obj, dict):
        val = obj.get(key)
    else:
        val = getattr(obj, key, None)
    return default if val is None else val

def run_benchmark():
    print(f"--- Inizio Benchmark: {MODEL_NAME} ---")
    
    monitor = StatsMonitor()
    monitor_thread = threading.Thread(target=monitor.monitor)
    
    start_time = time.time()
    monitor_thread.start()

    # Chiamata API ufficiale a Ollama
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=PROMPT,
            think=False,
        )
        
        end_time = time.time()
        monitor.stop_flag = True
        monitor_thread.join()

        # Elaborazione Dati
        total_time = end_time - start_time
        load_duration = get_val(response, 'load_duration', 0) / 1_000_000_000 # nanosecondi -> secondi
        eval_count = get_val(response, 'eval_count', 0) # Token generati
        eval_duration = get_val(response, 'eval_duration', 1) / 1_000_000_000 # secondi
        
        tps = eval_count / eval_duration if eval_duration > 0 else 0

        # Mostra l'output dell'IA
        ai_response = get_val(response, 'response', 'Nessuna risposta disponibile')
        print(f"\n### RISPOSTA IA ###\n{ai_response.strip() if ai_response.strip() else 'Nessuna risposta generata'}\n")

        # Calcoli sicuri (prevengono ZeroDivisionError nel caso di esecuzione troppo rapida o fallimenti)
        cpu_avg = sum(monitor.cpu_usage) / len(monitor.cpu_usage) if monitor.cpu_usage else 0.0
        cpu_max = max(monitor.cpu_usage) if monitor.cpu_usage else 0.0
        mem_max = max(monitor.mem_usage) if monitor.mem_usage else 0.0
        pwr_avg = sum(monitor.power_usage) / len(monitor.power_usage) if monitor.power_usage else 0.0

        # Risultati
        print("\n### RISULTATI BENCHMARK ###")
        print(f"TEMPO DI CARICAMENTO:   {load_duration:.2f} s")
        print(f"TEMPO DI RISPOSTA TOT:  {total_time:.2f} s")
        print(f"TOKEN GENERATI:         {eval_count}")
        print(f"VELOCITÀ (TPS):         {tps:.2f} token/s")
        print("-" * 30)
        print(f"CPU USAGE (MED/MAX):    {cpu_avg:.1f}% / {cpu_max:.1f}%")
        print(f"MEMORY USAGE (MAX):     {mem_max:.2f} MB")
        print(f"CONSUMO ENERGETICO MED: {pwr_avg:.2f} W")
        print("-" * 30)

    except Exception as e:
        print(f"Errore durante il benchmark: {e}")
        monitor.stop_flag = True

if __name__ == "__main__":
    run_benchmark()
