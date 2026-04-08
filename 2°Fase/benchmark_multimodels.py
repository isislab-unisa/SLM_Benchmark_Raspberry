import time
import psutil
import ollama
import threading
import os
import sys
import glob
import subprocess
import re

# Configurazione Modelli
MODELS_TO_TEST = [
    "phi:2.7b", "llama3.1:8b", "qwen3.5:2b","deepseek-r1:8b","alibayram/smollm3:latest","gemma3:4b"
]

SYSTEM_PROMPT = """Role: Autonomous navigation AI.
Task: Calculate evasive movements.
Rules:
1. Output ONLY command codes and integer values. No intro, no text.
2. Command codes: U(Up), D(Down), L(Left), R(Right), RL(Rotate Left), RR(Rotate Right), F(Forward), B(Backward).
3. Logic: Obstacle FRONT -> B or RR. Obstacle LEFT -> R. Obstacle RIGHT -> L.
4. Format: <CODE><VALUE>. Separate multiple commands with one space.

Examples:
Input: object_front: 10cm
Output: B15 RR90

Input: Warning: obstacle detected on the left at 5cm.
Output: R20

Input: object_right: 8cm. current action: moving forward.
Output: L15 F10"""

TEST_COMMANDS = [
    "Move forward 20 steps.",
    "Go down 15 and rotate left by 90 degrees.",
    "Turn right 45, then move backward 10.",
    "Go up 50, move forward 100, and rotate right 180.",
    "Move left 20 and then right 20.",
    "Rotate left 90, go backward 5.",
    "Go down 50 steps.",
    "Move right 15, then move forward 30.",
    "Go up 10 and turn left 45.",
    "Move backward 50, rotate left 180, and move left 10.",
    "Warning: obstacle detected at 15cm in front.",
    "current action: moving forward. object_right: 8cm.",
    "Object on the left at 5cm. current action: going down.",
    "object_rear: 20cm.",
    "Alert: multiple objects. object_front: 10cm, object_left: 10cm.",
    "current action: rotating right. object_front: 5cm.",
    "Sensor 3 reports object detected at 50cm on the right.",
    "object_left: 2cm. immediate evasion required.",
    "current action: going up. object_front: 12cm.",
    "Obstacle closing in. object_rear: 5cm."
]

class StatsMonitor:
    def __init__(self):
        self.stop_flag = False
        self.cpu_usage = []
        self.mem_usage = []
        self.power_usage = []

    def get_power(self):
        try:
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

            paths = glob.glob("/sys/class/hwmon/hwmon*/power1_input")
            if paths:
                with open(paths[0], "r") as f:
                    return int(f.read()) / 1_000_000
                    
            paths = glob.glob("/sys/class/power_supply/*/power_now")
            if paths:
                with open(paths[0], "r") as f:
                    return int(f.read()) / 1_000_000
        except:
            pass
        return 0.0

    def monitor(self):
        while not self.stop_flag:
            self.cpu_usage.append(psutil.cpu_percent(interval=0.5))
            self.mem_usage.append(psutil.virtual_memory().used / (1024 * 1024))
            self.power_usage.append(self.get_power())

def get_val(obj, key, default):
    if isinstance(obj, dict):
        val = obj.get(key)
    else:
        val = getattr(obj, key, None)
    return default if val is None else val

class MultiLogger:
    """Classe personalizzata per scrivere simultaneamente su terminale e su file di log."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_test_sequence(model_name, persistent_context=False):
    context_type = "CONTESTO PERSISTENTE" if persistent_context else "CONTESTO RESETTATO"
    print(f"\n--- Inizio Test: {context_type} ---")
    
    # --- WARMUP / PRE-LOAD ---
    print(f"-> [Warm-up] Caricamento modello in memoria in corso...")
    try:
        warmup_resp = ollama.chat(
            model=model_name, 
            messages=[{"role": "user", "content": "ready?"}],
            options={"num_predict": 5},
            think=False
        )
        warmup_load_time = get_val(warmup_resp, 'load_duration', 0) / 1_000_000_000
    except Exception as e:
        print(f"Errore durante il caricamento: {e}")
        warmup_load_time = 0
    print(f"-> [Warm-up] Modello pronto. Tempo di caricamento: {warmup_load_time:.2f} s\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []

    for i, cmd in enumerate(TEST_COMMANDS):
        print(f"Esecuzione comando {i+1}/20: {cmd}")
        
        if not persistent_context:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        messages.append({"role": "user", "content": cmd})
        
        monitor = StatsMonitor()
        monitor_thread = threading.Thread(target=monitor.monitor)
        
        start_time = time.time()
        monitor_thread.start()

        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={"num_predict": 40, "temperature": 0.0},
                think=False
            )
            
            end_time = time.time()
            monitor.stop_flag = True
            monitor_thread.join()

            total_time = end_time - start_time
            eval_count = get_val(response, 'eval_count', 0)
            eval_duration = get_val(response, 'eval_duration', 1) / 1_000_000_000
            load_duration = get_val(response, 'load_duration', 0) / 1_000_000_000
            
            tps = eval_count / eval_duration if eval_duration > 0 else 0
            pure_time = total_time - load_duration
            
            ai_response = get_val(get_val(response, 'message', {}), 'content', '').strip()
            # Rimuove il contenuto del thinking generato dai modelli come Deepseek
            ai_response = re.sub(r'<think>.*?</think>\n*', '', ai_response, flags=re.DOTALL).strip()
            
            # Se persistente, aggiungi la risposta al contesto
            if persistent_context:
                messages.append({"role": "assistant", "content": ai_response})

            cpu_avg = sum(monitor.cpu_usage) / len(monitor.cpu_usage) if monitor.cpu_usage else 0.0
            mem_max = max(monitor.mem_usage) if monitor.mem_usage else 0.0
            pwr_avg = sum(monitor.power_usage) / len(monitor.power_usage) if monitor.power_usage else 0.0

            results.append({
                "tps": tps,
                "time": pure_time,
                "load_time": load_duration,
                "response": ai_response,
                "cpu": cpu_avg,
                "mem": mem_max,
                "pwr": pwr_avg
            })
            
            print(f"Risposta: {ai_response}")
            if load_duration > 0.01:
                print(f"Performance: {tps:.2f} t/s | Time: {pure_time:.2f}s (Load: {load_duration:.2f}s)")
            else:
                print(f"Performance: {tps:.2f} t/s | Time: {pure_time:.2f}s")

        except Exception as e:
            print(f"Errore: {e}")
            monitor.stop_flag = True
            if monitor_thread.is_alive():
                monitor_thread.join()

    # Riepilogo finale per la sequenza
    avg_tps = sum(r['tps'] for r in results) / len(results) if results else 0
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_cpu = sum(r['cpu'] for r in results) / len(results) if results else 0
    max_mem = max(r['mem'] for r in results) if results else 0
    avg_pwr = sum(r['pwr'] for r in results) / len(results) if results else 0

    print(f"\n### RIEPILOGO {context_type} ###")
    if warmup_load_time > 0.01:
        print(f"TEMPO CARICAMENTO: {warmup_load_time:.2f} s")
    print(f"TPS MEDIO:         {avg_tps:.2f}")
    print(f"TEMPO MEDIO:       {avg_time:.2f} s")
    print(f"CPU MEDIA:         {avg_cpu:.1f}%")
    print(f"MEMORIA MAX:       {max_mem:.2f} MB")
    print(f"POWER MEDIO:       {avg_pwr:.2f} W")
    print("-" * 40)

if __name__ == "__main__":
    # Assicurati che esista la cartella per raccogliere i log
    os.makedirs("output", exist_ok=True)
    
    for model in MODELS_TO_TEST:
        # Pulisce i due punti che non possono essere inclusi nei nomi file su Windows/Linux
        safe_model_name = model.replace(":", "_")
        safe_model_name = safe_model_name.replace("/", "_")
        log_file = os.path.join("output", f"{safe_model_name}_results.txt")
        
        # Sostituiamo sys.stdout per sdoppiare il flusso
        original_stdout = sys.stdout
        logger = MultiLogger(log_file)
        sys.stdout = logger
        
        try:
            print(f"==================================================")
            print(f" AVVIO BENCHMARK PER IL MODELLO: {model.upper()}")
            print(f"==================================================")
            
            # Esegue prima il test con contesto resettato
            run_test_sequence(model, persistent_context=False)
            
            # Esegue poi il test con contesto persistente
            run_test_sequence(model, persistent_context=True)
            
            print(f"Test completato per {model}.\n")
        except Exception as e:
            print(f"Errore fatale: {e}")
        finally:
            print(f"---\nScaricamento modello {model} dalla VRAM per evitare blocchi...")
            try:
                # Forza il rilascio della VRAM
                ollama.generate(model=model, prompt='', keep_alive=0)
            except Exception:
                pass
            time.sleep(3) # Pausa di sicurezza per il Raspberry Pi
            
            # Ripristina il flusso standard al termine del modello e chiude il file
            sys.stdout = original_stdout
            logger.log.close()
            print(f"Output salvato in -> {log_file}\n")
