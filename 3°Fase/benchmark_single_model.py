import time
import psutil
import ollama
import threading
import os
import sys
import glob
import subprocess
import re
import argparse
import random

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

def generate_commands(n=100):
    commands = []
    actions = ["moving forward", "going down", "going up", "rotating right", "moving backward", "rotating left"]
    for _ in range(n):
        tipo = random.randint(1, 5)
        if tipo == 1:
            cmd = f"Move {random.choice(['forward', 'backward', 'left', 'right'])} {random.randint(5, 100)} steps."
        elif tipo == 2:
            cmd = f"Go {random.choice(['up', 'down'])} {random.randint(10, 50)} and rotate {random.choice(['left', 'right'])} by {random.choice([45, 90, 180])} degrees."
        elif tipo == 3:
            cmd = f"Warning: obstacle detected at {random.randint(2, 30)}cm in {random.choice(['front', 'left', 'right'])}."
        elif tipo == 4:
            cmd = f"current action: {random.choice(actions)}. object_{random.choice(['left', 'right', 'front', 'rear'])}: {random.randint(2, 20)}cm."
        else:
            cmd = f"Alert: multiple objects. object_front: {random.randint(5,15)}cm, object_{random.choice(['left', 'right'])}: {random.randint(5,15)}cm."
        commands.append(cmd)
    return commands

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

def run_test_sequence(model_name, test_commands, persistent_context=False):
    context_type = "CONTESTO PERSISTENTE" if persistent_context else "CONTESTO RESETTATO"
    print(f"\n--- Inizio Test: {context_type} ({len(test_commands)} comandi) ---")
    
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

    for i, cmd in enumerate(test_commands):
        print(f"Esecuzione comando {i+1}/{len(test_commands)}: {cmd}")
        
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
            ai_response = re.sub(r'<think>.*?</think>\n*', '', ai_response, flags=re.DOTALL).strip()
            
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

    avg_tps = sum(r['tps'] for r in results) / len(results) if results else 0
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    avg_cpu = sum(r['cpu'] for r in results) / len(results) if results else 0
    max_mem = max(r['mem'] for r in results) if results else 0
    avg_pwr = sum(r['pwr'] for r in results) / len(results) if results else 0

    print(f"\n### RIEPILOGO {context_type} ###")
    if warmup_load_time > 0.01:
        print(f"TEMPO CARICAMENTO INIZIALE: {warmup_load_time:.2f} s")
    print(f"TPS MEDIO:         {avg_tps:.2f}")
    print(f"TEMPO MEDIO:       {avg_time:.2f} s")
    print(f"CPU MEDIA:         {avg_cpu:.1f}%")
    print(f"MEMORIA MAX:       {max_mem:.2f} MB")
    print(f"POWER MEDIO:       {avg_pwr:.2f} W")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Esegui test multipli su un singolo modello.")
    parser.add_argument("model", help="Il nome del modello da testare (es. phi:2.7b)")
    parser.add_argument("--count", type=int, default=100, help="Numero di comandi da generare e testare (default: 100)")
    args = parser.parse_args()

    model = args.model
    count = args.count

    os.makedirs("output", exist_ok=True)
    
    # Generiamo i comandi casuali in anticipo per usare gli stessi in entrambi i test (resettato/persistente)
    test_commands = generate_commands(count)

    safe_model_name = model.replace(":", "_").replace("/", "_")
    log_file = os.path.join("output", f"{safe_model_name}_{count}_tests.txt")
    
    original_stdout = sys.stdout
    logger = MultiLogger(log_file)
    sys.stdout = logger
    
    try:
        print(f"==================================================")
        print(f" AVVIO BENCHMARK SINGOLO MODELLO: {model.upper()}")
        print(f" NUMERO COMANDI: {count}")
        print(f"==================================================")
        
        # Test con contesto resettato
        run_test_sequence(model, test_commands, persistent_context=False)
        
        # Test con contesto persistente
        run_test_sequence(model, test_commands, persistent_context=True)
        
        print(f"Test completato per {model}.\n")
    except Exception as e:
        print(f"Errore fatale: {e}")
    finally:
        print(f"---\nScaricamento modello {model} dalla VRAM per evitare blocchi...")
        try:
            ollama.generate(model=model, prompt='', keep_alive=0)
        except Exception:
            pass
        time.sleep(3)
        
        sys.stdout = original_stdout
        logger.log.close()
        print(f"Output salvato in -> {log_file}\n")

if __name__ == "__main__":
    main()
