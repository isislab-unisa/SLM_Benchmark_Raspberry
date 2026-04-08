import time
import psutil
import ollama
import threading
import os
import glob
import subprocess
import json

# Configurazione
MODEL_NAME = "phi:2.7b"  # Cambia con il modello che vuoi testare
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
            self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
            self.mem_usage.append(psutil.virtual_memory().used / (1024 * 1024))
            self.power_usage.append(self.get_power())

def get_val(obj, key, default):
    if isinstance(obj, dict):
        val = obj.get(key)
    else:
        val = getattr(obj, key, None)
    return default if val is None else val

def run_test_sequence(persistent_context=False):
    context_type = "CONTESTO PERSISTENTE" if persistent_context else "CONTESTO RESETTATO"
    print(f"\n--- Inizio Test: {context_type} ---")
    
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
                model=MODEL_NAME,
                messages=messages
            )
            
            end_time = time.time()
            monitor.stop_flag = True
            monitor_thread.join()

            total_time = end_time - start_time
            eval_count = get_val(response, 'eval_count', 0)
            eval_duration = get_val(response, 'eval_duration', 1) / 1_000_000_000
            tps = eval_count / eval_duration if eval_duration > 0 else 0
            
            ai_response = get_val(get_val(response, 'message', {}), 'content', '').strip()
            
            # Se persistente, aggiungi la risposta al contesto
            if persistent_context:
                messages.append({"role": "assistant", "content": ai_response})

            cpu_avg = sum(monitor.cpu_usage) / len(monitor.cpu_usage) if monitor.cpu_usage else 0.0
            mem_max = max(monitor.mem_usage) if monitor.mem_usage else 0.0
            pwr_avg = sum(monitor.power_usage) / len(monitor.power_usage) if monitor.power_usage else 0.0

            results.append({
                "tps": tps,
                "time": total_time,
                "response": ai_response,
                "cpu": cpu_avg,
                "mem": mem_max,
                "pwr": pwr_avg
            })
            
            print(f"Risposta: {ai_response}")
            print(f"Performance: {tps:.2f} t/s | {total_time:.2f}s")

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
    print(f"TPS MEDIO:         {avg_tps:.2f}")
    print(f"TEMPO MEDIO:       {avg_time:.2f} s")
    print(f"CPU MEDIA:         {avg_cpu:.1f}%")
    print(f"MEMORIA MAX:       {max_mem:.2f} MB")
    print(f"POWER MEDIO:       {avg_pwr:.2f} W")
    print("-" * 40)

if __name__ == "__main__":
    print(f"Modello in test: {MODEL_NAME}")
    # Esegue prima il test con contesto resettato
    run_test_sequence(persistent_context=False)
    # Esegue poi il test con contesto persistente
    run_test_sequence(persistent_context=True)
