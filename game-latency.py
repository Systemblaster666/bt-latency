#!/usr/bin/env python3
"""
game-latency.py — Measure real in-game audio latency.

Records loopback (digital reference) + microphone (acoustic output) simultaneously.
Runs 5 trials — each trial beeps, you fire one shot, it measures the delay.
Results are averaged for accuracy.

Usage:
    python3 game-latency.py

Hold the mic against the headphone ear cup throughout.
Fire ONE shot per beep.
"""

import os
import sys
import time
import threading
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
from scipy.signal import correlate

SAMPLE_RATE   = 48000
NUM_TRIALS    = 5
TRIAL_SECS    = 3.0    # recording window per trial
BEEP_OFFSET   = 0.4    # seconds after recording start when beep fires
ANALYSIS_SKIP = 0.6    # skip first N seconds before cross-correlating (past the beep)
INTER_TRIAL   = 1.5    # pause between trials
CONFIDENCE_MIN = 10


# ── Device listing ────────────────────────────────────────────────────────────

def pactl_devices(kind):
    out = subprocess.check_output(['pactl', 'list', f'{kind}s'], text=True)
    devices, name, desc = [], None, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Name:'):
            name = line.split(':', 1)[1].strip()
        elif line.startswith('Description:'):
            desc = line.split(':', 1)[1].strip()
            if name and desc:
                devices.append((name, desc))
            name = desc = None
    return devices

def sinks():
    return pactl_devices('sink')

def sources_no_monitor():
    return [(n, d) for n, d in pactl_devices('source') if '.monitor' not in n]

def find_monitor(sink_name):
    monitor_name = sink_name + '.monitor'
    for name, desc in pactl_devices('source'):
        if name == monitor_name:
            return name, desc
    return monitor_name, f'Monitor of {sink_name}'


# ── UI ────────────────────────────────────────────────────────────────────────

def pick(devices, label):
    print(f"\n{label}:")
    for i, (_, desc) in enumerate(devices):
        print(f"  [{i}] {desc}")
    while True:
        try:
            choice = int(input(f"Select [0-{len(devices)-1}]: ").strip())
            if 0 <= choice < len(devices):
                return devices[choice]
        except (ValueError, EOFError):
            pass
        except KeyboardInterrupt:
            sys.exit(0)


# ── Audio cue ─────────────────────────────────────────────────────────────────

def beep(freq=880, duration=0.12):
    t   = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sig *= np.hanning(len(sig))
    try:
        sd.play(sig, samplerate=SAMPLE_RATE, blocking=True)
    except Exception:
        pass


# ── Recording ─────────────────────────────────────────────────────────────────

def record_timed(source_name, filepath, duration):
    proc = subprocess.Popen(
        ['pw-record', '--target', source_name,
         '--rate', str(SAMPLE_RATE),
         '--channels', '1',
         '--format', 'f32',
         filepath],
        stderr=subprocess.DEVNULL
    )
    time.sleep(duration)
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Analysis ──────────────────────────────────────────────────────────────────

def load_f32(path):
    return np.fromfile(path, dtype=np.float32)

def parabolic_peak_offset(y, idx):
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom

def measure_latency(loopback, mic):
    lb = loopback / (np.max(np.abs(loopback)) + 1e-10)
    m  = mic      / (np.max(np.abs(mic))      + 1e-10)
    corr     = correlate(m, lb, mode='full')
    corr_abs = np.abs(corr)
    peak_idx = int(np.argmax(corr_abs))
    lag      = peak_idx - (len(lb) - 1) + parabolic_peak_offset(corr_abs, peak_idx)
    confidence = float(np.max(corr_abs)) / (float(np.median(corr_abs)) + 1e-9)
    return (lag / SAMPLE_RATE) * 1000, confidence

def describe(ms):
    if ms < 0:
        return "Negative — check device selection"
    elif ms < 20:
        return "Wired-equivalent — imperceptible"
    elif ms < 50:
        return "Excellent — competitive grade"
    elif ms < 100:
        return "Good — fine for most gaming"
    elif ms < 250:
        return "Moderate — noticeable in rhythm games"
    else:
        return "High — significant BT or software buffering"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_trial(mon_name, in_name, tmp):
    lb_file  = os.path.join(tmp, 'lb.f32')
    mic_file = os.path.join(tmp, 'mic.f32')

    # Start recording first
    t_lb  = threading.Thread(target=record_timed, args=(mon_name, lb_file,  TRIAL_SECS))
    t_mic = threading.Thread(target=record_timed, args=(in_name,  mic_file, TRIAL_SECS))
    t_lb.start()
    t_mic.start()

    # Let pw-record spin up, then beep
    time.sleep(BEEP_OFFSET)
    beep(freq=880,  duration=0.10)
    time.sleep(0.08)
    beep(freq=1200, duration=0.10)

    t_lb.join()
    t_mic.join()

    loopback = load_f32(lb_file)
    mic      = load_f32(mic_file)

    if loopback.size == 0 or mic.size == 0:
        return None, 0.0, "empty recording"

    min_len  = min(loopback.size, mic.size)
    skip     = int(SAMPLE_RATE * ANALYSIS_SKIP)

    if min_len - skip < SAMPLE_RATE:
        return None, 0.0, "recording too short"

    latency, confidence = measure_latency(loopback[skip:min_len], mic[skip:min_len])
    return latency, confidence, None


def main():
    print("=== game-latency — In-Game Audio Latency Tester ===")
    print("Each beep = fire ONE shot in-game.")
    print("Hold the mic firmly against the headphone ear cup throughout.\n")

    out_name, out_desc = pick(sinks(),              "Game output device")
    in_name,  in_desc  = pick(sources_no_monitor(), "Microphone")
    mon_name, mon_desc = find_monitor(out_name)

    print(f"\nOutput  : {out_desc}")
    print(f"Monitor : {mon_desc}")
    print(f"Mic     : {in_desc}")
    print(f"\nRunning {NUM_TRIALS} trials. Fire one shot per beep.\n")
    time.sleep(1)

    latencies = []

    with tempfile.TemporaryDirectory() as tmp:
        for trial in range(NUM_TRIALS):
            print(f"  Trial {trial + 1}/{NUM_TRIALS} ...", end='', flush=True)

            latency, confidence, err = run_trial(mon_name, in_name, tmp)

            if err:
                print(f" failed ({err})")
            elif confidence < CONFIDENCE_MIN:
                print(f" no clear signal (conf={confidence:.1f}x) — did you fire?")
            elif not (0 < latency < 800):
                print(f" invalid ({latency:.1f} ms, conf={confidence:.1f}x)")
            else:
                latencies.append(latency)
                print(f" {latency:.1f} ms  (conf={confidence:.1f}x)")

            if trial < NUM_TRIALS - 1:
                time.sleep(INTER_TRIAL)

    if latencies:
        print(f"\n{'─' * 44}")
        print(f"  Device : {out_desc}")
        print(f"  Trials : {len(latencies)}/{NUM_TRIALS} valid")
        print(f"  Min    : {min(latencies):.1f} ms")
        print(f"  Max    : {max(latencies):.1f} ms")
        print(f"  Mean   : {np.mean(latencies):.1f} ms")
        print(f"  Median : {np.median(latencies):.1f} ms")
        print(f"  {describe(np.median(latencies))}")
        print(f"{'─' * 44}")
    else:
        print("\nNo valid results. Check:")
        print("  - Mic is held against the headphone cup")
        print("  - Game audio is playing through the selected output")
        print("  - You fired a shot after each beep")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
