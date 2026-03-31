#!/usr/bin/env python3
"""
game-latency.py — Measure real in-game audio latency.

Records simultaneously:
  - PipeWire monitor source  (digital loopback — what the game outputs, zero-latency reference)
  - Microphone               (physical pickup — when sound exits your headphones)

Cross-correlates the two to find end-to-end latency for any game audio.

Usage:
    python3 game-latency.py

During the recording window, make a loud distinct in-game sound
(e.g. Cassidy primary fire in Overwatch, gunshot, ability with a sharp transient).
Hold the mic against the headphone ear cup throughout.
"""

import os
import sys
import time
import threading
import tempfile
import subprocess
import numpy as np
from scipy.signal import correlate

SAMPLE_RATE  = 48000
RECORD_SECS  = 8
CHANNELS     = 1
COUNTDOWN    = 3


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
    """Return the monitor source name for a given sink."""
    for name, desc in pactl_devices('source'):
        if '.monitor' in name and sink_name.split('alsa_output.')[-1].split('.analog')[0] in name:
            return name, desc
    # Fallback — PipeWire always creates <sink_name>.monitor
    return sink_name + '.monitor', f'Monitor of {sink_name}'


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


# ── Recording ─────────────────────────────────────────────────────────────────

def record(source_name, filepath, duration):
    """Record from a PipeWire source into a raw float32 file."""
    proc = subprocess.Popen(
        ['pw-record', '--target', source_name,
         '--rate', str(SAMPLE_RATE),
         '--channels', str(CHANNELS),
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
    """Sub-sample peak refinement — returns fractional offset in [-0.5, 0.5]."""
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def measure_latency(loopback, mic):
    """
    Cross-correlate loopback (reference) and mic (delayed) recordings.
    Returns (latency_ms, confidence) where confidence is peak/median SNR.
    """
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
        return "Negative result — check device selection (mic before loopback?)"
    elif ms < 20:
        return "Wired-equivalent — imperceptible."
    elif ms < 50:
        return "Excellent — competitive grade."
    elif ms < 100:
        return "Good — fine for most gaming."
    elif ms < 250:
        return "Moderate — noticeable in rhythm games."
    else:
        return "High — significant Bluetooth or software buffering overhead."


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== game-latency — In-Game Audio Latency Tester ===")
    print("Records game audio + mic simultaneously, cross-correlates to find latency.")
    print("Hold the mic firmly against the headphone ear cup throughout.\n")

    out_name,  out_desc  = pick(sinks(),              "Game output device")
    in_name,   in_desc   = pick(sources_no_monitor(), "Microphone")
    mon_name,  mon_desc  = find_monitor(out_name)

    print(f"\nOutput  : {out_desc}")
    print(f"Monitor : {mon_desc}")
    print(f"Mic     : {in_desc}")
    print(f"\nCountdown then {RECORD_SECS}s recording window.")
    print("Fire a shot (Cassidy, Soldier, etc.) once you hear 'GO'.\n")

    for i in range(COUNTDOWN, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO — make your sound!\n")

    with tempfile.TemporaryDirectory() as tmp:
        lb_file  = os.path.join(tmp, 'loopback.f32')
        mic_file = os.path.join(tmp, 'mic.f32')

        t_lb  = threading.Thread(target=record, args=(mon_name, lb_file,  RECORD_SECS))
        t_mic = threading.Thread(target=record, args=(in_name,  mic_file, RECORD_SECS))
        t_lb.start()
        t_mic.start()

        for remaining in range(RECORD_SECS, 0, -1):
            print(f"\r  {remaining}s remaining...", end='', flush=True)
            time.sleep(1)
        print("\r  Recording done.     ")

        t_lb.join()
        t_mic.join()

        loopback = load_f32(lb_file)
        mic      = load_f32(mic_file)

        if loopback.size == 0 or mic.size == 0:
            print("\nERROR: Nothing recorded. Is pw-record installed? Check device selection.")
            sys.exit(1)

        min_len  = min(loopback.size, mic.size)
        latency, confidence = measure_latency(loopback[:min_len], mic[:min_len])

    print(f"\n{'─' * 44}")
    print(f"  Device     : {out_desc}")
    print(f"  Latency    : {latency:.1f} ms")
    print(f"  Confidence : {confidence:.1f}x  {'✓ clear signal' if confidence > 12 else '⚠ weak — try a louder sound or hold mic closer'}")
    print(f"  {describe(latency)}")
    print(f"{'─' * 44}")
    print("\nTip: run it several times and average — a single gunshot can vary slightly.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
