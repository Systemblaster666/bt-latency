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
(e.g. Cassidy primary fire in Overwatch).
Hold the mic against the headphone ear cup throughout.
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
    """PipeWire always creates <sink_name>.monitor as a source."""
    monitor_name = sink_name + '.monitor'
    # Verify it exists in pactl sources
    for name, desc in pactl_devices('source'):
        if name == monitor_name:
            return name, desc
    # Return it anyway — pw-record will error if it doesn't exist
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


# ── Audible beep ──────────────────────────────────────────────────────────────

def beep(freq=880, duration=0.15):
    """Play a short tone through the default output so the user knows when to shoot."""
    t   = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sig *= np.hanning(len(sig))
    try:
        sd.play(sig, samplerate=SAMPLE_RATE, blocking=True)
    except Exception:
        pass  # non-fatal if output fails


# ── Recording ─────────────────────────────────────────────────────────────────

def record(source_name, filepath, duration, errors):
    """Record from a PipeWire source into a raw float32 file."""
    result = subprocess.run(
        ['pw-record', '--target', source_name,
         '--rate', str(SAMPLE_RATE),
         '--channels', str(CHANNELS),
         '--format', 'f32',
         filepath],
        timeout=duration + 3,
        capture_output=True,
        text=True
    )
    if result.returncode not in (0, -15):  # -15 = SIGTERM (normal termination)
        errors.append(f"{source_name}: {result.stderr.strip()}")


def record_timed(source_name, filepath, duration, errors):
    """Wrapper that terminates pw-record after duration seconds."""
    proc = subprocess.Popen(
        ['pw-record', '--target', source_name,
         '--rate', str(SAMPLE_RATE),
         '--channels', str(CHANNELS),
         '--format', 'f32',
         filepath],
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(duration)
    proc.terminate()
    try:
        _, stderr = proc.communicate(timeout=2)
        # pw-record prints the output filepath to stderr as a normal status line — ignore it
        real_errors = [l for l in (stderr or '').splitlines()
                       if l.strip() and filepath not in l and 'Recording' not in l]
        if real_errors:
            errors.append(f"{source_name}: {'; '.join(real_errors)}")
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
        return "Negative result — check device selection"
    elif ms < 20:
        return "Wired-equivalent — imperceptible"
    elif ms < 50:
        return "Excellent — competitive grade"
    elif ms < 100:
        return "Good — fine for most gaming"
    elif ms < 250:
        return "Moderate — noticeable in rhythm games"
    else:
        return "High — significant BT or software buffering overhead"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== game-latency — In-Game Audio Latency Tester ===")
    print("Hold the mic firmly against the headphone ear cup throughout.\n")

    out_name,  out_desc  = pick(sinks(),              "Game output device")
    in_name,   in_desc   = pick(sources_no_monitor(), "Microphone")
    mon_name,  mon_desc  = find_monitor(out_name)

    print(f"\nOutput  : {out_desc}")
    print(f"Monitor : {mon_desc}")
    print(f"Mic     : {in_desc}")
    print(f"\nRecording will start, then a BEEP signals when to shoot.")
    print(f"You have {RECORD_SECS}s after the beep.\n")

    with tempfile.TemporaryDirectory() as tmp:
        lb_file  = os.path.join(tmp, 'loopback.f32')
        mic_file = os.path.join(tmp, 'mic.f32')
        errors   = []

        # Start recording FIRST, then beep so no sounds are missed
        t_lb  = threading.Thread(target=record_timed, args=(mon_name, lb_file,  RECORD_SECS + 1, errors))
        t_mic = threading.Thread(target=record_timed, args=(in_name,  mic_file, RECORD_SECS + 1, errors))
        t_lb.start()
        t_mic.start()

        # Brief pause to let pw-record spin up, then beep
        time.sleep(0.5)
        print("  *** BEEP = shoot now ***")
        beep(freq=880, duration=0.15)
        time.sleep(0.1)
        beep(freq=1200, duration=0.15)  # double beep so it's obvious

        for remaining in range(RECORD_SECS, 0, -1):
            print(f"\r  {remaining}s remaining...", end='', flush=True)
            time.sleep(1)
        print("\r  Recording done.        ")

        t_lb.join()
        t_mic.join()

        # Show debug info
        lb_size  = os.path.getsize(lb_file)  if os.path.exists(lb_file)  else 0
        mic_size = os.path.getsize(mic_file) if os.path.exists(mic_file) else 0
        print(f"\n  Loopback : {lb_size // 1024} KB recorded")
        print(f"  Mic      : {mic_size // 1024} KB recorded")

        if errors:
            print("\nRecording errors:")
            for e in errors:
                print(f"  {e}")

        if lb_size == 0 or mic_size == 0:
            print("\nERROR: One or both recordings are empty.")
            print("  - Check that pw-record is installed: which pw-record")
            print(f"  - Monitor source may not exist: {mon_name}")
            print("    Run: pactl list sources | grep monitor")
            sys.exit(1)

        loopback = load_f32(lb_file)
        mic      = load_f32(mic_file)
        min_len  = min(loopback.size, mic.size)
        latency, confidence = measure_latency(loopback[:min_len], mic[:min_len])

    print(f"\n{'─' * 44}")
    print(f"  Device     : {out_desc}")
    print(f"  Latency    : {latency:.1f} ms")
    print(f"  Confidence : {confidence:.1f}x  {'✓' if confidence > 12 else '⚠ weak — louder sound or closer mic'}")
    print(f"  {describe(latency)}")
    print(f"{'─' * 44}")
    print("\nTip: run several times and average — a single shot can vary slightly.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
