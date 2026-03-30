#!/usr/bin/env python3
"""
bt-latency.py — Measure Bluetooth audio latency using a microphone.

Place the mic close to the headphone ear cup before running.
"""

import sys
import time
import subprocess
import numpy as np
import sounddevice as sd
from scipy.signal import chirp, correlate

SAMPLE_RATE    = 48000
RECORD_SECS    = 2.5
CHIRP_OFFSET   = 0.4
CHIRP_DURATION = 0.05
NUM_TRIALS     = 5


def pactl_devices(kind):
    """Return list of (pactl_name, friendly_description) for sinks or sources."""
    out = subprocess.check_output(['pactl', 'list', f'{kind}s'], text=True)
    devices = []
    name = desc = None
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


def pick(devices, label):
    if not devices:
        raise RuntimeError(f"No {label.lower()} detected by pactl.")

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


def generate_chirp():
    t = np.linspace(0, CHIRP_DURATION, int(SAMPLE_RATE * CHIRP_DURATION), endpoint=False)
    sig = chirp(t, f0=200, f1=8000, t1=CHIRP_DURATION, method='logarithmic')
    sig *= np.hanning(len(sig))
    return sig.astype(np.float32)


def measure(out_pactl, in_pactl):
    # Temporarily set PipeWire defaults so sounddevice picks them up
    orig_sink   = subprocess.check_output(['pactl', 'get-default-sink'],   text=True).strip()
    orig_source = subprocess.check_output(['pactl', 'get-default-source'], text=True).strip()
    subprocess.run(['pactl', 'set-default-sink',   out_pactl], check=True)
    subprocess.run(['pactl', 'set-default-source', in_pactl],  check=True)
    time.sleep(0.3)  # let PipeWire settle

    chirp_sig    = generate_chirp()
    click_sample = int(SAMPLE_RATE * CHIRP_OFFSET)
    total        = int(SAMPLE_RATE * RECORD_SECS)

    playback = np.zeros(total, dtype=np.float32)
    end = min(click_sample + len(chirp_sig), total)
    playback[click_sample:end] = chirp_sig[:max(0, end - click_sample)]

    latencies = []

    try:
        for trial in range(NUM_TRIALS + 1):  # +1 for throwaway warmup
            play_pos = [0]
            rec_pos  = [0]
            rec_buf  = np.zeros(total, dtype=np.float32)
            saw_xrun = [False]

            def out_cb(outdata, frames, time_info, status):
                if status:
                    saw_xrun[0] = True
                s, e = play_pos[0], min(play_pos[0] + frames, total)
                n = e - s
                outdata[:n, 0] = playback[s:e]
                if n < frames:
                    outdata[n:, 0] = 0.0
                play_pos[0] = e

            def in_cb(indata, frames, time_info, status):
                if status:
                    saw_xrun[0] = True
                s, e = rec_pos[0], min(rec_pos[0] + frames, total)
                n = e - s
                if n > 0:
                    rec_buf[s:e] = indata[:n, 0]
                    rec_pos[0]   = e

            if trial == 0:
                print(f"  Warmup ...", end='', flush=True)
            else:
                print(f"  Trial {trial}/{NUM_TRIALS} ...", end='', flush=True)

            try:
                with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1,
                                     callback=out_cb, latency='low'), \
                     sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    callback=in_cb, latency='low'):
                    time.sleep(RECORD_SECS + 0.3)
            except sd.PortAudioError as e:
                print(f" ERROR: {e}")
                continue

            corr = correlate(rec_buf, chirp_sig, mode='full')
            corr_abs = np.abs(corr)
            lag  = int(np.argmax(corr_abs)) - (len(chirp_sig) - 1)
            latency_ms = ((lag - click_sample) / SAMPLE_RATE) * 1000
            peak = float(np.max(corr_abs))
            med = float(np.median(corr_abs)) + 1e-9
            confidence = peak / med

            if trial == 0:
                print(f" {latency_ms:.1f} ms (discarded)")
            elif saw_xrun[0]:
                print(" skipped (audio over/underrun)")
            elif 10 < latency_ms < 800 and confidence > 15:
                latencies.append(latency_ms)
                print(f" {latency_ms:.1f} ms")
            else:
                print(f" no signal ({latency_ms:.1f} ms) — hold mic against ear cup")

            time.sleep(0.3)
    finally:
        subprocess.run(['pactl', 'set-default-sink',   orig_sink],   check=False)
        subprocess.run(['pactl', 'set-default-source', orig_source], check=False)

    return latencies


def main():
    print("=== bt-latency — Bluetooth Audio Latency Tester ===")
    print("Hold the mic firmly against one ear cup throughout.\n")

    try:
        sinks   = pactl_devices('sink')
        sources = pactl_devices('source')
    except FileNotFoundError:
        print("ERROR: 'pactl' not found. Install PulseAudio/PipeWire utilities.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: failed to query audio devices with pactl: {e}")
        sys.exit(1)

    out_pactl, out_desc = pick(sinks,   "Output device (headphones)")
    in_pactl,  in_desc  = pick(sources, "Input device  (microphone)")

    print(f"\nOutput : {out_desc}")
    print(f"Input  : {in_desc}")
    print(f"\nRunning {NUM_TRIALS} trials...\n")

    latencies = measure(out_pactl, in_pactl)

    if latencies:
        print(f"\n{'─' * 44}")
        print(f"  Device : {out_desc}")
        print(f"  Trials : {len(latencies)}/{NUM_TRIALS} valid")
        print(f"  Min    : {min(latencies):.1f} ms")
        print(f"  Max    : {max(latencies):.1f} ms")
        print(f"  Mean   : {np.mean(latencies):.1f} ms")
        print(f"  Median : {np.median(latencies):.1f} ms")
        print(f"{'─' * 44}")
    else:
        print("\nNo valid results. Try:")
        print("  - Press mic firmly against the headphone cup")
        print("  - Increase mic gain  →  alsamixer or pavucontrol")
        print("  - Re-check device selection")


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
