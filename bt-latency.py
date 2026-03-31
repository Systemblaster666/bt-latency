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

FALLBACK_SAMPLE_RATE = 48000
RECORD_SECS    = 2.5
CHIRP_OFFSET   = 0.4
CHIRP_DURATION = 0.05
NUM_TRIALS     = 5
MIN_LAT_MS     = 10
MAX_LAT_MS     = 800
CONFIDENCE_MIN = 12


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


def pactl_device_sample_rate(kind, pactl_name):
    """Return native sample rate in Hz for a specific sink/source, if reported."""
    out = subprocess.check_output(['pactl', 'list', f'{kind}s'], text=True)
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Name:') and line.split(':', 1)[1].strip() == pactl_name:
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(f'{kind.capitalize()} #'):
                spec = lines[j].strip()
                if spec.startswith('Sample Specification:'):
                    # Example: "Sample Specification: s16le 2ch 48000Hz"
                    for token in spec.split():
                        if token.endswith('Hz') and token[:-2].isdigit():
                            return int(token[:-2])
                j += 1
            return None
        i += 1
    return None


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


def generate_chirp(sample_rate):
    t = np.linspace(0, CHIRP_DURATION, int(sample_rate * CHIRP_DURATION), endpoint=False)
    sig = chirp(t, f0=200, f1=8000, t1=CHIRP_DURATION, method='logarithmic')
    sig *= np.hanning(len(sig))
    return sig.astype(np.float32)


def parabolic_peak_offset(y, idx):
    """Return sub-sample peak offset in [-0.5, 0.5] around integer idx."""
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def measure(out_pactl, in_pactl, sample_rate):
    # Temporarily set PipeWire defaults so sounddevice picks them up
    orig_sink   = subprocess.check_output(['pactl', 'get-default-sink'],   text=True).strip()
    orig_source = subprocess.check_output(['pactl', 'get-default-source'], text=True).strip()
    subprocess.run(['pactl', 'set-default-sink',   out_pactl], check=True)
    subprocess.run(['pactl', 'set-default-source', in_pactl],  check=True)
    time.sleep(0.3)  # let PipeWire settle

    chirp_sig    = generate_chirp(sample_rate)
    click_sample = int(sample_rate * CHIRP_OFFSET)
    total        = int(sample_rate * RECORD_SECS)

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

            def duplex_cb(indata, outdata, frames, time_info, status):
                if status:
                    saw_xrun[0] = True
                # Playback
                s, e = play_pos[0], min(play_pos[0] + frames, total)
                n = e - s
                outdata[:n, 0] = playback[s:e]
                if n < frames:
                    outdata[n:, 0] = 0.0
                play_pos[0] = e
                # Recording
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
                with sd.Stream(samplerate=sample_rate,
                               channels=1,
                               dtype='float32',
                               callback=duplex_cb,
                               latency='low'):
                    time.sleep(RECORD_SECS + 0.3)
            except sd.PortAudioError as e:
                print(f" ERROR: {e}")
                continue

            corr = correlate(rec_buf, chirp_sig, mode='full')
            corr_abs = np.abs(corr)
            peak_idx = int(np.argmax(corr_abs))
            lag = peak_idx - (len(chirp_sig) - 1) + parabolic_peak_offset(corr_abs, peak_idx)
            latency_ms = ((lag - click_sample) / sample_rate) * 1000
            peak = float(np.max(corr_abs))
            med = float(np.median(corr_abs)) + 1e-9
            confidence = peak / med

            if trial == 0:
                print(f" {latency_ms:.1f} ms (discarded)")
            elif saw_xrun[0]:
                print(" skipped (audio over/underrun)")
            elif MIN_LAT_MS < latency_ms < MAX_LAT_MS and confidence > CONFIDENCE_MIN:
                latencies.append(latency_ms)
                print(f" {latency_ms:.1f} ms (conf={confidence:.1f})")
            else:
                print(f" no signal ({latency_ms:.1f} ms, conf={confidence:.1f}) — hold mic against ear cup")

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

    try:
        out_rate = pactl_device_sample_rate('sink', out_pactl)
        in_rate = pactl_device_sample_rate('source', in_pactl)
    except subprocess.CalledProcessError:
        out_rate = None
        in_rate = None
    if out_rate and in_rate and out_rate == in_rate:
        sample_rate = out_rate
        rate_note = "sink/source native rates match"
    elif out_rate and in_rate and out_rate != in_rate:
        sample_rate = out_rate
        rate_note = "native rates differ; resampling likely on one side"
    elif out_rate or in_rate:
        sample_rate = out_rate or in_rate
        rate_note = "only one side reported native rate"
    else:
        sample_rate = FALLBACK_SAMPLE_RATE
        rate_note = "native rate not reported by pactl"

    print(f"\nOutput : {out_desc}")
    print(f"Input  : {in_desc}")
    print(f"Rate   : {sample_rate} Hz ({rate_note})")
    print(f"\nRunning {NUM_TRIALS} trials...\n")

    latencies = measure(out_pactl, in_pactl, sample_rate)

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
