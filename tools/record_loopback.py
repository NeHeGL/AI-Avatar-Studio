"""Record whatever your PC is currently playing (like holding a phone up to the speaker, but lossless).

    python tools/record_loopback.py --out ml/out/movie_audio.wav            # Ctrl+C to stop
    python tools/record_loopback.py --out clip.wav --seconds 120

Start it, then play the scene you want in any app. Use headphones or keep the room quiet (it records the PC's
output, not the room). Nothing is downloaded and no copy protection is touched: this only captures sound as it plays.
"""
import argparse
import sys
import time
import wave

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf


def find_loopback(p):
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if not out.get("isLoopbackDevice"):
        for lb in p.get_loopback_device_info_generator():
            if out["name"] in lb["name"]:
                return lb
        raise SystemExit("No loopback device found for the default speakers: " + out["name"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--stop-file", default="", help="stop (and save) when this file appears")
    a = ap.parse_args()
    p = pyaudio.PyAudio()
    dev = find_loopback(p)
    ch, rate = int(dev["maxInputChannels"]), int(dev["defaultSampleRate"])
    print(f"Recording what plays on: {dev['name']}  ({ch} ch, {rate} Hz).  Ctrl+C to stop.", flush=True)
    frames = []
    stream = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, frames_per_buffer=2048, input=True,
                    input_device_index=dev["index"])
    t0 = time.time()
    try:
        import os
        while (a.seconds <= 0 or time.time() - t0 < a.seconds) and not (a.stop_file and os.path.exists(a.stop_file)):
            frames.append(stream.read(2048, exception_on_overflow=False))
            if not a.quiet and len(frames) % 20 == 0:
                print(f"\r  {time.time()-t0:6.1f}s", end="", flush=True)
    except KeyboardInterrupt:
        pass
    stream.stop_stream(); stream.close(); p.terminate()
    pcm = np.frombuffer(b"".join(frames), dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
    sf.write(a.out, pcm, rate)
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    print(f"\nsaved {a.out}: {len(pcm)/rate:.1f}s, peak level {peak:.3f}" + ("  (silent - was anything playing?)" if peak < 0.001 else ""))


if __name__ == "__main__":
    main()
