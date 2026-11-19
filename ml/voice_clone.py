"""Speak new text in a cloned voice (Chatterbox, MIT) on the GPU.
usage: python voice_clone.py <reference_audio> "<text>" <out.wav>
"""
import sys, time
import numpy as np
import soundfile as sf
import torch
import torchaudio

ref, text, out = sys.argv[1], sys.argv[2], sys.argv[3]

# Reference clip -> mono wav the model can read (handles .ogg/.webm/.wav)
data, sr = sf.read(ref, always_2d=True)
mono = data.mean(axis=1).astype(np.float32)
ref_wav = out.rsplit(".", 1)[0] + "_ref.wav"
sf.write(ref_wav, mono, sr)

from chatterbox.tts import ChatterboxTTS
t0 = time.time()
model = ChatterboxTTS.from_pretrained(device="cuda")
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
wav = model.generate(text, audio_prompt_path=ref_wav, exaggeration=0.5, cfg_weight=0.5)
print(f"generated in {time.time()-t0:.1f}s, {wav.shape[-1]/model.sr:.1f}s of audio", flush=True)
sf.write(out, wav.squeeze().cpu().numpy(), model.sr)
print("saved", out)
