"""
Lab 4 - Hear: transcribe speech with a local Whisper model (WhisperPipeline).

Input: a 16 kHz mono WAV file, or record from the microphone.

Get the model once (pre-converted, ~150 MB):
    hf download OpenVINO/whisper-base-fp16-ov --local-dir whisper-base

Run:
    python speech_basics.py --wav my_recording.wav
    python speech_basics.py --record 5            # speak for 5 seconds
    python speech_basics.py --record 5 --device GPU

Exercise (voice-to-image): pass the transcript to image_gen_ov.py.
"""
import argparse
import time
import wave

import numpy as np
import openvino_genai as ov_genai

SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono float samples


def load_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need a 16 kHz, mono, 16-bit WAV file")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def record(seconds):
    import sounddevice as sd
    print(f"Recording {seconds}s... speak now")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main():
    parser = argparse.ArgumentParser(description="Lab 4: speech to text with local Whisper")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="16 kHz mono WAV file")
    source.add_argument("--record", type=float, metavar="SECONDS", help="record from the microphone")
    parser.add_argument("--model", default="./whisper-base")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU", "NPU"])
    args = parser.parse_args()

    raw_speech = load_wav(args.wav) if args.wav else record(args.record)

    pipe = ov_genai.WhisperPipeline(args.model, args.device)
    start = time.time()
    result = pipe.generate(raw_speech.tolist())
    print(str(result).strip())
    print(f"\n--- {args.device} | {time.time() - start:.2f}s for {len(raw_speech) / SAMPLE_RATE:.1f}s of audio")


if __name__ == "__main__":
    main()
