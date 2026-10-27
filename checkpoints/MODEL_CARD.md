---
license: mit
language:
- uz
pipeline_tag: text-to-speech
tags:
- tts
- text-to-speech
- chatterbox
- uzbek
- lora
- voice-cloning
---

# Uzbek Chatterbox TTS (LoRA fine-tune)

A LoRA fine-tune of [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (Standard/non-Turbo)
for **Uzbek** speech synthesis, trained on ~30 hours of single-speaker Uzbek audiobook narration.

The LoRA adapter has been merged into the base model, so these are standalone `T3` weights —
no PEFT required at inference time. The model also supports **zero-shot voice cloning**: pass
any reference wav and it will speak your text in that voice.

## Demo

### Fixed-voice synthesis (the trained speaker)

<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/tts_demo/sample1.wav"></audio>
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/tts_demo/sample2.wav"></audio>
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/tts_demo/sample3.wav"></audio>
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/tts_demo/sample4.wav"></audio>
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/tts_demo/sample5.wav"></audio>

### Voice cloning (unrelated reference speakers, same fine-tuned model)

**Bobur Akmalov**
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/voice_cloning_demo/Bobur%20Akmalov.wav"></audio>

**Davron Fayziyev**
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/voice_cloning_demo/Davron%20Fayziyev.wav"></audio>

**Jennifer Lawrence**
<audio controls src="https://huggingface.co/Abduqayum/uzbek-tts-natural-speech-chatterbox/resolve/main/samples/voice_cloning_demo/Jennifer%20Lawrence.wav"></audio>

## Files

| File | Description |
|---|---|
| `t3_finetuned_merged.safetensors` | Standalone fine-tuned T3 (speech-token AR model), LoRA merged in |
| `tokenizer.json` | Grapheme tokenizer, 2454 tokens (unchanged from base — Standard mode uses a fixed multilingual grapheme vocab that already covers Uzbek Latin script) |
| `reference_voice.wav` | Default reference clip for the fixed training voice |
| `samples/` | Demo audio (see above) |
| `inference.py` | Standalone script to run synthesis, fixed-voice or cloning |

The `s3gen` (vocoder) and `ve` (voice encoder) weights are **not** re-uploaded here since they're
unmodified from the base model — `inference.py` expects them from a base install (see Usage).

## Usage

This model was trained with, and is meant to be run through,
[gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning) —
the same toolkit used to train it. That repo vendors the Chatterbox model code and gives you
`s3gen.safetensors` / `ve.safetensors` / `conds.pt` via its `setup.py`.

```bash
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning
pip install -r requirements.txt
# if you hit a pkg_resources / setuptools ImportError from resemble-perth:
pip install "setuptools<81"

python setup.py   # downloads base s3gen/ve/conds + the 2454-token tokenizer

# download this repo's files into the toolkit root (or anywhere, and pass paths as flags):
huggingface-cli download <this-repo-id> --local-dir uzbek_chatterbox
cp uzbek_chatterbox/inference.py .
```

**Fixed voice:**
```bash
python inference.py \
  --text "Salom, bugungi kunda sun'iy intellekt texnologiyalari juda tez rivojlanmoqda." \
  --audio_prompt uzbek_chatterbox/reference_voice.wav \
  --merged_weights uzbek_chatterbox/t3_finetuned_merged.safetensors \
  --output out.wav
```

**Voice cloning** (swap in any reference wav):
```bash
python inference.py \
  --text "Salom, qandaysiz?" \
  --audio_prompt path/to/some_other_voice.wav \
  --merged_weights uzbek_chatterbox/t3_finetuned_merged.safetensors \
  --output out.wav
```

Text of any length is handled — `inference.py` splits long input into sentence-sized chunks
internally (a single generate() call is capped at 1000 speech tokens and can otherwise cut off
mid-word on long paragraphs).

## Training

See [TRAINING.md](TRAINING.md) for the full step-by-step recipe (dataset format, exact config,
commands) to reproduce this fine-tune on your own dataset/language.

## License

MIT, inherited from the base [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
model. The training toolkit ([gokhaneraslan/chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning))
is Apache 2.0.
