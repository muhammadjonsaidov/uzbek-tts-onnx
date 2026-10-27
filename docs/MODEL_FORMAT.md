# ONNX bundle contract

`models/manifest.json` describes graph paths and the default voice tensors.
All model paths are relative to the bundle. Tensor `.bin` files are row-major,
little-endian float32 or int64 according to their metadata. No Python pickle is used.

## Graphs

| Graph | Inputs | Outputs |
|---|---|---|
| embeddings | `input_ids`, `position_ids`, `token_kind`: int64[2,S]; `exaggeration`: float32 scalar | `inputs_embeds`: float32[2,S,1024]; `emotion_embed`: float32[1,1,1024] |
| language_model | `inputs_embeds`: float32[2,Q,1024]; `attention_mask`: int64[2,P+Q]; `cfg_weight`: float32 scalar; 60 `past_key_values.{0..29}.{key,value}` tensors: float32[2,16,P,64] | `logits`: float32[1,Q,8194]; `attn_layers`: float32[3,16,Q,P+Q]; 60 `present.{0..29}.{key,value}` tensors |
| decoder | `speech_tokens`: int64[1,L]; `speaker_embeddings`: float32[1,192]; `speaker_features`: float32[1,R,80]; `flow_noise`: float32[1,80,2L]; `source_noise`: float32[1,9,(2L-R)*480] | `waveform`: float32[1,(2L-R)*480] |

The decoder receives prompt tokens followed by generated speech tokens. The
prompt feature count R can be odd; compute noise lengths from R rather than
assuming exactly 960 output samples per generated token.

## Text and generation

Use the bundled BPE tokenizer, replacing spaces with `[SPACE]`, and surround
text IDs with start-text 255 and stop-text 0. Punctuation normalization matches
the local model toolkit. `runtime/tts.py` and `TextTokenizer.java` are reference
implementations; Java parity fixtures are in `tests/tokenizer-cases.json`.

Batch row 0 is conditional and row 1 is unconditional. For text, CFG clears the
token embedding of row 1 before adding position embeddings. It does not clear
position embeddings. `token_kind=0` selects text and `token_kind=1` selects speech.

The prefill concatenates 33 cached conditioning vectors, one emotion vector,
text embeddings, and two BOS speech embeddings at speech position 0. This
preserves the provided toolkit's inference behavior. The first language-model
call uses P=0. Subsequent calls supply one speech embedding at the next speech
position and reuse the returned KV caches. CFG is combined inside the LM graph.

Sample the next token with repetition penalty, temperature, and min-p filtering.
Use alignment heads (9,2), (12,15), and (13,11) for EOS integrity checks. The
reference runtimes implement these checks. Stop at speech EOS 6562; generated
tokens passed to the decoder must be less than 6561. Reference max generation
is 1000 tokens, with long text split into chunks of at most 140 characters.

Default-voice conditioning is precalculated. New reference voices need a
reference-audio encoder; that graph is not part of this first release.

## Native integration

Any language with ONNX Runtime can load these graphs, but must implement the
preprocessing and generation loop above. The ready JAR includes the complete
loop and can be used directly in Java or exposed over HTTP to other languages.
The runtime does not need `conversion/`, `checkpoints/`, Torch, or training code.
