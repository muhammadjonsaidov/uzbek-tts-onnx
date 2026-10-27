# Device selection

The published `dist/uzbek-tts-0.1.0.jar` accepts `--device cpu|mps|cuda|auto`.
Default: **auto**. The flag applies to both
one-shot synthesis and `--serve` mode.

| Mode | Backend | Verification |
|---|---|---|
| cpu | CPUExecutionProvider | Real audio and original default behavior passed |
| mps | WebGpuExecutionProvider on Apple Metal | Real audio, all three graph GPU profiles, packaged JAR, and HTTP passed on macOS ARM64 |
| cuda | CUDAExecutionProvider, GPU 0 | Linux x64/Windows x64 native libraries bundled; NVIDIA hardware unavailable for testing here |
| auto | Prefer CUDA, then Mac GPU, then CPU | Real Mac GPU selection and explicit initialization-failure fallback to CPU passed |

`mps` is the application's Apple GPU device name. Its ONNX implementation uses
native WebGPU/Dawn with Metal, rather than PyTorch's MPS runtime. This path was
selected after CoreML failed to initialize the dynamic language-model graph.
The ONNX Runtime documentation describes the native WebGPU-to-Metal backend:
https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html

## Commands from dist/

```bash
java -jar uzbek-tts-0.1.0.jar --list-devices
java -jar uzbek-tts-0.1.0.jar --device mps --text "Salom!" --output speech.wav
java -jar uzbek-tts-0.1.0.jar --device cpu --text "Salom!" --output speech.wav
java -jar uzbek-tts-0.1.0.jar --device cuda --text "Salom!" --output speech.wav
java -jar uzbek-tts-0.1.0.jar --device auto --serve --port 8000
```

The console and synthesis JSON show `requested_device`, effective `device`,
`provider`, and ONNX Runtime version. `/ready` includes the same information.
`--list-devices` lists registered runtime support, not proof of installed GPU
hardware/drivers. Hardware/library failures are detected during session creation.
An explicit unavailable `mps` or `cuda` request fails. Only `auto` may switch the
whole model back to CPU, recording a `fallback_reason`.

GPU providers can still assign unsupported operators to CPU. That is reported
with `cpu_operator_fallback` and is visible in `--profile DIR` traces. The GPU
smoke test verified WebGPU kernel events in embeddings, language_model, and
decoder; this is not just a renamed CPU mode.

## CUDA prerequisites

The same JAR bundles CUDA-enabled ORT core/JNI/provider libraries for Linux x64
and Windows x64, and Mac native libraries for Apple GPU/CPU. CUDA execution still
needs a compatible NVIDIA driver and CUDA/cuDNN host libraries. For the bundled
ORT 1.23.2 CUDA build, use CUDA 12.8+ and cuDNN 9. CUDA libraries are loaded when
CUDA is requested; the Linux ORT core's shared-library dependency inspection
showed only standard CPU system libraries.

Official compatibility details:
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

CUDA hardware execution is **not claimed as tested** on this Mac. The unavailable
CUDA error path was tested and does not silently run on CPU.

## Evidence and performance

- `validation-devices.json`: release checksum, GPU native packaging, waveform parity.
- `device-jar-mps-result.json`: real inference using the packaged JAR and GPU events.
- `device-auto-metal-result.json`: auto selected Apple GPU and generated audio.
- `device-auto-cpu-fallback-result.json`: initialization failure was reported, CPU inference completed.
- `validation-http.json`: GPU HTTP mode passed 19 checks and generated WAV.
- `validation-standalone.json`: the updated JAR ran from an otherwise empty directory.

CPU and GPU PCM waveform cosine on the fixed greedy smoke input was
0.9999999991. This is a tested example, not a guarantee of bit-identical samples
across devices. Cold startup, shader compilation, profiling, and CPU/GPU copies
can affect timings; GPU is not guaranteed to outperform CPU for every input.
The profiler-enabled standalone tests are not a controlled throughput benchmark.

Earlier failed CoreML experiment reports are retained for diagnostics. The
published default Apple path is Metal/WebGPU on ORT 1.23.2; no runtime upgrade
or Python dependency was needed.

## Runtime shutdown order

The bundled Java `OrtEnvironment.java` has a small local patch against upstream
v1.23.2: its environment shutdown hook first closes registered TTS models, then
releases the native environment. Without this ordering, the original environment
hook and the application's server hook race, freeing WebGPU buffers after the
shared GPU context has already gone away. TTS close is synchronized with inference
and idempotent, so the server hook and environment hook can safely overlap.

`tests/shutdown_smoke.py` checks real HTTP audio followed by both SIGTERM and
SIGINT. `validation-shutdown-mps.json` records the tested JAR checksum and exit
results. Keep the vendored Java wrapper and native runtime pinned to the same
version when upgrading. See `THIRD_PARTY.md` and `LICENSE.onnxruntime`.
