# Standalone ONNX/JAR with CPU and GPU device selection

Release artifact: `dist/uzbek-tts-0.1.0.jar` (2,956,955,285 bytes).
SHA-256: `68f53df17a685a62c4119ab5f4959f1015c65fe8ad4214daf112e8c515bf54c0`.

The JAR includes the models, tokenizer, default voice conditioning, Java engine,
and platform-native ONNX Runtime. Java 17+ is required. Python, Docker, and the
old source/API projects are not runtime dependencies.

## Verified

- 10 Java test methods passed, including tokenizer parity on 114 text inputs and device selection logic, and resource cleanup ordering.
- Embedding and language-model parity with PyTorch passed; LM maximum logit error was approximately 0.00002.
- Deterministic decoder adapter parity passed, waveform cosine 1.0.
- CPU, explicit Mac GPU, automatic Mac GPU selection, and initialization-failure fallback to CPU generated real audio.
- Packaged JAR GPU profiling recorded GPU kernel execution in all three ONNX graphs.
- CPU/GPU waveform cosine on the fixed smoke input: 0.9999999991.
- An otherwise empty directory containing only the updated JAR generated 2.86s audio without an external model path/classpath.
- GPU HTTP mode passed 19 checks: auth, validation, CORS, discovery, real WAV, and concurrent 429 response.
- Packaged GPU HTTP server shutdown passed both SIGTERM and SIGINT after real audio; exit codes 143/130, no native crash.
- The GPU HTTP request produced 1.02s audio in 3.14s on this machine.

## Device support

Use `--device cpu|mps|cuda|auto`; default is auto. Mac `mps` uses Metal via native
WebGPU. CUDA provider binaries are bundled for Linux x64 and Windows x64, but
NVIDIA hardware testing was unavailable. CUDA requires compatible driver,
CUDA 12.8+, and cuDNN 9 host libraries.

Full commands, platform limits and evidence: [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md).

Current audio features: bundled default voice, 24-kHz PCM16 WAV, no voice cloning
or real-time streaming. Other OS/architectures remain untested here. First use
extracts about 2.68 GB of model assets to cache; GPU providers may use CPU for
unsupported operators. Different devices can produce slightly different samples.

Default device changed to auto in CLI, HTTP mode, and Java constructors. The JAR was updated without running tests, as requested. Validation reports above describe the preceding tested release.
