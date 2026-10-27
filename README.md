# Uzbek TTS ONNX

Standalone Uzbek text-to-speech, packaged as a single executable JAR.

## Purpose

This project makes an Uzbek TTS model easy to distribute and use in other
applications. The goal is simple: provide one package, pass it text, and receive
speech without setting up a Python inference environment or the original
Chatterbox project.

The same engine can run from the command line, inside a Java application, or as
an HTTP service for applications written in any language.

## How it works

The Uzbek Chatterbox model is exported into ONNX graphs. A Java inference engine
handles text tokenization, speech-token generation, and audio decoding using
ONNX Runtime.

The exported models, tokenizer, default voice data, Java engine, and native
runtime libraries are bundled into **`uzbek-tts-0.1.0.jar`**. Model conversion and
packaging are development steps; users work with the prebuilt JAR.

## Distribution status

A prebuilt JAR is available locally in `dist/`; a public download has not been
published yet. Remaining work before public distribution:

- Reduce the artifact to under 2 GiB for single-file GitHub Releases distribution,
  and validate audio quality and CPU/GPU compatibility after optimization.
- Publish the JAR with its SHA-256 checksum and add the download link here.

## Use the prebuilt JAR

Install **Java 17 or newer** and obtain the prebuilt
`uzbek-tts-0.1.0.jar` distributed with this project. In the development checkout,
the ready artifact is located in `dist/`. Generated JARs and model binaries are
excluded from the source repository; cloning the source alone does not include
the ready artifact.

Using the JAR does not require Python, PyTorch, Maven, Docker, Hugging Face
downloads, or a separate `chatterbox-finetuning` checkout.

From the directory containing the JAR:

```bash
java -jar uzbek-tts-0.1.0.jar --text "Salom, yaxshimisiz?" --output speech.wav
```

This creates a **24 kHz, mono, PCM16 WAV** file using the bundled default voice.
For a UTF-8 text file:

```bash
java -jar uzbek-tts-0.1.0.jar --text-file input.txt --output speech.wav
```

On first use, the models are extracted from the JAR into
`~/.cache/uzbek-tts/`. This uses about 2.68 GB of additional disk space. Later
runs reuse the cache. Speech generation works offline; the application does not
download models at runtime. Use `--cache DIR` to choose a different cache location.

## CPU and GPU selection

The default is **`auto`**: prefer CUDA, then Apple GPU, then CPU. If GPU session
initialization fails, automatic mode falls back to CPU and reports the reason.
The device is selected once at startup.

| Option | Device |
|---|---|
| `--device auto` | Automatic selection; default |
| `--device cpu` | CPU |
| `--device mps` | Apple GPU through native WebGPU/Metal on macOS |
| `--device cuda` | NVIDIA GPU 0 on supported Linux/Windows x64 systems |

To explicitly use the CPU:

```bash
java -jar uzbek-tts-0.1.0.jar --device cpu --text "Salom!" --output speech.wav
```

CUDA requires compatible NVIDIA drivers, CUDA 12.8+ and cuDNN 9 libraries on the
host. Some GPU operators may run on the CPU. CPU and Apple GPU inference have
been tested on macOS ARM64; NVIDIA hardware execution has not been tested here.

List runtime providers or view all command-line options:

```bash
java -jar uzbek-tts-0.1.0.jar --list-devices
java -jar uzbek-tts-0.1.0.jar --help
```

## Use it from other applications

Start the HTTP server:

```bash
java -jar uzbek-tts-0.1.0.jar --serve --port 8000
```

After the server reports that it is ready, send text from another terminal:

```bash
curl --fail http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"uzbek-chatterbox","input":"Salom!","voice":"default","response_format":"wav"}' \
  --output speech.wav
```

Java, Spring Boot, Python, JavaScript, Go, and other applications can call this
endpoint with a normal HTTP client and save the response as a WAV file.
`GET /ready` reports server readiness and the selected device. Stop the server
with Ctrl+C.

For direct Java integration, add the supplied JAR to your application's
classpath:

```java
try (var tts = new uz.academix.tts.UzbekTts()) {
    byte[] wav = tts.synthesize("Salom, yaxshimisiz?");
    java.nio.file.Files.write(java.nio.file.Path.of("speech.wav"), wav);
}
```

In a long-running application, create the model once, reuse it across requests,
and close it when the application shuts down.

## Current scope

The package provides one bundled default voice and WAV output. Voice cloning,
MP3 output, and real-time streaming are not included in this version.

## Credits and licenses

Based on Chatterbox and the Uzbek fine-tuned model. See
[THIRD_PARTY.md](THIRD_PARTY.md) for source attribution,
[LICENSE](LICENSE) for the project license,
[LICENSE.models](LICENSE.models) for the model license, and
[LICENSE.onnxruntime](LICENSE.onnxruntime) for the ONNX Runtime license.
