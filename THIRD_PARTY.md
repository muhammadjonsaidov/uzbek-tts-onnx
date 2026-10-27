# Third-party sources

- Chatterbox model code and weights: Resemble AI, MIT. The vendored toolkit is from https://github.com/gokhaneraslan/chatterbox-finetuning (Apache-2.0; included license).
- ONNX wrappers and numpy alignment analyzer adapted from https://github.com/gitbisector/chatterbox_onnx_conversion_scripts at e6fae93dbd7f33721bc76ae44796b440234ee9df, credited to Vladislav Bronzov and contributors (MIT per upstream README).
- Decoder adapters expose random noise as inputs for deterministic parity checks. Embedding adapter preserves this toolkit's CFG positional-embedding behavior.

- ONNX Runtime Java `OrtEnvironment.java`: Microsoft/Oracle and contributors, MIT, from v1.23.2. A local shutdown-order patch closes registered TTS sessions before the native environment releases shared WebGPU resources. The matching upstream class is excluded from the shaded dependency. All other runtime classes/native libraries retain upstream version 1.23.2.
