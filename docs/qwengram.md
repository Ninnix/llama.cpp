# QwenGram 0.8B and 2B

The 0.8B path runs the REAL-15M R=1 reader and its linear750 arbiter. The fork inserts the reader before Qwen3.5 decoder layers 2 and 8 (zero based), reads the 16 PLE rows for each token from an external GGUF, and evaluates the reader and gate in the model graph. The PLE is mapped on the host and only selected rows are dequantized; it is not copied to the GPU. See the [0.8B model files](https://huggingface.co/Ninnix96/Qwengram-0.8B), the 2B section below, and the [study](https://github.com/Ninnix/qwen-ple-transfer).

The model GGUFs contain the matching Qwen3.5 backbone, two reader projections per site, trained beta and gamma values, alpha2, and the IDX8 gate. They do not contain the PLE.

## Qwengram-2B

The [2B release](https://huggingface.co/Ninnix96/Qwengram-2b) uses REAL-10M + linear750 as its canonical balanced endpoint. A matched comparison of persisted 10M and 15M readers found clearer NLL gains at 15M but unresolved benchmark accuracy differences. The 15M reader and arbiter remain available as the max-LM/research endpoint. The release includes the paired intervals and both checkpoints.

Both models have 24 decoder layers and inject at IDX2/IDX8. The reader width follows the backbone: 1024 for 0.8B and 2048 for 2B. The query/key score is divided by sqrt(hidden width). Addressing, row order, reader projections, residual placement and arbitration formulas are unchanged. The loader checks the reader dimensions against the backbone.

Use `QwenGram-2B-BF16.gguf`, `QwenGram-2B-Q8_0.gguf`, or `QwenGram-2B-Q4_K_M.gguf` with the same external PLE below. All 11 reader and gate tensors stay F32 in each GGUF. File hashes and matched CPU stock-versus-Qwengram measurements are included in the release. The Q4_1 sidecar validation reproduces 4,096 addressed rows exactly under full prefill, split prefill and token decoding, including EOS and position-zero resets. These checks concern the Q4_1 sidecar, not equality with the original FP8 memory.

```sh
export QWENGRAM_PLE="$PWD/models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
build-qwengram-cpu/bin/llama-completion -m models/qwengram/QwenGram-2B-Q8_0.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

The remaining file hashes and BC-250 offload observations below refer to the 0.8B release.

## Files

Put these files under `models/qwengram/` (excluded from this repository's local Git index):

| File | SHA-256 |
| --- | --- |
| `QwenGram-0.8B-BF16.gguf` | `8eee4d1061fcd9d60f223e50e510183d0818758b381ebf22ca192b0a6271318a` |
| `QwenGram-0.8B-Q8_0.gguf` | `ec5f65a28be248b41f981f1be3074dcb09a52987977bcda8d1c352711f01c043` |
| `QwenGram-0.8B-Q6_K.gguf` | `841d67b233af609544e56d2ce450b008d3f754a68120ab596c6a175562b8998b` |
| `QwenGram-0.8B-Q4_K_M.gguf` | `77500ea47628c2a40155a4d2b6468a1a4aec4950de3bc717268704e40be52e33` |
| `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` | `66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1` |

The BF16, Q8_0, and Q4_K_M models are from the [QwenGram release](https://huggingface.co/Ninnix96/Qwengram-0.8B). Q6_K was made locally from that BF16 GGUF with llama.cpp's quantizer, retaining the 11 QwenGram tensors as F32. The external PLE is [Ivan Fioravanti's Q4_1 GGUF](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4/blob/main/Qwen3.8-Flash-Next-PLE-Q4_1.gguf); credit and source belong to Ivan.

## Build and run

```sh
cmake -S . -B build-qwengram-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-qwengram-cpu -j --target llama-completion
export QWENGRAM_PLE="$PWD/models/qwengram/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
build-qwengram-cpu/bin/llama-completion -m models/qwengram/QwenGram-0.8B-Q4_K_M.gguf -p 'The capital of France is' -n 16 -no-cnv -ngl 0
```

For Vulkan, configure a separate build with `-DGGML_VULKAN=ON` and run its `llama-completion` with `-ngl 99`. The external PLE remains host mapped. On the tested AMD BC-250, BF16, Q8_0, and Q6_K matched the CPU first token for the smoke prompt. Q4_K_M matched with `-ngl 25`, but selected a different first token with `-ngl 26` or `-ngl 99`. Use `-ngl 25` for Q4_K_M on this device until broader Vulkan checks establish its full-offload behavior.

The loader requires `QWENGRAM_PLE` for a QwenGram model and checks its Q4_1 row layout, 16 head sizes and offsets, and three hash multipliers. It rejects a missing or incompatible PLE. The sequence input currently requires token IDs and positions; embedding-only input and the extra MTP graph are unsupported. The Q4_1 PLE and GGUF inference have different numeric precision from the original FP8 PLE and PyTorch evaluation, so the published held-out scores are not GGUF parity claims.

The [GGUF runtime validation](https://github.com/Ninnix/qwen-ple-transfer/tree/main/qwen35-08b/gguf-runtime) compares each release GGUF with an exact stock-backbone control on a fixed WikiText-2 slice and reports Q8_0 and Q4_K_M reader-gain retention with paired intervals.
