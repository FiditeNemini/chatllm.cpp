# ChatLLM.cpp

[中文版](README_zh.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![](./docs/demo.gif)

Inference of a bunch of models from less than 3B to more than 300B, for real-time chatting with [RAG](./docs/rag.md) on your computer (CPU),
pure C++ implementation based on [@ggerganov](https://github.com/ggerganov)'s [ggml](https://github.com/ggerganov/ggml).

| [Supported Models](./docs/models.md) | [Download Quantized Models](https://modelscope.cn/models/judd2024/chatllm_quantized_models) |

**What's New:**

* 2024-04-30: Phi3-mini 128k
* 2024-04-27: Phi3-mini 4k

## Features

* [x] Accelerated memory-efficient CPU inference with int4/int8 quantization, optimized KV cache and parallel computing;
* [x] Use OOP to address the similarities between different _Transformer_ based models;
* [x] Streaming generation with typewriter effect;
* [x] Continuous chatting (content length is virtually unlimited)

    Two methods are available: _Restart_ and _Shift_. See `--extending` options.

* [x] Retrieval Augmented Generation (RAG) 🔥

* [ ] LoRA;
* [x] Python/JavaScript/C [Bindings](./docs/binding.md), web demo, and more possibilities.

## Usage

### Preparation

Clone the ChatLLM.cpp repository into your local machine:

```sh
git clone --recursive https://github.com/foldl/chatllm.cpp.git && cd chatllm.cpp
```

If you forgot the `--recursive` flag when cloning the repository, run the following command in the `chatllm.cpp` folder:

```sh
git submodule update --init --recursive
```

### Quantize Model

**Some quantized models can be downloaded from [here](https://modelscope.cn/models/judd2024/chatllm_quantized_models).**

Install dependencies of `convert.py`:

```sh
pip install -r requirements.txt
```

Use `convert.py` to transform models into quantized GGML format. For example, to convert the _fp16_ base model to q8_0 (quantized int8) GGML model, run:

```sh
# For models such as ChatLLM-6B, ChatLLM2-6B, InternLM, LlaMA, LlaMA-2, Baichuan-2, etc
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin

# For some models such as CodeLlaMA, model type should be provided by `-a`
# Find `-a ...` option for each model in `docs/models.md`.
python3 convert.py -i path/to/model -t q8_0 -o quantized.bin -a CodeLlaMA
```

Note: Appropriately, only HF format is supported; Format of the generated `.bin` files is different from the one (GGUF) used by `llama.cpp`.

### Build & Run

Compile the project using CMake:

```sh
cmake -B build
# On Linux, WSL:
cmake --build build -j
# On Windows with MSVC:
cmake --build build -j --config Release
```

Now you may chat with a quantized model by running:

```sh
./build/bin/main -m chatglm-ggml.bin                            # ChatGLM-6B
# 你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。
./build/bin/main -m llama2.bin  --seed 100                      # Llama-2-Chat-7B
# Hello! I'm here to help you with any questions or concerns ....
```

To run the model in interactive mode, add the `-i` flag. For example:

```sh
# On Windows
.\build\bin\Release\main -m model.bin -i

# On Linux (or WSL)
rlwrap ./build/bin/main -m model.bin -i
```

In interactive mode, your chat history will serve as the context for the next-round conversation.

Run `./build/bin/main -h` to explore more options!

## Acknowledgements

* This project is started as refactoring of [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp), without which, this project could not be possible.

* Thank those who have released their the model sources and checkpoints.

## Note

This project is my hobby project to learn DL & GGML, and under active development. PRs of features won't
be accepted, while PRs for bug fixes are warmly welcome.