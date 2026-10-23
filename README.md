# LLM Lab

An interactive, visual laboratory that explains **how a Large Language Model works**, from raw text to training, one piece at a time.
Every module follows the same rhythm: *intuition → hands-on lab → the math → quick quiz*.

The explanations are written in Italian (technical terms are kept in English, e.g. *attention*, *embedding*, *logits*).
The app is a single self-contained HTML file: no server, no build tooling, no runtime dependencies.

## Modules

| # | Module | What you can play with |
|---|--------|------------------------|
| 0 | The pipeline | Tensor shapes at every stage, parameter calculator (GPT-2, Llama-style presets) |
| 1 | Tokenization | A Byte-Pair Encoding trained live on text you edit, step by step |
| 2 | Embeddings | 3D embedding space, the lookup table, cosine similarity, word analogies |
| 3 | Positions | Why attention is order-blind, sinusoidal encoding, RoPE rotations |
| 4 | Self-attention | Q/K/V with editable weights, every step of softmax(QKᵀ/√d)V, why √d |
| 5 | Multi-head attention | Head patterns, induction heads built from two composed heads |
| 6 | MLP, residual, LayerNorm | Function fitting with ReLU/GELU, gradient flow through depth, normalization |
| 7 | Sampling | Temperature, top-k, top-p on a live distribution |
| 8 | Training | Cross-entropy, backprop by hand, optimizers on a 3D loss landscape, a language model trained in your browser |
| 9 | KV cache and inference | Prefill vs decode, cache memory (MHA vs GQA), a memory-bound throughput model, speculative decoding |
| 10 | SFT, RLHF, DPO | Loss masking, a reward model with reward hacking under a KL penalty, DPO from the preference pairs |
| 11 | Mixture of Experts | A trainable top-k router, load balancing and capacity, active vs total parameters, batching vs expert reads |
| 12 | Scaling laws | The Chinchilla loss surface L(N, D), the compute-optimal allocation in closed form (two published fits compared), over-training when inference dominates |
| 13 | Quantization | A real round-to-nearest quantizer: bits, per-tensor vs per-group scales, outliers, weight memory |
| 14 | Interpretability | A toy model of superposition trained in the page (5 features in 2 dimensions), linear probes on a real small LM with controls |
| 15 | RAG, tools, agents | Real BM25 / TF-IDF retrieval, chunking sweep, a safe calculator, an agent loop (rule-based planner standing in for the LLM) |
| 16 | The complete transformer | A decoder-only transformer with hand-written backprop trained in the browser: loss curves, attention maps, generation |

The course is complete as planned. Ideas for later: long context and state-space models, multimodality, evaluation and benchmarks, safety and alignment in depth.

## Run it

```bash
python3 build.py          # bundles src/ into dist/index.html
open dist/index.html      # or just double-click the file
```

A pre-built `dist/index.html` is committed, so you can open it right away. Web fonts load from Google Fonts when online and fall back to system fonts otherwise.

## Project layout

```
src/index.html      page template
src/style.css       design tokens (light/dark) and components
src/js/core.js      DOM helpers, canvas/chart/heatmap/3D camera, linear algebra, quiz
src/js/m*.js        one file per module
src/js/main.js      app shell (the left rail is the model stack)
build.py            concatenates everything into dist/index.html
tests/              dependency-free checks (fake DOM + numerical tests)
```

## Tests

```bash
node tests/logic.js    # BPE, parameter counts, analogies, attention properties...
node tests/smoke.js    # runs every module in a fake DOM and pokes every control
```

## Honest simplifications

The labs favour clarity over fidelity: the word vectors in Module 2 and the head patterns in Module 5 are hand-built to show what *can* happen,
while the training demo in Module 8 is a real (small) neural language model with hand-written backpropagation. Modules 10 and 11 use small hand-defined toy setups (12 candidate answers; 240 points in a plane), module 15 a small invented knowledge base, so the mechanisms can be seen and measured; the tests check the gradients and the closed-form results. Module 12 uses published scaling-law fits whose constants are debated, and says so. In module 15 the agent's planner is a set of hand-written rules, not an LLM. Each lab says which is which.

## Versioning

Semantic-ish tags (`v0.1.0` = core modules 0–8, `v0.2.0` = modules 9–11, `v0.3.0` = modules 12–16, the full course). Changes are tracked per module in the commit history.
