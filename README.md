# OpenMythos Example — watch a language model think

A chat interface for a **282M-parameter recurrent-depth language model**, where the
model's reasoning is visible in real time: every word it speaks is preceded by an
internal loop you can watch, slow down, inspect, and even switch off.

The model was trained from scratch (50M tokens of FineWeb-Edu, a single RTX 4090,
~2 hours) — it speaks grammatical nonsense with total confidence. That is the point:
**the purpose is observability, not capability.** Same machinery as large models,
tiny scale, glass walls.

> Weights: [asadusmonov/openmythos-smol-282m](https://huggingface.co/asadusmonov/openmythos-smol-282m) ·
> Architecture: [OpenMythos](https://github.com/kyegomez/OpenMythos) (recurrent depth + MoE + MLA)

---

## What you can see

| In the UI | Where it comes from |
|---|---|
| **Ghost word solidifying** in the sentence while it thinks | The reasoning loop is run at every budget 1…4 in parallel — the pass-*k* guess is literally what the model would say if it stopped thinking after *k* passes |
| **Live tooltip** with pass dots + shifting candidate list | Per-budget next-token distributions, decoded on the fly |
| **Word tint** (pale → warm) | How unstable the answer was across passes + final sampling uncertainty |
| **Token record** (click any word) | Answer evolution across passes, attention targets (hooked from the attention matrix), MoE expert routing (hooked from the router), final top-5 distribution |
| **Reasoning passes 1–4** in settings | `n_loops` is passed straight into the model's forward — reducing it is a true ablation, and the output visibly degrades |
| **Generation speed** slider | Client-side playback pacing of the SSE event stream |

Nothing is simulated. Every number in the interface is read from the forward pass.

## How it works

```
Browser ──SSE──▶ FastAPI (app.py)
                   │  4 parallel KV-cached streams, n_loops = 1,2,3,4
                   │  forward hooks: MoE router logits, attention matrix
                   ▼
              OpenMythos (282M, prelude → recurrent loop → coda)
```

Per generated token the server emits one event carrying: per-pass guesses and
confidences, top-5 candidates, expert routing weights, attention targets, and the
sampled token. The frontend (a single `index.html`, no build step) animates the
stream and stores each token's record for inspection.

## Run locally

Requires Python 3.11+ and ~3GB RAM. The model (~930MB) downloads automatically
from Hugging Face on first start.

```bash
git clone https://github.com/<you>/openmythos-example.git
cd openmythos-example
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://localhost:7860**. First reply after a cold start is slower (weights
loading); afterwards expect ~1–2 words/sec on a laptop CPU — the thinking
animation is designed to make that latency the show.

### Docker

```bash
docker build -t openmythos-example .
docker run -p 7860:7860 openmythos-example
```

## Things to try

1. Open settings (⚙) → drag speed toward **observe** → ask something → read the
   model's mind changing between passes.
2. Set **reasoning passes to 1** and ask again — same weights, visibly dumber
   output. That's the recurrent loop doing its job, ablated live.
3. Click a warm-tinted word → its Token record usually shows the answer flipping
   between passes.

## Project structure

```
app.py            FastAPI server + instrumented multi-budget generation
index.html        entire frontend (no build step)
requirements.txt  CPU-only torch + model deps
Dockerfile        container build for any VPS
```

## Acknowledgements

- [OpenMythos](https://github.com/kyegomez/OpenMythos) — the architecture implementation this model uses
- Recurrent-depth / latent reasoning line of research that inspired the "thinks before it speaks" design
- FineWeb-Edu (HuggingFaceFW) — training data

## License

MIT
