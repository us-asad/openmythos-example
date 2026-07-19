"""OpenMythos Example — instrumented inference server.

Streams per-token generation events over SSE, including:
- per-pass guesses (parallel streams run at n_loops=1..depth — the guess at pass k
  is literally what the model would say if it stopped reasoning after k loops)
- final top-5 candidate distribution
- MoE expert routing (router hook)
- attention targets (attention-matrix hook)
"""
import json
import os
import threading

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from open_mythos import OpenMythos
from open_mythos.tokenizer import MythosTokenizer

REPO = "asadusmonov/openmythos-smol-282m"
CKPT = "openmythos_smol_282m.pt"
MAX_CTX = 1000          # model max_seq_len is 1024; keep headroom
MAX_NEW_DEFAULT = 60

app = FastAPI()
LOCK = threading.Lock()   # one generation at a time (single CPU worker)
model = None
tok = None
cap = None


class Capture:
    """Forward hooks on the recurrent block's router + attention."""

    def __init__(self, m, cfg):
        self.router_logits = None
        self.attn = None
        self.topk = getattr(cfg, "n_experts_per_tok", 4)
        self.router_bias = None
        for name, buf in m.named_buffers():
            if "recurrent" in name and name.endswith("router_bias"):
                self.router_bias = buf
        for name, mod in m.named_modules():
            if "recurrent" not in name:
                continue
            if name.endswith(".router"):
                mod.register_forward_hook(self._save_router)
            if name.endswith(".attn_drop"):
                mod.register_forward_hook(self._save_attn)

    def _save_router(self, module, inp, out):
        self.router_logits = out.detach()

    def _save_attn(self, module, inp, out):
        self.attn = out.detach()

    def experts(self):
        if self.router_logits is None:
            return []
        logits = self.router_logits.reshape(-1, self.router_logits.shape[-1])[-1]
        biased = logits + (self.router_bias if self.router_bias is not None else 0)
        idx = biased.topk(self.topk).indices
        scores = F.softmax(logits, dim=-1)
        w = scores[idx]
        w = w / w.sum()
        return [{"e": int(i), "w": round(float(x), 3)} for i, x in zip(idx.tolist(), w.tolist())]

    def attention_targets(self, all_ids, prompt_len, k=3):
        if self.attn is None:
            return []
        a = self.attn
        while a.dim() > 1:
            a = a.mean(dim=0) if a.dim() > 2 else a[-1]
        n = min(a.shape[-1], len(all_ids))
        a = a[:n]
        if n <= 1:
            return []
        vals, idx = a.topk(min(k, n))
        s = float(vals.sum()) or 1.0
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            text = decode_ids([all_ids[i]]).strip()
            if not text:
                continue
            out.append({"text": text, "w": round(v / s, 2), "rel": i - prompt_len})
        return out


def decode_ids(ids):
    try:
        return tok.decode(ids)
    except AttributeError:
        return tok.tokenizer.decode(ids)


@app.on_event("startup")
def load_model():
    global model, tok, cap
    path = hf_hub_download(REPO, CKPT)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = OpenMythos(cfg)
    sd = {
        k: (v.to(torch.float32) if v.is_floating_point() else v)
        for k, v in ck["model"].items()
    }
    m.load_state_dict(sd)
    m.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok_local = MythosTokenizer()
    model = m
    tok = tok_local
    cap = Capture(m, cfg)


def generate_events(prompt: str, depth: int, max_new: int, temperature=0.8, top_k=50):
    with LOCK, torch.no_grad():
        ids = tok.encode(prompt)[-MAX_CTX // 2 :]
        prompt_len = len(ids)
        x = torch.tensor([ids], dtype=torch.long)

        # one stream per reasoning budget k=1..depth; stream `depth` is canonical
        streams = [{"n": k, "cache": {}} for k in range(1, depth + 1)]
        for s in streams:
            logits = model(x, n_loops=s["n"], kv_cache=s["cache"], start_pos=0)
            s["last"] = logits[0, -1].float()

        all_ids = list(ids)
        pos = len(ids)

        for _ in range(max_new):
            if pos >= MAX_CTX:
                break

            passes = []
            for s in streams:
                probs = F.softmax(s["last"], dim=-1)
                p, i = probs.max(dim=-1), probs.argmax(dim=-1)
                g = decode_ids([int(i)]).strip()
                passes.append({"guess": g or "·", "conf": round(float(probs.max()), 3)})

            final = streams[-1]["last"].clone()
            experts = cap.experts()
            looks = cap.attention_targets(all_ids, prompt_len)

            logits_t = final / temperature
            if top_k > 0:
                v, _ = logits_t.topk(top_k)
                logits_t[logits_t < v[-1]] = float("-inf")
            probs = F.softmax(logits_t, dim=-1)
            nxt = int(torch.multinomial(probs, 1))

            tp, ti = probs.topk(5)
            top5 = []
            for p, t in zip(tp.tolist(), ti.tolist()):
                w = decode_ids([t]).strip()
                if w:
                    top5.append({"w": w, "p": round(p, 3), "win": t == nxt})

            text = decode_ids([nxt])
            yield {
                "type": "token",
                "text": text,
                "passes": passes,
                "top5": top5,
                "experts": experts,
                "looks": looks,
                "chosen_p": round(float(probs[nxt]), 3),
            }

            all_ids.append(nxt)
            step = torch.tensor([[nxt]], dtype=torch.long)
            for s in streams:
                logits = model(step, n_loops=s["n"], kv_cache=s["cache"], start_pos=pos)
                s["last"] = logits[0, -1].float()
            pos += 1

        yield {"type": "done"}


@app.get("/api/generate")
def api_generate(request: Request, prompt: str, depth: int = 4, max_new: int = MAX_NEW_DEFAULT):
    depth = max(1, min(4, depth))
    max_new = max(1, min(120, max_new))

    def sse():
        try:
            for ev in generate_events(prompt, depth, max_new):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface errors to the client instead of dying silently
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:200]})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    return {"ok": model is not None}


@app.get("/")
def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
