#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyAI v2.1 — تشغيل عادي بأمر واحد:
    python3 arqam.py

إذا شغّلته بدون أي أمر:
  1. يجمع بيانات (إن لم تكن موجودة)
  2. يبني tokenizer (إن لم يكن موجوداً)
  3. يدرّب الموديل (إن لم يكن موجوداً)
  4. يفتح واجهة ويب على المنفذ المحدد

للأوامر المتقدمة:
    python3 arqam.py setup --limit 5000
    python3 arqam.py chat
    python3 arqam.py serve --port 8080
    python3 arqam.py info
"""

import argparse
import asyncio
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  [0] إعدادات التشغيل التلقائي (عدّلها لتغيير السلوك)
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  [0] إعدادات التشغيل التلقائي
# ═══════════════════════════════════════════════════════════════════

AUTO_RUN = os.environ.get("MYAI_AUTO_RUN", "1") == "1"
AUTO_LIMIT = int(os.environ.get("MYAI_LIMIT", "2000"))
AUTO_HOST = os.environ.get("MYAI_HOST", "0.0.0.0")
AUTO_PORT = int(os.environ.get("MYAI_PORT", "24686"))
AUTO_TRAIN = os.environ.get("MYAI_AUTO_TRAIN", "1") == "1"

# HuggingFace token — مباشرة في الكود
HF_TOKEN = "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"

# مرّره لـ HuggingFace
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print(f"[+] HF_TOKEN loaded: {HF_TOKEN[:8]}...")


# ═══════════════════════════════════════════════════════════════════
#  [1] استيرادات كسولة
# ═══════════════════════════════════════════════════════════════════

def lazy_torch():
    try:
        import torch
        return torch
    except ImportError:
        print("[!] pip install torch", file=sys.stderr)
        sys.exit(1)


def lazy_dataset():
    try:
        from datasets import load_dataset, concatenate_datasets
        return load_dataset, concatenate_datasets
    except ImportError:
        print("[!] pip install datasets", file=sys.stderr)
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
#  [2] Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    work_dir: str = "myai_data"
    corpus_path: str = "myai_data/corpus.txt"
    tokenizer_path: str = "myai_data/tokenizer.json"
    model_path: str = "myai_data/model.pt"
    docs_dir: str = "myai_data/docs"
    interactions_path: str = "myai_data/interactions.jsonl"
    feedback_path: str = "myai_data/feedback.jsonl"
    memory_path: str = "myai_data/memory.jsonl"

    vocab_size: int = 16000
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    max_len: int = 128
    dropout: float = 0.1

    block_size: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 2
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    interaction_min_score: float = 0.3
    interaction_max_in_context: int = 3
    retrain_min_positive: int = 50
    retrain_min_rating: int = 1

    @property
    def model_size(self) -> int:
        embed = self.vocab_size * self.d_model
        pos = self.max_len * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model
            + 8 * self.d_model * self.d_model
            + 4 * self.d_model
        )
        head = self.d_model * self.vocab_size
        return embed + pos + self.n_layers * per_layer + head


CFG = Config()


# ═══════════════════════════════════════════════════════════════════
#  [3] Data collection
# ═══════════════════════════════════════════════════════════════════

def collect_data(limit: int = 50_000):
    print("[*] collecting data...")
    load_dataset, concatenate_datasets = lazy_dataset()

    sources = []
    try:
        print("  - Wikipedia Arabic")
        wiki = load_dataset(
            "wikimedia/wikipedia", "20231101.ar",
            split=f"train[:{limit}]",
        )
        sources.append(wiki)
    except Exception as e:
        print(f"    failed: {e}")

    try:
        print("  - Arabic 100k")
        a100k = load_dataset(
            "Omartificial-Intelligence-Space/Arabic_100k_dataset",
            split="train",
        )
        if hasattr(a100k, "select"):
            a100k = a100k.select(range(min(limit, len(a100k))))
        sources.append(a100k)
    except Exception as e:
        print(f"    failed: {e}")

    if not sources:
        print("[!] no sources loaded. Creating dummy data.")
        texts = ["مرحباً بك في العالم. " * 20] * 100
    else:
        print(f"[*] merging {len(sources)} sources...")
        combined = concatenate_datasets(sources) if len(sources) > 1 else sources[0]

        def norm(ex):
            for k in ("text", "content", "article", "body"):
                if k in ex and ex[k]:
                    return {"text": str(ex[k])}
            return {"text": ""}

        combined = combined.map(norm, remove_columns=combined.column_names)

        def clean(ex):
            t = ex["text"]
            t = re.sub(r"http\S+", "", t)
            t = re.sub(r"\s+", " ", t).strip()
            return {"text": t}

        print("[*] cleaning...")
        combined = combined.filter(lambda x: 200 < len(x["text"]) < 100_000)
        combined = combined.map(clean)
        combined = combined.filter(lambda x: len(x["text"]) > 200)
        texts = combined["text"]

    os.makedirs(CFG.work_dir, exist_ok=True)
    with open(CFG.corpus_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n\n")

    size_mb = os.path.getsize(CFG.corpus_path) / 1024 / 1024
    print(f"[+] corpus: {CFG.corpus_path} ({size_mb:.1f} MB, {len(texts)} docs)")
    return len(texts)


# ═══════════════════════════════════════════════════════════════════
#  [4] Tokenizer
# ═══════════════════════════════════════════════════════════════════

def train_tokenizer():
    print("[*] training tokenizer...")
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
        from tokenizers.normalizers import NFKC
    except ImportError:
        print("[!] pip install tokenizers", file=sys.stderr)
        sys.exit(1)

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
    tok.normalizer = NFKC()

    trainer = BpeTrainer(
        vocab_size=CFG.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )

    def iterator():
        with open(CFG.corpus_path, "r", encoding="utf-8") as f:
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= 1000:
                    yield "".join(buf)
                    buf = []
            if buf:
                yield "".join(buf)

    tok.train_from_iterator(iterator(), trainer=trainer)
    tok.save(CFG.tokenizer_path)
    print(f"[+] tokenizer: {CFG.tokenizer_path} (vocab={tok.get_vocab_size()})")

    sample = tok.encode("مرحباً بك في العالم")
    print(f"    test: {sample.tokens[:10]}")


# ═══════════════════════════════════════════════════════════════════
#  [5] Model
# ═══════════════════════════════════════════════════════════════════

class ModelBuilder:
    _model_class = None

    @classmethod
    def build(cls):
        if cls._model_class is not None:
            return cls._model_class

        torch = lazy_torch()
        import torch.nn as nn
        import torch.nn.functional as F

        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, n_heads, max_len, dropout):
                super().__init__()
                assert d_model % n_heads == 0
                self.n_heads = n_heads
                self.d_head = d_model // n_heads
                self.W_q = nn.Linear(d_model, d_model, bias=False)
                self.W_k = nn.Linear(d_model, d_model, bias=False)
                self.W_v = nn.Linear(d_model, d_model, bias=False)
                self.W_o = nn.Linear(d_model, d_model, bias=False)
                self.dropout = nn.Dropout(dropout)
                self.register_buffer(
                    "mask",
                    torch.tril(torch.ones(max_len, max_len))
                         .view(1, 1, max_len, max_len),
                )

            def forward(self, x):
                B, T, C = x.shape
                q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.dropout(att)
                out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
                return self.W_o(out)

        class FeedForward(nn.Module):
            def __init__(self, d_model, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                return self.net(x)

        class Block(nn.Module):
            def __init__(self, d_model, n_heads, max_len, dropout):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = MultiHeadAttention(d_model, n_heads, max_len, dropout)
                self.ln2 = nn.LayerNorm(d_model)
                self.ff = FeedForward(d_model, dropout)

            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                x = x + self.ff(self.ln2(x))
                return x

        class MyGPT(nn.Module):
            def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, dropout):
                super().__init__()
                self.max_len = max_len
                self.tok_emb = nn.Embedding(vocab_size, d_model)
                self.pos_emb = nn.Embedding(max_len, d_model)
                self.drop = nn.Dropout(dropout)
                self.blocks = nn.Sequential(*[
                    Block(d_model, n_heads, max_len, dropout)
                    for _ in range(n_layers)
                ])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size, bias=False)
                self.apply(self._init)

            def _init(self, m):
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)

            def forward(self, idx, targets=None):
                B, T = idx.shape
                pos = torch.arange(T, device=idx.device)
                x = self.tok_emb(idx) + self.pos_emb(pos)
                x = self.drop(x)
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.head(x)
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                    )
                return logits, loss

            @torch.no_grad()
            def generate(self, idx, max_new=100, temperature=0.8, top_k=40):
                for _ in range(max_new):
                    idx_crop = idx[:, -self.max_len:]
                    logits, _ = self(idx_crop)
                    logits = logits[:, -1, :] / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, top_k)
                        logits[logits < v[:, [-1]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                    idx = torch.cat([idx, nxt], dim=1)
                return idx

        cls._model_class = MyGPT
        return MyGPT


# ═══════════════════════════════════════════════════════════════════
#  [6] Dataset
# ═══════════════════════════════════════════════════════════════════

class DatasetBuilder:
    @staticmethod
    def build():
        torch = lazy_torch()
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, ids, block_size):
                self.samples = []
                for i in range(0, len(ids) - block_size - 1, block_size):
                    self.samples.append(ids[i : i + block_size + 1])

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                chunk = self.samples[i]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                return x, y

        return TextDataset


# ═══════════════════════════════════════════════════════════════════
#  [7] Training
# ═══════════════════════════════════════════════════════════════════

def train_model(resume: bool = True, extra_corpus: Optional[str] = None):
    torch = lazy_torch()
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader

    if not os.path.exists(CFG.tokenizer_path):
        print("[!] tokenizer not found. Run: python3 arqam.py tokenize")
        return
    tok = Tokenizer.from_file(CFG.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    print("[*] tokenizing corpus...")
    t0 = time.time()
    if not os.path.exists(CFG.corpus_path):
        print("[!] corpus not found. Run: python3 arqam.py collect")
        return

    with open(CFG.corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    if extra_corpus and os.path.exists(extra_corpus):
        print(f"[*] adding extra corpus: {extra_corpus}")
        with open(extra_corpus, "r", encoding="utf-8") as f:
            text += "\n\n" + f.read()

    print(f"    corpus size: {len(text):,} chars")
    enc = tok.encode(text)
    ids = enc.ids
    print(f"    {len(ids):,} tokens in {time.time()-t0:.1f}s")

    D = DatasetBuilder.build()
    ds = D(ids, CFG.block_size)
    print(f"    {len(ds):,} training samples")

    loader = DataLoader(
        ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    MyGPT = ModelBuilder.build()
    model = MyGPT(
        vocab_size=vocab_size,
        d_model=CFG.d_model,
        n_layers=CFG.n_layers,
        n_heads=CFG.n_heads,
        max_len=CFG.max_len,
        dropout=CFG.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay,
    )

    total_steps = len(loader) * CFG.epochs
    warmup = max(1, int(total_steps * CFG.warmup_ratio))

    def lr_at(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    start_step = 0
    if resume and os.path.exists(CFG.model_path):
        try:
            ckpt = torch.load(CFG.model_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            print(f"[*] resumed from step {start_step}")
        except Exception as e:
            print(f"[!] resume failed: {e}")

    model.train()
    step = start_step
    t_last = time.time()
    loss_avg = 0.0
    loss_n = 0

    for epoch in range(CFG.epochs):
        print(f"\n=== Epoch {epoch+1}/{CFG.epochs} ===")
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            lr = CFG.learning_rate * lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

            loss_avg += loss.item()
            loss_n += 1
            step += 1

            if step % 50 == 0:
                dt = time.time() - t_last
                tps = (50 * CFG.batch_size * CFG.block_size) / dt if dt > 0 else 0
                avg = loss_avg / max(1, loss_n)
                print(f"  step {step:6d} | loss {loss.item():.4f} | avg {avg:.4f} | lr {lr:.2e} | {tps:.0f} tok/s")
                loss_avg = 0.0
                loss_n = 0
                t_last = time.time()

            if step % 500 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }, CFG.model_path)
                print(f"  [saved] {CFG.model_path}")

                model.eval()
                try:
                    bos_id = tok.token_to_id("<bos>") or 0
                    prompt = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                    out = model.generate(prompt, max_new=30, temperature=0.8)
                    sample = tok.decode(out[0].tolist())
                    print(f"  [sample] {sample[:200]}")
                except Exception as e:
                    print(f"  [sample failed] {e}")
                model.train()

    torch.save({
        "model": model.state_dict(),
        "step": step,
    }, CFG.model_path)
    print(f"\n[+] training complete. model: {CFG.model_path}")


# ═══════════════════════════════════════════════════════════════════
#  [8] Chat Engine
# ═══════════════════════════════════════════════════════════════════

class ChatEngine:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.torch = None
        self.tok = None
        self.model = None
        self.device = None

    def load(self):
        if self.model is not None:
            return
        torch = lazy_torch()
        from tokenizers import Tokenizer

        if not os.path.exists(CFG.tokenizer_path):
            raise RuntimeError(f"tokenizer not found: {CFG.tokenizer_path}")
        if not os.path.exists(CFG.model_path):
            raise RuntimeError(f"model not found: {CFG.model_path}")

        self.tok = Tokenizer.from_file(CFG.tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MyGPT = ModelBuilder.build()
        vocab_size = self.tok.get_vocab_size()
        self.model = MyGPT(
            vocab_size=vocab_size,
            d_model=CFG.d_model,
            n_layers=CFG.n_layers,
            n_heads=CFG.n_heads,
            max_len=CFG.max_len,
            dropout=0.0,
        ).to(self.device)

        ckpt = torch.load(CFG.model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.torch = torch
        print(f"[+] chat engine loaded (device={self.device}, vocab={vocab_size})")

    def generate(self, prompt: str, max_new: int = 200,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        self.load()
        torch = self.torch

        if "### سؤال:" not in prompt:
            formatted = f"### سؤال:\n{prompt}\n\n### جواب:\n"
        else:
            formatted = prompt

        ids = self.tok.encode(formatted).ids
        if not ids:
            return ""

        x = torch.tensor([ids[-CFG.max_len:]], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x, max_new=max_new, temperature=temperature, top_k=top_k,
        )
        full_ids = out[0].tolist()
        text = self.tok.decode(full_ids)

        if "### جواب:" in text:
            text = text.split("### جواب:", 1)[1]
        if "###" in text:
            text = text.split("###", 1)[0]
        return text.strip()


# ═══════════════════════════════════════════════════════════════════
#  [9] DocStore
# ═══════════════════════════════════════════════════════════════════

class DocStore:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.isdir(CFG.docs_dir):
            return

        for p in Path(CFG.docs_dir).iterdir():
            if p.suffix.lower() in (".txt", ".md"):
                try:
                    self.docs.append({
                        "path": str(p), "name": p.name,
                        "text": p.read_text(encoding="utf-8"),
                    })
                except Exception:
                    pass
            elif p.suffix.lower() == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(p))
                    text = "\n".join(
                        (page.extract_text() or "") for page in reader.pages
                    )
                    self.docs.append({
                        "path": str(p), "name": p.name, "text": text,
                    })
                except Exception:
                    pass

        print(f"[+] docs loaded: {len(self.docs)} files")

    def search(self, query: str, k: int = 3) -> List[str]:
        self.load()
        if not self.docs:
            return []

        q_words = set(re.findall(r"\w+", query.lower()))
        if not q_words:
            return []

        scored = []
        for d in self.docs:
            d_words = set(re.findall(r"\w+", d["text"].lower()))
            if not d_words:
                continue
            inter = q_words & d_words
            score = len(inter) / (len(q_words) + 1e-6)
            if score > 0.05:
                scored.append((score, d))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, d in scored[:k]:
            sentences = re.split(r"[.!?؟\n]+", d["text"])
            best = ""
            best_score = 0
            for s in sentences:
                s_words = set(re.findall(r"\w+", s.lower()))
                if not s_words:
                    continue
                sc = len(q_words & s_words) / (len(q_words) + 1e-6)
                if sc > best_score:
                    best_score = sc
                    best = s.strip()
            if best:
                results.append(f"[من {d['name']}]: {best}")
        return results


# ═══════════════════════════════════════════════════════════════════
#  [10] InteractionLogger
# ═══════════════════════════════════════════════════════════════════

class InteractionLogger:
    @staticmethod
    def log(prompt: str, response: str,
            meta: Optional[Dict[str, Any]] = None,
            source: str = "unknown") -> str:
        os.makedirs(CFG.work_dir, exist_ok=True)
        interaction_id = f"{int(time.time()*1000)}_{os.urandom(4).hex()}"
        entry = {
            "id": interaction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "response": response,
            "source": source,
            "meta": meta or {},
            "rating": 0,
            "comment": "",
        }
        with open(CFG.interactions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return interaction_id

    @staticmethod
    def load_all() -> List[Dict[str, Any]]:
        if not os.path.exists(CFG.interactions_path):
            return []
        out = []
        with open(CFG.interactions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    @staticmethod
    def rate(interaction_id: str, rating: int,
             comment: str = "") -> bool:
        entries = InteractionLogger.load_all()
        found = False
        for e in entries:
            if e.get("id") == interaction_id:
                e["rating"] = rating
                e["comment"] = comment
                e["rated_at"] = datetime.now(timezone.utc).isoformat()
                found = True
                break
        if not found:
            return False

        with open(CFG.interactions_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        with open(CFG.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "interaction_id": interaction_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
        return True

    @staticmethod
    def stats() -> Dict[str, Any]:
        entries = InteractionLogger.load_all()
        total = len(entries)
        rated = [e for e in entries if e.get("rating")]
        positive = [e for e in rated if e["rating"] > 0]
        negative = [e for e in rated if e["rating"] < 0]
        return {
            "total": total,
            "rated": len(rated),
            "positive": len(positive),
            "negative": len(negative),
            "unrated": total - len(rated),
            "ready_for_retrain": len(positive) >= CFG.retrain_min_positive,
        }


# ═══════════════════════════════════════════════════════════════════
#  [11] MemoryStore
# ═══════════════════════════════════════════════════════════════════

class MemoryStore:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._loaded = True
        entries = InteractionLogger.load_all()
        for e in entries:
            if e.get("rating", 0) >= 0:
                self.entries.append(e)
        try:
            with open(CFG.memory_path, "w", encoding="utf-8") as f:
                for e in self.entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception:
            pass
        print(f"[+] memory: {len(self.entries)} entries loaded")

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    def search(self, query: str, k: int = 3,
               min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        self.load()
        if not self.entries:
            return []

        min_score = min_score if min_score is not None else CFG.interaction_min_score
        q_words = self._tokenize(query)
        if not q_words:
            return []

        scored = []
        for e in self.entries[-2000:]:
            p_words = self._tokenize(e.get("prompt", ""))
            if not p_words:
                continue
            inter = q_words & p_words
            if not inter:
                continue
            score = len(inter) / (len(q_words) + 1e-6)
            if e.get("rating", 0) > 0:
                score *= 1.3
            if score >= min_score:
                scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def context_for(self, query: str, k: Optional[int] = None) -> str:
        k = k or CFG.interaction_max_in_context
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = ["### محادثات سابقة ذات صلة:"]
        for h in hits:
            lines.append(f"- سُئل سابقاً: {h.get('prompt', '')}")
            lines.append(f"  الجواب كان: {h.get('response', '')[:300]}")
        return "\n".join(lines) + "\n\n"


# ═══════════════════════════════════════════════════════════════════
#  [12] Retraining Loop
# ═══════════════════════════════════════════════════════════════════

class RetrainingLoop:
    @staticmethod
    def interactions_to_corpus(only_positive: bool = True,
                                output_path: str = "myai_data/corpus_interactions.txt") -> int:
        entries = InteractionLogger.load_all()
        if only_positive:
            entries = [e for e in entries if e.get("rating", 0) > 0]
        if not entries:
            return 0
        os.makedirs(CFG.work_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for e in entries:
                p = e.get("prompt", "").strip()
                r = e.get("response", "").strip()
                if not p or not r:
                    continue
                f.write(f"### سؤال:\n{p}\n\n### جواب:\n{r}\n\n")
        return len(entries)

    @staticmethod
    def retrain(only_positive: bool = True):
        print("[*] retraining on interactions...")
        n = RetrainingLoop.interactions_to_corpus(
            only_positive=only_positive,
            output_path="myai_data/corpus_interactions.txt",
        )
        if n == 0:
            print("[!] no interactions to train on")
            return
        print(f"[+] {n} interactions converted to corpus")
        # append to main corpus
        with open(CFG.corpus_path, "a", encoding="utf-8") as f:
            with open("myai_data/corpus_interactions.txt", "r", encoding="utf-8") as g:
                f.write("\n\n" + g.read())
        # retrain
        train_model(resume=True)


# ═══════════════════════════════════════════════════════════════════
#  [13] Web UI
# ═══════════════════════════════════════════════════════════════════

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MyAI</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Segoe UI", sans-serif;
  background: #0d1117; color: #c9d1d9;
  height: 100vh; display: flex; flex-direction: column;
}
header {
  padding: 16px 24px; background: #161b22;
  border-bottom: 1px solid #30363d;
  display: flex; justify-content: space-between; align-items: center;
}
header h1 { color: #58a6ff; font-size: 18px; }
header .status { font-size: 13px; color: #8b949e; }
#chat {
  flex: 1; overflow-y: auto; padding: 24px;
  display: flex; flex-direction: column; gap: 16px;
}
.msg {
  max-width: 780px; padding: 12px 16px; border-radius: 8px;
  line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;
  position: relative;
}
.msg.user {
  background: #1f6feb; color: #fff; align-self: flex-end;
  border-bottom-left-radius: 2px;
}
.msg.bot {
  background: #21262d; border: 1px solid #30363d; align-self: flex-start;
  border-bottom-right-radius: 2px;
}
.msg.bot .meta {
  color: #8b949e; font-size: 11px; margin-top: 8px;
  padding-top: 8px; border-top: 1px solid #30363d;
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px;
}
.rate-btns { display: flex; gap: 6px; }
.rate-btns button {
  background: transparent; border: 1px solid #30363d;
  padding: 4px 10px; border-radius: 4px;
  cursor: pointer; font-size: 14px;
  color: #c9d1d9; transition: all 0.15s;
}
.rate-btns button:hover { border-color: #58a6ff; }
.rate-btns button.active.up { background: #238636; border-color: #238636; }
.rate-btns button.active.down { background: #da3633; border-color: #da3633; }
footer {
  padding: 16px 24px; background: #161b22;
  border-top: 1px solid #30363d;
  display: flex; gap: 12px;
}
#input {
  flex: 1; padding: 12px 16px;
  background: #0d1117; border: 1px solid #30363d;
  border-radius: 6px; color: #c9d1d9;
  font-family: inherit; font-size: 15px;
  resize: none; min-height: 44px; max-height: 150px;
}
#input:focus { outline: none; border-color: #58a6ff; }
#send {
  padding: 12px 24px; background: #238636;
  border: none; border-radius: 6px; color: #fff;
  font-weight: 500; cursor: pointer; font-family: inherit;
}
#send:hover { background: #2ea043; }
#send:disabled { background: #21262d; color: #8b949e; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>🧠 MyAI</h1>
  <span class="status" id="status">loading...</span>
</header>
<div id="chat"></div>
<footer>
  <textarea id="input" placeholder="اكتب سؤالك..." rows="1"></textarea>
  <button id="send">إرسال</button>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const send = document.getElementById('send');
const status = document.getElementById('status');

function add(role, text, meta, interactionId) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  if (role === 'bot') {
    const m = document.createElement('div');
    m.className = 'meta';
    const info = document.createElement('span');
    info.textContent = meta || '';
    m.appendChild(info);
    if (interactionId) {
      const btns = document.createElement('div');
      btns.className = 'rate-btns';
      const up = document.createElement('button');
      up.textContent = '👍';
      up.onclick = () => rate(interactionId, 1, up, down);
      const down = document.createElement('button');
      down.textContent = '👎';
      down.onclick = () => rate(interactionId, -1, down, up);
      btns.appendChild(up);
      btns.appendChild(down);
      m.appendChild(btns);
    }
    d.appendChild(m);
  }
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

async function rate(id, value, activeBtn, otherBtn) {
  try {
    await fetch('/api/rate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({interaction_id: id, rating: value})
    });
    activeBtn.classList.add('active', value > 0 ? 'up' : 'down');
    otherBtn.classList.remove('active', 'up', 'down');
    status.textContent = 'rated ✓';
    setTimeout(() => status.textContent = 'ready', 1500);
  } catch (e) {
    status.textContent = 'rate failed';
  }
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  add('user', text);
  send.disabled = true;
  status.textContent = 'thinking...';
  const t0 = Date.now();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text }),
      signal: controller.signal
    });
clearTimeout(timeoutId);
    const data = await r.json();
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    add('bot', data.text || '(no reply)',
        `${dt}s | memory: ${data.memory_used || 0} | docs: ${data.docs_used || 0}`,
        data.interaction_id);
    status.textContent = 'ready';
  } catch (e) {
    add('bot', 'خطأ: ' + e.message);
    status.textContent = 'error';
  } finally {
    send.disabled = false;
    input.focus();
  }
}

send.addEventListener('click', sendMessage);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 150) + 'px';
});

fetch('/api/health').then(r => r.json()).then(d => {
  status.textContent = d.ready ? 'ready' : 'not trained';
});
input.focus();
</script>
</body>
</html>
"""


def serve_web(host: str = "0.0.0.0", port: int = 8080):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("[!] pip install fastapi uvicorn pydantic", file=sys.stderr)
        return 1

    class ChatReq(BaseModel):
        prompt: str

    class RateReq(BaseModel):
        interaction_id: str
        rating: int
        comment: str = ""

    app = FastAPI(title="MyAI")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/health")
    def health():
        ready = (
            os.path.exists(CFG.tokenizer_path)
            and os.path.exists(CFG.model_path)
        )
        return {"ready": ready, "version": "2.1"}

    @app.post("/api/chat")
    def chat(req: ChatReq):
        try:
            engine = ChatEngine.get()
            mem = MemoryStore()
            mem_ctx = mem.context_for(req.prompt, k=CFG.interaction_max_in_context)
            mem_hits = mem.search(req.prompt, k=CFG.interaction_max_in_context)
            docs = DocStore().search(req.prompt, k=3)
            doc_ctx = ""
            if docs:
                doc_ctx = "### معلومات من المستندات:\n" + "\n".join(docs) + "\n\n"
            full_prompt = mem_ctx + doc_ctx + req.prompt
            t0 = time.time()
            text = engine.generate(full_prompt, max_new=50, temperature=0.8)
            dt = time.time() - t0
            interaction_id = InteractionLogger.log(
                req.prompt, text or "",
                meta={"memory_used": len(mem_hits), "docs_used": len(docs)},
                source="web",
            )
            return {
                "text": text or "(no response)",
                "provider": f"local ({dt:.1f}s)",
                "docs_used": len(docs),
                "memory_used": len(mem_hits),
                "interaction_id": interaction_id,
            }
        except Exception as e:
            return JSONResponse(
                {"text": f"خطأ: {e}", "provider": "error"},
                status_code=500,
            )

    @app.post("/api/rate")
    def rate(req: RateReq):
        ok = InteractionLogger.rate(req.interaction_id, req.rating, req.comment)
        return {"ok": ok}

    @app.get("/api/stats")
    def stats():
        return {
            "interactions": InteractionLogger.stats(),
            "docs": len(DocStore().docs) if DocStore()._loaded else 0,
        }

    print("=" * 60)
    print(f"  MyAI web server")
    print(f"  http://{host}:{port}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


# ═══════════════════════════════════════════════════════════════════
#  [14] CLI + Auto-run
# ═══════════════════════════════════════════════════════════════════

def cmd_setup(args):
    print("=" * 60)
    print("  MyAI — full setup")
    print("=" * 60)
    print()

    if not os.path.exists(CFG.corpus_path):
        collect_data(limit=args.limit)
    else:
        print(f"[+] corpus already exists: {CFG.corpus_path}")

    if not os.path.exists(CFG.tokenizer_path):
        train_tokenizer()
    else:
        print(f"[+] tokenizer already exists: {CFG.tokenizer_path}")

    train_model()
    print()
    print("=" * 60)
    print("  DONE. Run: python3 arqam.py serve")
    print("=" * 60)


def cmd_collect(args):
    collect_data(limit=args.limit)


def cmd_tokenize(args):
    train_tokenizer()


def cmd_train(args):
    train_model()


def cmd_chat(args):
    engine = ChatEngine.get()
    mem = MemoryStore()
    print("=" * 60)
    print("  MyAI chat — اكتب 'exit' للخروج، '/stats' للإحصاءات")
    print("=" * 60)
    while True:
        try:
            user = input("\nأنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "خروج"):
            break
        if user == "/stats":
            print(json.dumps(InteractionLogger.stats(), ensure_ascii=False, indent=2))
            continue
        mem_ctx = mem.context_for(user, k=CFG.interaction_max_in_context)
        docs = DocStore().search(user, k=3)
        doc_ctx = ""
        if docs:
            doc_ctx = "### معلومات من المستندات:\n" + "\n".join(docs) + "\n\n"
        prompt = mem_ctx + doc_ctx + user
        print("AI: ", end="", flush=True)
        try:
            out = engine.generate(prompt, max_new=200, temperature=0.8)
            print(out or "(no response)")
            iid = InteractionLogger.log(user, out, source="cli")
            print(f"    [id: {iid}] rate: /up or /down")
        except Exception as e:
            print(f"[!] {e}")


def cmd_serve(args):
    serve_web(host=args.host, port=args.port)


def cmd_info(args):
    print("=" * 60)
    print(f"  MyAI v2.1 — {CFG.work_dir}")
    print("=" * 60)
    print(f"  vocab_size      : {CFG.vocab_size:,}")
    print(f"  d_model         : {CFG.d_model}")
    print(f"  n_layers        : {CFG.n_layers}")
    print(f"  n_heads         : {CFG.n_heads}")
    print(f"  max_len         : {CFG.max_len}")
    print(f"  estimated params: {CFG.model_size:,} ({CFG.model_size/1e6:.1f}M)")
    print()
    files = [
        ("corpus", CFG.corpus_path),
        ("tokenizer", CFG.tokenizer_path),
        ("model", CFG.model_path),
        ("interactions", CFG.interactions_path),
    ]
    for name, p in files:
        if os.path.exists(p):
            size = os.path.getsize(p) / 1024 / 1024
            print(f"  ✓ {name:<14} {p} ({size:.1f} MB)")
        else:
            print(f"  ✗ {name:<14} {p} (not found)")
    if os.path.isdir(CFG.docs_dir):
        n = len(list(Path(CFG.docs_dir).iterdir()))
        print(f"  docs           {n} files in {CFG.docs_dir}")
    st = InteractionLogger.stats()
    print(f"  interactions   total={st['total']} rated={st['rated']} "
          f"pos={st['positive']} neg={st['negative']}")
    print("=" * 60)


def cmd_stats(args):
    print(json.dumps(InteractionLogger.stats(), ensure_ascii=False, indent=2))


def cmd_retrain(args):
    RetrainingLoop.retrain(only_positive=True)


def cmd_auto(args):
    """التشغيل التلقائي — يبني كل ما يحتاج ثم يفتح السيرفر."""
    print("=" * 60)
    print("  MyAI v2.1 — AUTO MODE")
    print("=" * 60)
    print()

    # 1. تحقق من البيانات
    if not os.path.exists(CFG.corpus_path):
        print("[*] no corpus found. collecting...")
        try:
            collect_data(limit=AUTO_LIMIT)
        except Exception as e:
            print(f"[!] data collection failed: {e}")
            print("[*] creating dummy corpus to continue...")
            os.makedirs(CFG.work_dir, exist_ok=True)
            with open(CFG.corpus_path, "w", encoding="utf-8") as f:
                f.write("مرحباً بك في العالم. " * 1000)
    else:
        print(f"[+] corpus exists: {CFG.corpus_path}")

    # 2. تحقق من tokenizer
    if not os.path.exists(CFG.tokenizer_path):
        print("[*] training tokenizer...")
        train_tokenizer()
    else:
        print(f"[+] tokenizer exists: {CFG.tokenizer_path}")

    # 3. تحقق من الموديل
    if not os.path.exists(CFG.model_path) and AUTO_TRAIN:
        print("[*] training model (this may take a while)...")
        try:
            train_model(resume=False)
        except Exception as e:
            print(f"[!] training failed: {e}")
    elif os.path.exists(CFG.model_path):
        print(f"[+] model exists: {CFG.model_path}")
    else:
        print("[!] no model and AUTO_TRAIN=0 — starting without model")

    # 4. شغّل السيرفر
    print()
    print("[*] starting web server...")
    serve_web(host=AUTO_HOST, port=AUTO_PORT)


def main():
    ap = argparse.ArgumentParser(
        prog="arqam",
        description="MyAI — build your own AI from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "التشغيل العادي (بدون أي أمر):\n"
            "    python3 arqam.py                  # يبني كل شي + يفتح السيرفر\n"
            "\n"
            "الأوامر المتقدمة:\n"
            "    python3 arqam.py setup --limit 5000\n"
            "    python3 arqam.py collect --limit 5000\n"
            "    python3 arqam.py tokenize\n"
            "    python3 arqam.py train\n"
            "    python3 arqam.py chat\n"
            "    python3 arqam.py serve --port 8080\n"
            "    python3 arqam.py info\n"
            "    python3 arqam.py stats\n"
            "    python3 arqam.py retrain\n"
        ))
    ap.add_argument("--version", action="version", version="myai 2.1")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("setup", help="run everything: collect + tokenize + train")
    p.add_argument("--limit", type=int, default=AUTO_LIMIT)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("collect", help="collect training data")
    p.add_argument("--limit", type=int, default=AUTO_LIMIT)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("tokenize", help="train tokenizer")
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("train", help="train model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("chat", help="chat in terminal")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="web UI")
    p.add_argument("--host", default=AUTO_HOST)
    p.add_argument("--port", type=int, default=AUTO_PORT)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("info", help="show status")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("stats", help="interaction stats")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("retrain", help="retrain on positive interactions")
    p.set_defaults(func=cmd_retrain)

    p = sub.add_parser("auto", help="auto mode (default when no command)")
    p.set_defaults(func=cmd_auto)

    args = ap.parse_args()

    # ✨ المهم: إذا لم يُعطَ أي أمر، شغّل auto تلقائياً
    if not args.cmd:
        return cmd_auto(args)

    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())