"""Request-size workloads: an empirical table of (prompt_tokens, output_tokens) pairs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Workload:
    prompt: np.ndarray
    output: np.ndarray
    source: str = "synthetic"

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(len(self.prompt), size=n)
        return self.prompt[idx], self.output[idx]

    def summary(self) -> dict:
        return {
            "source": self.source,
            "n": int(len(self.prompt)),
            "prompt_mean": float(np.mean(self.prompt)),
            "prompt_p50": float(np.median(self.prompt)),
            "prompt_p95": float(np.percentile(self.prompt, 95)),
            "output_mean": float(np.mean(self.output)),
            "output_p50": float(np.median(self.output)),
            "output_p95": float(np.percentile(self.output, 95)),
        }

    def to_dict(self) -> dict:
        return {"source": self.source, "prompt": self.prompt.tolist(), "output": self.output.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Workload":
        return cls(np.asarray(d["prompt"], np.int64), np.asarray(d["output"], np.int64), d.get("source", "file"))


def synthetic(prompt_mean: float = 1000, output_mean: float = 250, cv: float = 1.0, n: int = 4000,
              max_len: int | None = None, seed: int = 0) -> Workload:
    """Log-normal prompt and output lengths with the given means and coefficient of variation."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(np.log1p(cv ** 2))

    def draw(mean: float) -> np.ndarray:
        x = rng.lognormal(np.log(mean) - s ** 2 / 2, s, size=n)
        return np.maximum(1, np.round(x)).astype(np.int64)

    p, o = draw(prompt_mean), draw(output_mean)
    if max_len:
        p = np.minimum(p, max_len - 16)
        o = np.minimum(o, np.maximum(1, max_len - p))
    return Workload(p, o, f"synthetic lognormal (prompt mean {prompt_mean:g}, output mean {output_mean:g}, cv {cv:g})")


def load_jsonl(path: str | Path) -> Workload:
    """Read a JSONL file with one request per line: ``{"prompt_tokens": int, "output_tokens": int}``.

    The OpenAI ``usage`` field names are also accepted.
    """
    p, o = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "usage" in d and isinstance(d["usage"], dict):
                d = d["usage"]
            pt = d.get("prompt_tokens", d.get("input_tokens"))
            ot = d.get("output_tokens", d.get("completion_tokens"))
            if pt is None or ot is None:
                continue
            p.append(int(pt))
            o.append(max(1, int(ot)))
    if not p:
        raise ValueError(f"{path}: no rows with prompt_tokens and output_tokens")
    return Workload(np.asarray(p, np.int64), np.asarray(o, np.int64), f"file {Path(path).name}")
