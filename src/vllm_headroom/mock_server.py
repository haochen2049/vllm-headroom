"""A small OpenAI-compatible server that imitates a continuous-batching engine in real time.

It exists for the demo and the tests: no GPU is needed. It implements ``GET /v1/models``,
``GET /metrics`` (vLLM-style gauges) and streaming ``POST /v1/completions``. Each engine step sleeps
for a time given by a :class:`~vllm_headroom.sim.StepModel` plus multiplicative noise, then emits one
token to every decoding sequence. Its numbers are simulated, not measurements of any real system.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass

from .sim import StepModel

DEFAULT_MODEL = StepModel(b0=0.020, b_dec=0.00025, b_ctx=0.00003, b_pf=0.00006, b_pfatt=0.0, overhead=0.0)


@dataclass
class _Seq:
    prompt: int
    out: int
    q: asyncio.Queue
    done_pf: int = 0
    gen: int = 0


class MockEngine:
    def __init__(self, model: StepModel = DEFAULT_MODEL, max_num_seqs: int = 64, budget: int = 4096,
                 kv_tokens: int = 200_000, block_size: int = 16, noise: float = 0.03, seed: int = 0):
        self.m, self.S, self.B, self.kv_cap = model, max_num_seqs, budget, kv_tokens
        self.block_size = block_size
        self.noise = noise
        self.rng = random.Random(seed)
        self.waiting: deque[_Seq] = deque()
        self.running: list[_Seq] = []
        self.kv = 0
        self.wake = asyncio.Event()

    def submit(self, prompt: int, out: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.waiting.append(_Seq(prompt, max(1, out), q))
        self.wake.set()
        return q

    async def loop(self) -> None:
        t_next = time.perf_counter()
        while True:
            if not self.running and not self.waiting:
                self.wake.clear()
                await self.wake.wait()
                t_next = time.perf_counter()
            dec = [s for s in self.running if s.done_pf >= s.prompt]
            budget = self.B - len(dec)
            chunks = []
            for s in self.running:
                if s.done_pf < s.prompt and budget > 0:
                    k = min(s.prompt - s.done_pf, budget)
                    chunks.append((s, k))
                    budget -= k
            while self.waiting and budget > 0 and len(self.running) < self.S:
                s = self.waiting[0]
                if self.kv + s.prompt + s.out > self.kv_cap and self.running:
                    break
                self.waiting.popleft()
                self.running.append(s)
                self.kv += s.prompt + s.out           # reserve the whole sequence: no preemption here
                k = min(s.prompt, budget)
                chunks.append((s, k))
                budget -= k
            ctx = sum(s.prompt + s.gen for s in dec)
            pf = sum(k for _, k in chunks)
            att = sum(k * (s.done_pf + k / 2) for s, k in chunks)
            dt = self.m.step_time(len(dec), ctx, pf, att) * self.rng.lognormvariate(0, self.noise)
            t_next += dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                t_next = time.perf_counter()
            for s in dec:
                s.gen += 1
                s.q.put_nowait(1)
            for s, k in chunks:
                s.done_pf += k
                if s.done_pf >= s.prompt:
                    s.gen += 1
                    s.q.put_nowait(1)
            keep = []
            for s in self.running:
                if s.done_pf >= s.prompt and s.gen >= s.out:
                    s.q.put_nowait(None)
                    self.kv -= s.prompt + s.out
                else:
                    keep.append(s)
            self.running = keep


class MockServer:
    def __init__(self, engine: MockEngine, model_name: str = "mock-model", max_model_len: int = 16384):
        self.engine = engine
        self.name = model_name
        self.max_model_len = max_model_len

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            hdr = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
            body = await reader.readexactly(int(hdr.get("content-length", "0"))) if "content-length" in hdr else b""
            if method == "GET" and path.startswith("/v1/models"):
                await self._json(writer, {"object": "list", "data": [
                    {"id": self.name, "object": "model", "max_model_len": self.max_model_len}]})
            elif method == "GET" and path.startswith("/metrics"):
                e = self.engine
                txt = (f'vllm:cache_config_info{{block_size="{e.block_size}",'
                       f'num_gpu_blocks="{e.kv_cap // e.block_size}"}} 1.0\n'
                       f"vllm:num_requests_running {len(e.running)}\n"
                       f"vllm:num_requests_waiting {len(e.waiting)}\n")
                await self._raw(writer, 200, "text/plain", txt.encode())
            elif method == "POST" and path.startswith("/v1/completions"):
                await self._complete(writer, json.loads(body or b"{}"))
            else:
                await self._raw(writer, 404, "text/plain", b"not found")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _raw(self, w, code, ctype, data: bytes) -> None:
        w.write(f"HTTP/1.1 {code} OK\r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n"
                f"Connection: close\r\n\r\n".encode() + data)
        await w.drain()

    async def _json(self, w, obj) -> None:
        await self._raw(w, 200, "application/json", json.dumps(obj).encode())

    async def _complete(self, w, req: dict) -> None:
        p = req.get("prompt", "")
        n_prompt = len(p) if isinstance(p, list) else max(1, len(str(p).split()))
        n_out = int(req.get("max_tokens", 16))
        q = self.engine.submit(n_prompt, n_out)
        w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
        await w.drain()
        k = 0
        while True:
            x = await q.get()
            if x is None:
                break
            k += 1
            chunk = {"id": "cmpl-mock", "object": "text_completion", "model": self.name,
                     "choices": [{"index": 0, "text": " tok", "finish_reason": None}]}
            w.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            await w.drain()
        usage = {"id": "cmpl-mock", "object": "text_completion", "model": self.name, "choices": [],
                 "usage": {"prompt_tokens": n_prompt, "completion_tokens": k, "total_tokens": n_prompt + k}}
        w.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")
        await w.drain()


async def start(host: str = "127.0.0.1", port: int = 0, **engine_kw):
    """Start the engine loop and server; returns (server, port, engine_task)."""
    eng = MockEngine(**engine_kw)
    srv = MockServer(eng)
    task = asyncio.create_task(eng.loop())
    server = await asyncio.start_server(srv.handle, host, port)
    port = server.sockets[0].getsockname()[1]
    return server, port, task


def serve_forever(host: str = "127.0.0.1", port: int = 8001, **engine_kw) -> None:
    async def main():
        server, p, _ = await start(host, port, **engine_kw)
        print(f"mock server on http://{host}:{p}  (simulated engine, not a real model)")
        async with server:
            await server.serve_forever()
    asyncio.run(main())
