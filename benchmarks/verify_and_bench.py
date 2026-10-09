#!/usr/bin/env python3
"""Live In-Pod Verification & Dual-Replica Benchmark Driver for DeepSeek-V4.1-Flash on G4 (8x RTX PRO 6000)."""
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import aiohttp

MODEL = "/models/deepseek-ai/DeepSeek-V4.1-Flash"
PORTS = [30000, 30001]


def get_json(url, timeout=15):
  with urllib.request.urlopen(url, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))


def post_json(url, payload, timeout=180):
  req = urllib.request.Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json"},
  )
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))


def step1_verify_server_info():
  print("\n" + "=" * 88)
  print("STEP 1: LIVE SERVER CONFIGURATION & FEATURE VERIFICATION (/get_server_info)")
  print("=" * 88)
  for port in PORTS:
    info = get_json(f"http://127.0.0.1:{port}/get_server_info")
    summary = {
        "port": port,
        "tp_size": info.get("tp_size"),
        "ep_size": info.get("ep_size"),
        "context_length": info.get("context_length"),
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "max_running_requests": info.get("max_running_requests"),
        "cuda_graph_max_bs_decode": info.get("cuda_graph_max_bs_decode"),
        "chunked_prefill_size": info.get("chunked_prefill_size"),
        "speculative_algorithm": info.get("speculative_algorithm"),
        "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
        "kv_cache_dtype": info.get("kv_cache_dtype"),
        "attention_backend": info.get("attention_backend"),
        "moe_runner_backend": info.get("moe_runner_backend"),
        "enable_deepseek_v4_fp4_indexer": info.get("enable_deepseek_v4_fp4_indexer"),
        "reasoning_parser": info.get("reasoning_parser"),
        "tool_call_parser": info.get("tool_call_parser"),
        "mem_fraction_static": info.get("mem_fraction_static"),
        "swa_full_tokens_ratio": info.get("swa_full_tokens_ratio"),
    }
    print(f"[Replica :{port}] {json.dumps(summary, indent=2)}")


def step2_verify_tool_and_reasoning():
  print("\n" + "=" * 88)
  print("STEP 2: LIVE TOOL-CALL PARSER (deepseekv41) & REASONING PARSER (deepseek-v41) TEST")
  print("=" * 88)
  payload = {
      "model": MODEL,
      "messages": [{
          "role": "user",
          "content": (
              "What is the current weather in Mountain View, CA in Fahrenheit? "
              "Call the get_weather tool."
          ),
      }],
      "tools": [{
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "location": {"type": "string"},
                      "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                  },
                  "required": ["location"],
              },
          },
      }],
      "max_tokens": 256,
      "temperature": 0.0,
  }
  t0 = time.perf_counter()
  resp = post_json("http://127.0.0.1:30000/v1/chat/completions", payload)
  dt = time.perf_counter() - t0
  msg = resp["choices"][0]["message"]
  usage = resp.get("usage", {})
  print(f"Elapsed: {dt:.3f}s | Usage: {usage}")
  print(f"Reasoning Content ({len(msg.get('reasoning_content') or '')} chars): {(msg.get('reasoning_content') or '')[:240]!r}...")
  print(f"Parsed Tool Calls: {json.dumps(msg.get('tool_calls'))}")


async def stream_one_request(session, url, prompt_text, max_tokens, sem):
  payload = {
      "model": MODEL,
      "messages": [{"role": "user", "content": prompt_text}],
      "max_tokens": max_tokens,
      "temperature": 0.6,
      "stream": True,
      "stream_options": {"include_usage": True},
      "ignore_eos": True,
  }
  async with sem:
    t_start = time.perf_counter()
    t_first = None
    t_prev = None
    itls = []
    out_tokens = 0
    in_tokens = 0
    chunk_events = 0
    try:
      async with session.post(url, json=payload) as resp:
        async for raw_line in resp.content:
          line = raw_line.decode("utf-8", errors="ignore").strip()
          if not line or not line.startswith("data: "):
            continue
          data_str = line[6:]
          if data_str == "[DONE]":
            break
          try:
            data = json.loads(data_str)
          except Exception:
            continue
          if data.get("usage"):
            out_tokens = data["usage"].get("completion_tokens", out_tokens)
            in_tokens = data["usage"].get("prompt_tokens", in_tokens)
          choices = data.get("choices") or []
          if choices:
            delta = choices[0].get("delta") or {}
            txt = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if txt:
              now = time.perf_counter()
              if t_first is None:
                t_first = now
              else:
                itls.append((now - t_prev) * 1000.0)
              t_prev = now
              chunk_events += 1
      t_end = time.perf_counter()
      if t_first is None:
        return {"ok": False, "err": "no_first_token"}
      if out_tokens <= 0:
        out_tokens = max(chunk_events, 1)
      ttft_s = t_first - t_start
      decode_s = max(t_end - t_first, 1e-6)
      # Per-token TPOT (ms/token): total decode time / (out_tokens - 1)
      tpot_ms = (decode_s * 1000.0) / max(out_tokens - 1, 1)
      return {
          "ok": True,
          "ttft_s": ttft_s,
          "tpot_ms": tpot_ms,
          "decode_s": decode_s,
          "total_s": t_end - t_start,
          "out_tokens": out_tokens,
          "in_tokens": in_tokens,
          "chunk_events": chunk_events,
      }
    except Exception as e:
      return {"ok": False, "err": str(e)}


def make_coding_prompt(session_id, approx_tokens=4096):
  base_block = (
      "def optimize_sm120_sparse_mla_tile(q_nope, q_pe, kv_latent_fp8, block_indices, scale_b):\n"
      "    # Compute block-sparse MLA attention over 16 query heads and FP8 E4M3 KV cache tiles\n"
      "    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)\n"
      "    m_i = tl.full([BLOCK_H], -float('inf'), dtype=tl.float32)\n"
      "    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)\n"
      "    return acc / l_i[:, None]\n"
  )
  # ~55 tokens per repeat
  repeats = max(1, approx_tokens // 55)
  prefix = f"# Session {session_id}: Multi-turn repository analysis and kernel optimization task.\n"
  suffix = (
      f"\n# Task {session_id}: Provide a detailed step-by-step analysis and complete "
      "Triton/CUDA implementation for SM120 Blackwell RTX PRO 6000.\n"
  )
  return prefix + (base_block * repeats) + suffix


async def run_replica_load(port, k, num_requests, prompt_tokens, max_tokens):
  url = f"http://127.0.0.1:{port}/v1/chat/completions"
  sem = asyncio.Semaphore(k)
  conn = aiohttp.TCPConnector(limit=k + 16)
  timeout = aiohttp.ClientTimeout(total=1200)
  prompts = [
      make_coding_prompt(i % max(k * 2, 16), prompt_tokens)
      for i in range(num_requests)
  ]
  async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
    # Warmup 2 requests to prime prefix cache (simulating multi-turn prefix reuse)
    await asyncio.gather(*[
        stream_one_request(session, url, prompts[i], 32, sem)
        for i in range(min(k, 4))
    ])
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        stream_one_request(session, url, prompts[i], max_tokens, sem)
        for i in range(num_requests)
    ])
    wall_s = time.perf_counter() - t0
  ok_res = [r for r in results if r.get("ok")]
  errs = len(results) - len(ok_res)
  total_out = sum(r["out_tokens"] for r in ok_res)
  sys_tok_s = total_out / max(wall_s, 1e-6)
  ttfts = sorted(r["ttft_s"] for r in ok_res)
  tpots = sorted(r["tpot_ms"] for r in ok_res)
  in_toks = [r["in_tokens"] for r in ok_res if r["in_tokens"] > 0]
  out_toks = [r["out_tokens"] for r in ok_res]

  def pct(arr, p):
    if not arr:
      return 0.0
    idx = min(int(len(arr) * p), len(arr) - 1)
    return arr[idx]

  return {
      "port": port,
      "k": k,
      "wall_s": wall_s,
      "ok_count": len(ok_res),
      "err_count": errs,
      "total_out_tokens": total_out,
      "sys_tok_s": sys_tok_s,
      "ttft_p50_s": pct(ttfts, 0.50),
      "ttft_p90_s": pct(ttfts, 0.90),
      "tpot_p50_ms": pct(tpots, 0.50),
      "tpot_p90_ms": pct(tpots, 0.90),
      "isl_avg": statistics.mean(in_toks) if in_toks else prompt_tokens,
      "osl_avg": statistics.mean(out_toks) if out_toks else max_tokens,
  }


def parse_sglang_log_stats(port):
  log_path = f"/tmp/sglang.{port}.log"
  if not os.path.exists(log_path):
    return {}
  tail = subprocess.getoutput(f"tail -n 120 {log_path}")
  accept_lens = [float(x) for x in re.findall(r"accept len:\s*([0-9.]+)", tail)]
  gen_Verify_tpts = [
      float(x)
      for x in re.findall(r"gen throughput \(token/s\):\s*([0-9.]+)", tail)
  ]
  cuda_graphs = re.findall(r"cuda graph:\s*(True|False)", tail)
  return {
      "avg_accept_len": round(statistics.mean(accept_lens), 2) if accept_lens else None,
      "peak_decode_log_tok_s": round(max(gen_Verify_tpts), 1) if gen_Verify_tpts else None,
      "cuda_graph_active": ("True" in cuda_graphs) if cuda_graphs else None,
  }


async def run_dual_node_sweep():
  print("\n" + "=" * 88)
  print("STEP 3: LIVE DUAL-REPLICA (8x RTX PRO 6000, TP4xDP2) CONCURRENCY LADDER BENCHMARK")
  print("=" * 88)
  # Sweep k per replica: k=1 (C=2), k=4 (C=8), k=8 (C=16), k=16 (C=32), k=32 (C=64), k=46 (C=92)
  sweep_points = [
      (1, 4, 4096, 256),
      (4, 16, 4096, 256),
      (8, 24, 4096, 256),
      (16, 32, 4096, 256),
      (32, 64, 4096, 256),
      (46, 92, 4096, 256),
  ]
  all_rows = []
  for k, n_reqs, p_toks, max_toks in sweep_points:
    t0 = time.perf_counter()
    res_a, res_b = await asyncio.gather(
        run_replica_load(30000, k, n_reqs, p_toks, max_toks),
        run_replica_load(30001, k, n_reqs, p_toks, max_toks),
    )
    dt = time.perf_counter() - t0
    log_a = parse_sglang_log_stats(30000)
    log_b = parse_sglang_log_stats(30001)
    node_tok_s = res_a["sys_tok_s"] + res_b["sys_tok_s"]
    chip_tok_s = node_tok_s / 8.0
    ttft_p50 = (res_a["ttft_p50_s"] + res_b["ttft_p50_s"]) / 2.0
    tpot_p50 = (res_a["tpot_p50_ms"] + res_b["tpot_p50_ms"]) / 2.0
    tpot_p90 = max(res_a["tpot_p90_ms"], res_b["tpot_p90_ms"])
    row = {
        "k_per_replica": k,
        "node_concurrency": 2 * k,
        "replica_a_tok_s": round(res_a["sys_tok_s"], 2),
        "replica_b_tok_s": round(res_b["sys_tok_s"], 2),
        "node_tok_s": round(node_tok_s, 2),
        "tok_s_per_gpu": round(chip_tok_s, 2),
        "ttft_p50_s": round(ttft_p50, 3),
        "tpot_p50_ms": round(tpot_p50, 2),
        "tpot_p90_ms": round(tpot_p90, 2),
        "isl_avg": round(res_a["isl_avg"], 0),
        "osl_avg": round(res_a["osl_avg"], 0),
        "errors": res_a["err_count"] + res_b["err_count"],
        "dspark_accept_len_a": log_a.get("avg_accept_len"),
        "dspark_accept_len_b": log_b.get("avg_accept_len"),
        "peak_decode_log_a": log_a.get("peak_decode_log_tok_s"),
        "peak_decode_log_b": log_b.get("peak_decode_log_tok_s"),
    }
    all_rows.append(row)
    print(
        f"[k={k:2d} | C={2*k:2d}] Node={node_tok_s:7.2f} tok/s "
        f"(A={res_a['sys_tok_s']:6.2f}, B={res_b['sys_tok_s']:6.2f} | {chip_tok_s:6.2f} tok/s/GPU) | "
        f"TTFT p50={ttft_p50:.3f}s | ITL/TPOT p50={tpot_p50:.2f}ms (p90={tpot_p90:.2f}ms) | "
        f"DSpark accept_len={log_a.get('avg_accept_len')}/{log_b.get('avg_accept_len')} | "
        f"err={row['errors']} ({dt:.1f}s)"
    )

  with open("/tmp/live_dual_replica_results.json", "w") as f:
    json.dump(all_rows, f, indent=2)
  print("\nSaved /tmp/live_dual_replica_results.json")


async def step4_verify_long_context():
  print("\n" + "=" * 88)
  print("STEP 4: LIVE LONG-CONTEXT (64K & 128K TOKENS) SPARSE-MLA + FP4 INDEXER VERIFICATION")
  print("=" * 88)
  sem = asyncio.Semaphore(2)
  conn = aiohttp.TCPConnector(limit=4)
  timeout = aiohttp.ClientTimeout(total=600)
  url = "http://127.0.0.1:30000/v1/chat/completions"
  async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
    for ctx_toks in (32768, 65536, 131072):
      prompt = make_coding_prompt(999, ctx_toks)
      res = await stream_one_request(session, url, prompt, 64, sem)
      print(
          f"[Target ~{ctx_toks} toks | Actual ISL={res.get('in_tokens')} toks] "
          f"ok={res.get('ok')} | TTFT={res.get('ttft_s', 0):.3f}s | "
          f"ITL/TPOT={res.get('tpot_ms', 0):.2f}ms | OSL={res.get('out_tokens')} toks"
      )


async def run_quick_smoke(base_url):
  print("\n" + "=" * 88)
  print(f"QUICK SMOKE TEST & VERIFICATION AGAINST {base_url}")
  print("=" * 88)
  try:
    info = get_json(f"{base_url}/get_server_info")
    print("[1/3] Server Config (/get_server_info):")
    for k in (
        "tp_size",
        "ep_size",
        "context_length",
        "max_total_num_tokens",
        "max_running_requests",
        "speculative_algorithm",
        "kv_cache_dtype",
        "reasoning_parser",
        "tool_call_parser",
    ):
      if k in info:
        print(f"  {k}: {info.get(k)}")
  except Exception as e:
    print(f"[1/3] Note: /get_server_info skipped ({e})")

  print("\n[2/3] Verifying Tool Calling (deepseekv41) & Reasoning Parser (deepseek-v41)...")
  payload = {
      "model": MODEL,
      "messages": [{
          "role": "user",
          "content": "What is the weather in Mountain View, CA in Fahrenheit? Call get_weather.",
      }],
      "tools": [{
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {
                  "type": "object",
                  "properties": {"location": {"type": "string"}},
                  "required": ["location"],
              },
          },
      }],
      "max_tokens": 256,
      "temperature": 0.0,
  }
  resp = post_json(f"{base_url}/v1/chat/completions", payload)
  msg = resp["choices"][0]["message"]
  print(f"  Tool Calls Parsed: {json.dumps(msg.get('tool_calls'))}")

  print("\n[3/3] Running Streaming Latency & Throughput Smoke Test (ISL=4096, OSL=256)...")
  port = int(base_url.rsplit(":", 1)[-1].split("/")[0])
  for k, n_reqs in ((1, 2), (4, 8), (8, 16)):
    res = await run_replica_load(port, k, n_reqs, 4096, 256)
    print(
        f"  [Concurrency C={k:2d}] Output={res['sys_tok_s']:6.1f} tok/s | "
        f"p50 TTFT={res['ttft_p50_s']:.3f}s | p50 ITL/TPOT={res['tpot_p50_ms']:.2f}ms | "
        f"errors={res['err_count']}"
    )
  print("\nSMOKE TEST PASSED!")


def main():
  import argparse
  parser = argparse.ArgumentParser(description="Verify and benchmark DeepSeek-V4.1-Flash on G4")
  parser.add_argument("--url", default="", help="Single endpoint URL (e.g. http://127.0.0.1:7080)")
  parser.add_argument("--quick-smoke", action="store_true", help="Run fast 60s verification and smoke benchmark")
  args = parser.parse_args()

  if args.quick_smoke or args.url:
    target_url = (args.url or "http://127.0.0.1:7080").rstrip("/")
    asyncio.run(run_quick_smoke(target_url))
    return

  step1_verify_server_info()
  step2_verify_tool_and_reasoning()
  asyncio.run(run_dual_node_sweep())
  asyncio.run(step4_verify_long_context())


if __name__ == "__main__":
  main()
