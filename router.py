"""Mini SLM router: small model by default, large model only when needed.

Runs every task twice:
  1. baseline - everything goes to the large model
  2. routed   - a small model classifies each request, then easy work stays small

and compares latency, tokens, estimated cost and accuracy.

    python router.py           # real run with Ollama
    python router.py --mock    # rehearsal without models (simulated numbers)
"""
import argparse
import json
import os
import random
import re
import time

from rich.console import Console
from rich.table import Table

import config

console = Console()

TOOLS = [
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Create a calendar meeting",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "date": {"type": "string"},
            "time": {"type": "string"}, "attendee": {"type": "string"}},
            "required": ["title", "time", "attendee"]}}},
    {"type": "function", "function": {
        "name": "create_reminder",
        "description": "Create a reminder for the user",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "date": {"type": "string"}, "time": {"type": "string"}},
            "required": ["text", "time"]}}},
]

ROUTER_PROMPT = (
    "Classify the request into exactly one word:\n"
    "intent  = what does the customer want\n"
    "extract = pull fields out of a text\n"
    "tool    = do an action (meeting, reminder)\n"
    "plan    = open question that needs reasoning\n"
    "Request: {q}\nAnswer:"
)

TASK_PROMPTS = {
    "intent": 'Classify the customer message as one of: refund, complaint, booking, question. '
              'Reply as JSON {{"intent": "..."}}.\nMessage: {q}',
    "extract": 'Extract name, date (YYYY-MM-DD) and amount (digits only) as JSON '
               '{{"name": "", "date": "", "amount": ""}}. Keep names in their original language.\nText: {q}',
    "tool": "{q}",
    "plan": "{q}\nAnswer in at most 5 short lines.",
}


# ---------------------------------------------------------------- model calls
def call_model(model, prompt, kind=None, mock=False, task=None):
    """Return dict with text, tool_calls, tokens in/out and latency in seconds."""
    if mock:
        return mock_call(model, prompt, kind, task)
    import ollama
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "options": {"temperature": 0, "num_predict": 200}}
    if kind in ("intent", "extract"):
        kwargs["format"] = "json"
    if kind == "tool":
        kwargs["tools"] = TOOLS
    start = time.perf_counter()
    r = ollama.chat(**kwargs)
    latency = time.perf_counter() - start
    calls = [{"name": c.function.name, "args": dict(c.function.arguments)}
             for c in (r.message.tool_calls or [])]
    return {"text": r.message.content or "", "tool_calls": calls,
            "tok_in": r.prompt_eval_count or 0, "tok_out": r.eval_count or 0,
            "latency": latency}


def tier(model):
    return "small" if model == config.SMALL_MODEL else "large"


def cost(tier_name, tok_in, tok_out):
    p = config.PRICE_PER_M[tier_name]
    return (tok_in * p["in"] + tok_out * p["out"]) / 1_000_000


# ---------------------------------------------------------------- routing
def route(task, mock=False):
    """Ask the small model what kind of request this is, then pick a model."""
    r = call_model(config.SMALL_MODEL, ROUTER_PROMPT.format(q=task["prompt"]), "route", mock, task)
    words = re.findall(r"[a-z]+", r["text"].lower())
    kind = next((w for w in words if w in TASK_PROMPTS), "plan")  # unsure -> safe side
    model = config.SMALL_MODEL if kind in config.SMALL_KINDS else config.LARGE_MODEL
    return kind, model, r


# ---------------------------------------------------------------- scoring
def norm(s):
    return re.sub(r"[^\w]", "", str(s)).lower()


def score(task, r):
    """True / False for checkable tasks, None for open-ended ones."""
    exp, kind = task["expected"], task["kind"]
    if kind == "plan":
        return None
    if kind == "tool":
        if not r["tool_calls"]:
            return False
        c = r["tool_calls"][0]
        return c["name"] == exp["tool"] and all(k in c["args"] for k in exp["args"])
    try:
        out = json.loads(r["text"])
    except json.JSONDecodeError:
        return False
    if kind == "intent":
        return norm(out.get("intent")) == exp
    return all(norm(v) in norm(out.get(k, "")) for k, v in exp.items())


# ---------------------------------------------------------------- one pass
def run(tasks, mode, mock, log):
    rows = []
    for t in tasks:
        router_r = None
        if mode == "routed":
            kind, model, router_r = route(t, mock)
        else:
            kind, model = t["kind"], config.LARGE_MODEL
        r = call_model(model, TASK_PROMPTS[t["kind"]].format(q=t["prompt"]), t["kind"], mock, t)
        ok = score(t, r)
        c = cost(tier(model), r["tok_in"], r["tok_out"])
        lat = r["latency"]
        if router_r:  # the router's own call is part of the price of routing
            c += cost("small", router_r["tok_in"], router_r["tok_out"])
            lat += router_r["latency"]
        row = {"mode": mode, "id": t["id"], "lang": t["lang"], "kind": t["kind"],
               "routed_as": kind, "model": model, "ok": ok, "latency_s": round(lat, 3),
               "tok_in": r["tok_in"], "tok_out": r["tok_out"], "cost_usd": c,
               "prompt": t["prompt"], "output": r["tool_calls"] or r["text"][:300]}
        log.write(json.dumps(row, ensure_ascii=False) + "\n")  # S1: log every call
        rows.append(row)
    return rows


def summarise(rows):
    scored = [r for r in rows if r["ok"] is not None]
    n = len(rows)
    return {
        "latency": sum(r["latency_s"] for r in rows),
        "cost_1k": sum(r["cost_usd"] for r in rows) / n * 1000,
        "accuracy": 100 * sum(r["ok"] for r in scored) / len(scored),
        "small_share": 100 * sum(tier(r["model"]) == "small" for r in rows) / n,
    }


# ---------------------------------------------------------------- output
def show(rows_b, rows_r, s_b, s_r, mock):
    t = Table(title="Routed run, call by call" + ("  [MOCK]" if mock else ""))
    for col in ("#", "lang", "routed as", "model", "ok", "latency s", "cost $/1k"):
        t.add_column(col)
    for r in rows_r:
        mark = {True: "[green]yes[/]", False: "[red]no[/]", None: "-"}[r["ok"]]
        t.add_row(str(r["id"]), r["lang"], r["routed_as"], r["model"], mark,
                  f'{r["latency_s"]:.2f}', f'{r["cost_usd"] * 1000:.3f}')
    console.print(t)

    s = Table(title="All-large baseline vs routed")
    s.add_column("metric"); s.add_column("all large"); s.add_column("routed"); s.add_column("change")
    def pct(a, b): return f"{(b - a) / a * 100:+.0f}%" if a else "-"
    s.add_row("total latency (s)", f'{s_b["latency"]:.1f}', f'{s_r["latency"]:.1f}', pct(s_b["latency"], s_r["latency"]))
    s.add_row("est. cost per 1k requests ($)", f'{s_b["cost_1k"]:.3f}', f'{s_r["cost_1k"]:.3f}', pct(s_b["cost_1k"], s_r["cost_1k"]))
    s.add_row("accuracy on checkable tasks (%)", f'{s_b["accuracy"]:.0f}', f'{s_r["accuracy"]:.0f}', f'{s_r["accuracy"] - s_b["accuracy"]:+.0f} pts')
    s.add_row("calls served by small model (%)", f'{s_b["small_share"]:.0f}', f'{s_r["small_share"]:.0f}', "")
    console.print(s)


def chart(s_b, s_r, mock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("Total latency (s)", "latency"), ("Est. cost per 1k requests ($)", "cost_1k"),
               ("Accuracy (%)", "accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (label, key) in zip(axes, metrics):
        vals = [s_b[key], s_r[key]]
        bars = ax.bar(["All large", "Routed"], vals, color=["#B5B5B0", "#2F63E8"])
        ax.set_title(label, fontsize=12)
        ax.bar_label(bars, fmt="%.2f" if key == "cost_1k" else "%.0f")
        ax.spines[["top", "right"]].set_visible(False)
    title = f"{config.LARGE_MODEL} only vs routed with {config.SMALL_MODEL}"
    fig.suptitle(title + ("  (mock data)" if mock else ""), fontsize=13)
    fig.tight_layout()
    fig.savefig(config.CHART_FILE, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="simulate models (rehearsal, no Ollama)")
    ap.add_argument("--tasks", default="tasks.json")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks, encoding="utf-8"))
    os.makedirs("results", exist_ok=True)
    with open(config.LOG_FILE, "w", encoding="utf-8") as log:
        console.print(f"[bold]Baseline:[/] every task -> {config.LARGE_MODEL}")
        rows_b = run(tasks, "baseline", args.mock, log)
        console.print(f"[bold]Routed:[/] {config.SMALL_MODEL} decides, easy work stays small")
        rows_r = run(tasks, "routed", args.mock, log)
    s_b, s_r = summarise(rows_b), summarise(rows_r)
    show(rows_b, rows_r, s_b, s_r, args.mock)
    chart(s_b, s_r, args.mock)
    console.print(f"Logged every call to {config.LOG_FILE}; chart saved to {config.CHART_FILE}")


# ---------------------------------------------------------------- mock backend
def mock_call(model, prompt, kind, task):
    """Fake but plausible responses so the demo can be rehearsed offline."""
    rnd = random.Random(f"{model}|{prompt}")
    small = model == config.SMALL_MODEL
    tok_in = len(prompt) // 3 + 40
    if kind == "route":
        return {"text": task["kind"], "tool_calls": [], "tok_in": tok_in, "tok_out": 2,
                "latency": rnd.uniform(0.08, 0.15)}
    exp = task["expected"]
    tok_out = {"intent": 8, "extract": 30, "tool": 35, "plan": 160}[kind]
    per_tok = 0.006 if small else 0.03
    latency = rnd.uniform(0.05, 0.1) + tok_out * per_tok + (0.1 if small else 0.4)
    text, calls = "", []
    if kind == "intent":
        text = json.dumps({"intent": exp})
    elif kind == "extract":
        out = dict(exp)
        if small and task["lang"] == "ar":
            out["name"] = ""  # small models slip on Arabic names now and then
        text = json.dumps(out, ensure_ascii=False)
    elif kind == "tool":
        calls = [{"name": exp["tool"], "args": {k: "x" for k in exp["args"]}}]
    else:
        text = "1. ...\n2. ...\n3. ..."
    return {"text": text, "tool_calls": calls, "tok_in": tok_in, "tok_out": tok_out,
            "latency": latency}


if __name__ == "__main__":
    main()
