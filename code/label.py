"""Label every tuple of every task with every physical labeler (model x prompting mode) and record per-tuple cost.

Usage: python label.py <labeler> [task ...]
Output: results/labels/<task>__<labeler>.csv with columns tid, pred (label index or -1), cost_usd, in_tok, out_tok.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import llm  # keys from LLM_KEYS_JSON
from tasks import TASKS, load, ROOT

# USD per 1M tokens (input, output). DeepSeek: official page, peak rate (2026-11-03).
# GLM and Kimi: third-party aggregators quoting the vendors' price pages (2026-10); glm-4.5-flash is free.
PRICES = {"glm-flash": (0.0, 0.0), "glm-air": (0.20, 1.10), "ds-flash": (0.30, 1.20), "ds-pro": (1.32, 3.96),
          "kimi-k2.6": (0.95, 4.00), "kimi-k3": (3.00, 15.00)}
llm.MODELS.update({"glm-flash": ("glm", "glm-4.5-flash"), "glm-air": ("glm2", "glm-4.5-air"),
                   "ds-flash": ("deepseek", "deepseek-flash"), "ds-pro": ("deepseek", "deepseek-v4-pro"),
                   "kimi-k2.6": ("kimi", "kimi-k2.6"), "kimi-k3": ("kimi", "kimi-k3")})
off = {"thinking": {"type": "disabled"}}
llm.FIXED.update({"glm-flash": off, "glm-air": off, "kimi-k2.6": {**off, "temperature": 0.6},
                  "kimi-k3": {**off, "temperature": 0.6}, "ds-flash": off, "ds-pro": off,
                  "ds-flash-think": {"reasoning_effort": "low"}, "ds-pro-think": {"reasoning_effort": "high"}})
llm.MODELS.update({"ds-flash-think": ("deepseek", "deepseek-flash"), "ds-pro-think": ("deepseek", "deepseek-v4-pro")})
PRICES.update({"ds-flash-think": PRICES["ds-flash"], "ds-pro-think": PRICES["ds-pro"]})
# physical labeler -> (model, batch size)
LABELERS = {"glm-flash": ("glm-flash", 1), "glm-air": ("glm-air", 1), "ds-flash": ("ds-flash", 1),
            "ds-flash-b10": ("ds-flash", 10), "ds-flash-think": ("ds-flash-think", 1), "kimi-k2.6": ("kimi-k2.6", 1),
            "ds-pro": ("ds-pro", 1), "kimi-k3": ("kimi-k3", 1), "ds-pro-think": ("ds-pro-think", 1),
            # GLM and Kimi accounts are rate-limited to a few requests per second, so they run batched
            "glm-flash-b10": ("glm-flash", 10), "glm-air-b10": ("glm-air", 10), "kimi-k2.6-b10": ("kimi-k2.6", 10),
            "kimi-k3-b10": ("kimi-k3", 10)}
WORKERS = {"glm-flash": 2, "glm-air": 3, "kimi-k2.6": 3, "kimi-k3": 2}
OUT = os.path.join(ROOT, "results", "labels")
os.makedirs(OUT, exist_ok=True)


def parse(text, labels):
    t = text.strip().lower().strip(" .\"'*`")
    for i, l in enumerate(labels):
        if t == l: return i
    hits = []
    for i, l in enumerate(labels):
        m = re.search(r"(^|\W)" + re.escape(l) + r"($|\W)", t)
        if m: hits.append((m.start(), -len(l), i))
    return min(hits)[2] if hits else -1


def prompt_single(spec, text):
    opts = ", ".join(spec["labels"])
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nAnswer with exactly one of: {opts}.\n\nText:\n\"\"\"{text}\"\"\"\n\nAnswer:"}]


def prompt_batch(spec, texts):
    opts = ", ".join(spec["labels"])
    body = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nFor each of the {len(texts)} texts below, answer with exactly one of: {opts}.\n"
                                        f"Return one line per text in the form `[k] answer`, nothing else.\n\n{body}"}]


def cost(model, usage):
    pi, po = PRICES[model]
    return (usage.get("prompt_tokens", 0) * pi + usage.get("completion_tokens", 0) * po) / 1e6


def run(labeler, task):
    model, B = LABELERS[labeler]
    path = os.path.join(OUT, f"{task}__{labeler}.csv")
    spec, df = TASKS[task], load(task)
    groups = [list(range(i, min(i + B, len(df)))) for i in range(0, len(df), B)]
    old = None
    if os.path.exists(path):  # repair mode: only redo tuples whose call failed
        old = pd.read_csv(path)
        bad = set(old.tid[old.cost_usd.isna()])
        if not bad: return path
        groups = [g for g in groups if bad & set(g)]
        print(f"repair {task} {labeler}: {len(groups)} calls", flush=True)

    def work(g):
        try:
            if B == 1:
                r = llm.chat(model, prompt_single(spec, df.text[g[0]]), max_tokens=2000)
                preds = [parse(r["text"], spec["labels"])]
            else:
                r = llm.chat(model, prompt_batch(spec, [df.text[i] for i in g]), max_tokens=4000)
                found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
                preds = [parse(found.get(j + 1, ""), spec["labels"]) for j in range(len(g))]
            u = r["usage"]; c = cost(model, u) / len(g)
            return [(i, p, c, u.get("prompt_tokens", 0) / len(g), u.get("completion_tokens", 0) / len(g)) for i, p in zip(g, preds)]
        except Exception as e:
            print("ERR", labeler, task, g[0], str(e)[:120], flush=True)
            return [(i, -1, np.nan, np.nan, np.nan) for i in g]

    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", WORKERS.get(model, 12)))) as ex:
        rows = [x for res in ex.map(work, groups) for x in res]
    out = pd.DataFrame(rows, columns=["tid", "pred", "cost_usd", "in_tok", "out_tok"])
    if old is not None:
        out = pd.concat([old[~old.tid.isin(out.tid)], out])
    out = out.sort_values("tid").reset_index(drop=True)
    out.to_csv(path, index=False)
    acc = (out.pred.values == df.gold.values).mean()
    print(f"{task:10s} {labeler:13s} n={len(out)} acc={acc:.3f} unparsed={(out.pred < 0).mean():.3f} "
          f"cost=${out.cost_usd.sum():.3f} time={time.time() - t0:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    lab = sys.argv[1]
    for task in (sys.argv[2:] or list(TASKS)):
        run(lab, task)
