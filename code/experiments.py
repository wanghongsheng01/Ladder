"""All experiments of the paper. Usage: python experiments.py <exp> ; results go to results/exp_<exp>.csv"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
import simulate as S

OUT = os.path.join(S.ROOT, "results")
# GLM-4.5-Flash is free but its quota (about 86 tuples/minute in our runs) makes it unusable at scale; it is
# characterised in Figure 1 but left out of the planning catalog.
ALL9 = ["glm-flash-b10", "glm-air-b10", "ds-flash", "ds-flash-b10", "ds-flash-think", "ds-pro", "kimi-k2.6-b10",
        "kimi-k3-b10", "ds-pro-think"]
CATALOG = ["glm-air-b10", "ds-flash", "ds-flash-b10", "ds-flash-think", "ds-pro", "kimi-k2.6-b10",
           "kimi-k3-b10", "ds-pro-think"]
SINGLE_Q = ["imdb-pos", "yelp-1star", "yelp-avg", "agnews-biz", "emotion-surprise", "emotion-anger", "hate",
            "offensive", "amazon-avg", "amazon-pos"]
METHODS = ["SRS", "PPI++", "PPI++-sub", "Strat", "Strat-shrink", "MultiPPI", "Ladder"]
TRIALS = int(os.environ.get("TRIALS", 200))
_cache = {}


def setting(q, labs, c0, scale, oracle="human"):
    key = (q, tuple(labs), c0, scale, oracle)
    if key not in _cache:
        _cache[key] = S.Setting(q, labs, c0=c0, scale=scale, oracle=oracle)
    return _cache[key]


def job(args):
    q, labs, c0, scale, oracle, method, eps, pilot, seeds, tag = args
    st = setting(q, labs, c0, scale, oracle)
    rows = []
    for sd in seeds:
        t = time.time()
        try:
            r = S.trial(st, method, eps=eps, pilot_n=pilot, seed=sd)
        except Exception:
            import traceback
            print("FAILED", q, method, oracle, st.N, sd, traceback.format_exc(), flush=True)
            raise
        r.update(query=q, c0=st.c0, N=st.N, oracle=oracle, eps=eps, pilot=pilot, seed=sd, tag=tag,
                 plan_ms=1000 * (time.time() - t))
        rows.append(r)
    return rows


def run(jobs, name):
    with Pool(min(8, os.cpu_count())) as p:
        rows = [r for res in p.imap_unordered(job, jobs) for r in res]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"exp_{name}.csv"), index=False)
    return df


def seeds(): return [list(range(i, i + 25)) for i in range(0, TRIALS, 25)]


def main(exp):
    J = []
    if exp == "main":  # human oracle, three population sizes
        for q in SINGLE_Q:
            for scale in [1, 50, 500]:
                for m in METHODS:
                    J += [(q, CATALOG, 0.05, scale, "human", m, 0.05, 200, s, "main") for s in seeds()]
    elif exp == "llm_oracle":  # strongest LLM as oracle
        labs = [l for l in CATALOG if l != "ds-pro-think"]
        for q in SINGLE_Q:
            for scale in [1, 50]:
                for m in METHODS:
                    J += [(q, labs, None, scale, "ds-pro-think", m, 0.05, 200, s, "llm") for s in seeds()]
    elif exp == "oracle_cost":
        for q in ["imdb-pos", "hate", "emotion-anger", "yelp-avg"]:
            for c0 in [0.005, 0.02, 0.05, 0.2, 1.0]:
                for m in METHODS:
                    J += [(q, CATALOG, c0, 500, "human", m, 0.05, 200, s, "c0") for s in seeds()]
    elif exp == "eps":
        for q in ["imdb-pos", "hate", "emotion-anger", "yelp-avg"]:
            for eps in [0.02, 0.05, 0.1]:
                for m in METHODS:
                    J += [(q, CATALOG, 0.05, 500, "human", m, eps, 200, s, "eps") for s in seeds()]
    elif exp == "ablation":
        variants = {"Ladder": "Ladder", "Ladder-nobundle": "Ladder-nobundle", "Ladder-len1": "Ladder-len1",
                    "Ladder-nostrat": "Ladder-nostrat"}
        cats = {"all": CATALOG, "DeepSeek": [l for l in CATALOG if l.startswith("ds")],
                "GLM": [l for l in CATALOG if l.startswith("glm")], "Kimi": [l for l in CATALOG if l.startswith("kimi")]}
        for q in SINGLE_Q:
            for m in variants:
                J += [(q, CATALOG, 0.05, 500, "human", m, 0.05, 200, s, "abl-" + m) for s in seeds()]
            for cn, labs in cats.items():
                J += [(q, labs, 0.05, 500, "human", "Ladder", 0.05, 200, s, "cat-" + cn) for s in seeds()]
    elif exp == "depth":  # where deeper ladders pay off: expensive oracle and a tight target
        for q in ["imdb-pos", "hate", "emotion-anger", "yelp-avg", "agnews-biz", "amazon-pos"]:
            for c0 in [0.05, 1.0]:
                for eps in [0.05, 0.02]:
                    for m in ["Ladder", "Ladder-len1"]:
                        J += [(q, CATALOG, c0, 500, "human", m, eps, 200, s, f"depth-{c0}-{eps}") for s in seeds()]
    elif exp == "pilot":
        for q in ["imdb-pos", "hate", "emotion-anger", "yelp-avg"]:
            for pl in [100, 200, 400, 800]:
                J += [(q, CATALOG, 0.05, 500, "human", "Ladder", 0.05, pl, s, "pilot") for s in seeds()]
    return run(J, exp)





def overhead(queries=SINGLE_Q, reps=20):
    """Planning time, and whether skyline pruning changes the chosen ladder compared with exhaustive search."""
    import itertools, ladder as L
    rows = []
    for q in queries:
        st = setting(q, CATALOG, 0.05, 500, "human")
        cand_l = list(range(len(st.labelers)))
        for sd in range(reps):
            rng = np.random.default_rng(sd); pilot = rng.permutation(st.N)[:200]
            Yp, Pp = st.Y[pilot], st.P[pilot]
            unit = {k: st.Cost[pilot, k].mean() for k in cand_l}
            members = [(k,) for k in cand_l] + list(itertools.combinations(cand_l, 2))
            t0 = time.time()
            cands = [(m, sum(unit[k] for k in m), L.cv_rho2(Pp, Yp, list(m), st.C)) for m in members]
            t1 = time.time(); idx, sc = L.best_chain(cands, unit, st.c0, 3); t2 = time.time()
            # exhaustive: every ordered-by-rho2 chain of up to 3 candidates, no skyline
            order = sorted(range(len(cands)), key=lambda i: -cands[i][2]); best = (np.sqrt(st.c0), [])
            for Lk in range(1, 4):
                for path in itertools.combinations(order, Lk):
                    s2 = L.chain_score([cands[i][0] for i in path], [cands[i][2] for i in path], unit, st.c0)
                    if s2 < best[0]: best = (s2, list(path))
            t3 = time.time()
            rows.append(dict(query=q, seed=sd, fit_ms=1e3 * (t1 - t0), search_ms=1e3 * (t2 - t1), exhaustive_ms=1e3 * (t3 - t2),
                             same=abs(best[0] - sc) < 1e-12, gap=sc / best[0] - 1, n_sky=len(L.skyline(cands))))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "exp_overhead.csv"), index=False); return df


def groupby_exp():
    J = []
    for q in S.GROUPBY:
        for scale in [50, 500]:
            for m in ["SRS", "PPI++", "Ladder"]:
                J += [(q, scale, m, s) for s in seeds()]
    with Pool(min(8, os.cpu_count())) as p:
        rows = [r for res in p.imap_unordered(_gjob, J) for r in res]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "exp_groupby.csv"), index=False); return df


_gcache = {}
def _gjob(args):
    q, scale, m, sds = args
    if (q, scale) not in _gcache: _gcache[(q, scale)] = S.GroupSetting(q, CATALOG, c0=0.05, scale=scale)
    st = _gcache[(q, scale)]
    out = []
    for sd in sds:
        r = S.trial_groupby(st, m, eps=0.1, seed=sd); r.update(query=q, N=st.N, seed=sd); out.append(r)
    return out


# The enterprise case study (Section V-H) runs on de-identified company data that cannot be released;
# its per-run results are in results/exp_rma*.csv.


if __name__ == "__main__":
    t = time.time()
    name = sys.argv[1]
    df = overhead() if name == "overhead" else groupby_exp() if name == "groupby" else main(name)
    print(name, len(df), "rows", "%.0fs" % (time.time() - t))
