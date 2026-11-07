"""Recompute every number quoted in the prose of the paper from results/*.csv. Usage: python summarize.py > ../results/final_summary.txt"""
import pandas as pd, numpy as np, ast
import os
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
R = lambda n: pd.read_csv(os.path.join(RES, f"exp_{n}.csv"))
geo = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))
def agg(d, keys):
    return d.groupby(keys + ["method"]).agg(cost=("cost", "mean"), cov=("covered", "mean"), met=("met", "mean")).reset_index()
m = R("main"); G = agg(m, ["query", "N"])
P = G.pivot_table(index=["query", "N"], columns="method", values="cost")
C = G.pivot_table(index=["query", "N"], columns="method", values="cov")
MT = G.pivot_table(index=["query", "N"], columns="method", values="met")
allmet = MT.min(axis=1) >= 0.95
big = P.index[(P.index.get_level_values(1) >= 40000) & allmet.values]
r = P.loc[big]
pub = r[["PPI++", "Strat"]].min(axis=1); fix = r[["PPI++-sub", "Strat-shrink"]]
print("RQ1 n big", len(big))
print(" vsSRS geo %.2f range %.2f-%.2f" % (geo(r.SRS / r.Ladder), (r.SRS / r.Ladder).min(), (r.SRS / r.Ladder).max()))
print(" vsPub geo %.2f max %.2f; at 1e6 geo %.2f" % (geo(pub / r.Ladder), (pub / r.Ladder).max(), geo((pub / r.Ladder)[r.index.get_level_values(1) == 1000000])))
print(" vsBetterFixed geo %.3f; vsWorseFixed geo %.3f max %.3f (%s)" % (geo(fix.min(1) / r.Ladder), geo(fix.max(1) / r.Ladder), (fix.max(1) / r.Ladder).max(), (fix.max(1) / r.Ladder).idxmax()))
print(" PPI++-sub/Ladder max %.3f  Strat-shrink/Ladder max %.3f" % ((r["PPI++-sub"] / r.Ladder).max(), (r["Strat-shrink"] / r.Ladder).max()))
M6 = ["SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shrink", "Ladder"]
print(" Ladder / cheapest max %.3f" % (r.Ladder / r[M6].min(1)).max())
metc = P.index[allmet.values]
ch = P.loc[metc][M6].idxmin(axis=1).value_counts()
print(" cheapest among %d met configs:" % len(metc), ch.to_dict())
L = m[(m.method == "Ladder") & (m.N >= 100000)]
print(" bundle share %.3f; >=2 vendors %.3f" % (L.chain.fillna("").str.contains(r"\+").mean(),
      L.chain.fillna("").apply(lambda c: len({x.split('-')[0] for x in c.replace('|', '+').split('+') if x}) >= 2).mean()))
i = m[(m["query"] == "imdb-pos") & (m.N == 1000000)]
print(" IMDB 1e6 PPI++ %.1f Ladder %.1f" % (i[i.method == "PPI++"].cost.mean(), i[i.method == "Ladder"].cost.mean()))
li = i[i.method == "Ladder"].sizes.apply(lambda s: ast.literal_eval(s)[-1] if isinstance(s, str) and s.startswith("[") else np.nan)
print(" IMDB 1e6 Ladder last level mean %.0f" % li.mean())
print(" dagger (Ladder met<0.95):", list(MT.index[MT.Ladder < 0.95]), MT.Ladder[MT.Ladder < 0.95].round(3).to_dict())
print(" Strat-shrink met at those:", MT["Strat-shrink"][MT.Ladder < 0.95].round(3).to_dict())
# coverage over 82 configs
fr = []
for n, keys in [("main", ["query", "N"]), ("llm_oracle", ["query", "N"]), ("oracle_cost", ["query", "N", "c0"]), ("eps", ["query", "N", "eps"])]:
    d = R(n); a = d.groupby(keys + ["method"]).covered.mean().reset_index(); a["exp"] = n; fr.append(a)
A = pd.concat(fr)
for mm in ["Ladder", "SRS", "PPI++", "PPI++-sub", "Strat", "Strat-shrink"]:
    v = A[A.method == mm].covered
    print(" COV %-12s n%d mean %.3f min %.3f <0.9 %d <0.8 %d" % (mm, len(v), v.mean(), v.min(), (v < 0.9).sum(), (v < 0.8).sum()))
low = A[(A.method == "Ladder") & (A.covered < 0.9)]; print(low.to_string())
print(" Ladder cov on big %.3f" % C.loc[big, "Ladder"].mean())
# RQ3
oc = R("oracle_cost").groupby(["query", "c0", "method"]).cost.mean().unstack()
print(" OC vsPPI@1", (oc.PPI__ if False else (oc["PPI++"] / oc.Ladder)).xs(1.0, level="c0").round(2).to_dict())
print(" OC vsPPI@0.005", (oc["PPI++"] / oc.Ladder).xs(0.005, level="c0").round(2).to_dict())
print(" OC vsBetterFixed range %.2f-%.2f" % ((oc[["PPI++-sub", "Strat-shrink"]].min(1) / oc.Ladder).min(), (oc[["PPI++-sub", "Strat-shrink"]].min(1) / oc.Ladder).max()))
print(" OC SRS/PPI++ <1 (PPI++ dearer than SRS):", ((oc.SRS / oc["PPI++"]) > 1).groupby(level=1).sum().to_dict() if False else (oc["PPI++"] > oc.SRS)[lambda s: s].index.tolist())
print(" OC Strat dearer than SRS:", (oc["Strat"] > oc.SRS)[lambda s: s].index.tolist())
print(" OC planned dearer than SRS:", (oc[["Ladder", "PPI++-sub", "Strat-shrink"]].max(1) > oc.SRS)[lambda s: s].index.tolist())
ep = R("eps").groupby(["query", "eps", "method"]).cost.mean().unstack()
print(" EPS vsSRS", (ep.SRS / ep.Ladder).round(2).unstack().to_string())
print(" EPS vsPPI", (ep["PPI++"] / ep.Ladder).round(2).unstack().to_string())
w = r.loc[[k for k in big if k[0] in ("hate", "offensive", "emotion-anger")]]
print(" weak vsSRS", (w.SRS / w.Ladder).round(2).to_dict())
s2 = G[G.N < 5000].pivot_table(index="query", columns="method", values="cost")
print(" smallN PPI++/Ladder fig queries", (s2["PPI++"] / s2.Ladder).loc[["imdb-pos", "yelp-avg", "agnews-biz", "hate"]].round(3).to_dict())
# RQ4
l = R("llm_oracle"); GL = agg(l, ["query", "N"])
PL = GL.pivot_table(index=["query", "N"], columns="method", values="cost"); ML = GL.pivot_table(index=["query", "N"], columns="method", values="met")
ok = PL.index[(PL.index.get_level_values(1) >= 40000) & (ML.min(1) >= 0.95).values]
x = PL.loc[ok]; pubL = x[["PPI++", "Strat"]].min(1)
print("RQ4 n met", len(ok), "vsPub geo %.2f max %.2f; vsSRS geo %.2f" % (geo(pubL / x.Ladder), (pubL / x.Ladder).max(), geo(x.SRS / x.Ladder)))
print(" Strat-shrink cheaper avg %.3f; IMDB %.3f" % (1 - geo(x["Strat-shrink"] / x.Ladder), 1 - (x["Strat-shrink"] / x.Ladder).xs("imdb-pos", level=0).iloc[0]))
print(" Ladder/betterfixed geo %.3f" % geo(x.Ladder / x[["PPI++-sub", "Strat-shrink"]].min(1)))
im = PL.xs("imdb-pos", level=0); print(" IMDB LLM PPI++ %.2f SRS %.2f" % (im["PPI++"].iloc[-1], im.SRS.iloc[-1]))
# ablation
ab = R("ablation").groupby(["query", "tag"]).cost.mean().unstack(); base = ab["abl-Ladder"]
for c in ab.columns:
    if c == "abl-Ladder": continue
    rr = ab[c] / base - 1
    print(" ABL %-20s geo %+.1f%% max %+.0f%% (%s) min %+.0f%%" % (c, 100 * (geo(ab[c] / base) - 1), 100 * rr.max(), rr.idxmax(), 100 * rr.min()))
print(ab.apply(lambda c: (100 * (c / base - 1)).round(0)).to_string())
# pilot
pl = R("pilot").groupby(["query", "pilot"]).agg(cost=("cost", "mean"), cov=("covered", "mean")).round(3); print(pl.unstack().to_string())
# RMA
rm = R("rma").groupby(["query", "tag", "method"]).agg(cost=("cost", "mean"), cov=("covered", "mean"), met=("met", "mean")).round(3); print(rm.to_string())
rr = R("rma"); rl = rr[(rr.method == "Ladder") & (rr.tag == "rma")]
print(" code share", rl.groupby("query").chain.apply(lambda s: s.fillna("").str.contains("code").mean()).round(2).to_dict())
gb = R("rma_groupby").groupby("method").agg(cost=("cost", "mean"), cov=("cov_rate", "mean"), allc=("all_covered", "mean")).round(3); print(gb)
# MultiPPI-style allocator and all seven methods (RQ1, RQ4, RMA)
M7 = M6[:5] + ["MultiPPI", "Ladder"]
print("MPPI big: MultiPPI/Ladder geo %.3f range %.2f-%.2f; cov MultiPPI %.3f Ladder %.3f" % (
      geo(r.MultiPPI / r.Ladder), (r.MultiPPI / r.Ladder).min(), (r.MultiPPI / r.Ladder).max(), C.loc[big, "MultiPPI"].mean(), C.loc[big, "Ladder"].mean()))
print(" big: Ladder/cheapest-of-7 max %.3f geo %.3f; cheapest-of-7 counts %s" % (
      (r.Ladder / r[M7].min(1)).max(), geo(r.Ladder / r[M7].min(1)), r[M7].idxmin(1).value_counts().to_dict()))
print(" big: max over other methods of X/Ladder:", {k: round((r[k] / r.Ladder).max(), 3) for k in M7 if k != "Ladder"})
print(" MultiPPI met by N:", MT.MultiPPI.groupby(level=1).mean().round(3).to_dict())
v = A[A.method == "MultiPPI"].covered
print(" COV MultiPPI n%d mean %.3f min %.3f <0.9 %d" % (len(v), v.mean(), v.min(), (v < 0.9).sum()))
print("MPPI LLM: MultiPPI/Ladder geo %.3f; Ladder/cheapest-of-7 geo %.3f max %.3f; cheapest %s" % (
      geo(x.MultiPPI / x.Ladder), geo(x.Ladder / x[M7].min(1)), (x.Ladder / x[M7].min(1)).max(), x[M7].idxmin(1).value_counts().to_dict()))
print(" LLM Ladder/better-fixed max %.3f (%s)" % ((x.Ladder / x[["PPI++-sub", "Strat-shrink"]].min(1)).max(), (x.Ladder / x[["PPI++-sub", "Strat-shrink"]].min(1)).idxmax()))
print(" LLM all-method met:", ML.min(1).round(3).to_dict())
# depth: one-level vs full Ladder
dp = R("depth").groupby(["query", "c0", "eps", "method"]).cost.mean().unstack()
print("DEPTH len1/Ladder-1 (%):"); print((100 * (dp["Ladder-len1"] / dp.Ladder - 1)).round(1).unstack(["c0", "eps"]).to_string())
# no-oracle cheapest-labeler scan and LLM oracle vs human truth (2,000-tuple labeled samples)
import simulate as S
CAT = ["glm-air-b10", "ds-flash", "ds-flash-b10", "ds-flash-think", "ds-pro", "kimi-k2.6-b10", "kimi-k3-b10"]
for q in ["imdb-pos", "yelp-1star", "yelp-avg", "agnews-biz", "emotion-surprise", "emotion-anger", "hate", "offensive", "amazon-avg", "amazon-pos"]:
    task, fy = S.QUERIES[q]
    gold, Pp, Cc = S.load_task(task, CAT + ["ds-pro-think"])
    k = int(np.argmin(Cc[:, :len(CAT)].mean(0))); t = fy(gold).mean()
    ok = Pp[:, k] >= 0; est = fy(Pp[ok, k]).mean(); ok0 = Pp[:, -1] >= 0; orc = fy(Pp[ok0, -1]).mean()
    print(" NOORACLE %-16s cheapest %-12s $%.0f/1M bias %+.1f%%; LLM-oracle vs human %+.1f%%" % (q, CAT[k], 1e6 * Cc[:, k].mean(), 100 * (est / t - 1), 100 * (orc / t - 1)))
rmM = R("rma"); rmM = rmM[rmM.tag == "rma"].groupby(["query", "method"]).cost.mean().unstack()
print("RMA cost:", rmM[["SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shrink", "MultiPPI", "Ladder"]].round(2).to_string())
