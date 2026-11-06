"""Figures and tables for the paper, from results/*.csv and results/labels. Usage: python make_figs.py [what ...]"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulate as S
import ladder as L
from tasks import TASKS

ROOT = S.ROOT
FIG = os.path.join(ROOT, "05_paper", "figs"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(ROOT, "05_paper", "tables"); os.makedirs(TAB, exist_ok=True)
RES = os.path.join(ROOT, "results")
# reference palette (dataviz skill): first three categorical slots + neutral ink for the reference baseline
COL = {"Ladder": "#2a78d6", "PPI++": "#eb6834", "Strat": "#1baf7a", "SRS": "#52514e"}
VCOL = {"DeepSeek": "#2a78d6", "GLM": "#eb6834", "Kimi": "#1baf7a"}
MARK = {"Ladder": "o", "PPI++": "s", "Strat": "^", "SRS": "x", "PPI++-sub": "s", "Strat-shrink": "^"}
COL.update({"PPI++-sub": "#eb6834", "Strat-shrink": "#1baf7a"})
LS = {"SRS": "--", "PPI++-sub": ":", "Strat-shrink": ":"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42})
NICE = {"glm-flash-b10": "GLM-Flash/b10", "glm-air-b10": "GLM-Air/b10", "ds-flash": "DS-Flash",
        "ds-flash-b10": "DS-Flash/b10", "ds-flash-think": "DS-Flash+R", "ds-pro": "DS-Pro", "ds-pro-think": "DS-Pro+R",
        "kimi-k2.6-b10": "Kimi-K2.6/b10", "kimi-k3-b10": "Kimi-K3/b10"}
GBNICE = {"agnews-topic": "AG topic", "emotion-all": "Emotion", "yelp-stars": "Yelp stars", "amazon-stars": "Amazon stars"}
QNICE = {"imdb-pos": "IMDB pos", "yelp-1star": "Yelp 1-star", "yelp-avg": "Yelp AVG", "agnews-biz": "AG business",
         "emotion-surprise": "Emo surprise", "emotion-anger": "Emo anger", "hate": "Hate", "offensive": "Offensive",
         "amazon-avg": "Amazon AVG", "amazon-pos": "Amazon pos"}


def catalog_stats(labs):
    """Per task and labeler: accuracy, $ per 1k tuples, cross-validated rho^2 for the task's main query."""
    rows = []
    qmain = {"imdb": "imdb-pos", "yelp": "yelp-avg", "agnews": "agnews-biz", "emotion": "emotion-anger",
             "hate": "hate", "offensive": "offensive", "amazon": "amazon-avg"}
    for task, q in qmain.items():
        gold, P, C = S.load_task(task, labs)
        y = S.QUERIES[q][1](gold)
        for k, l in enumerate(labs):
            rows.append(dict(task=task, labeler=l, vendor=S.VENDOR[l], acc=(P[:, k] == gold).mean(),
                             cost_1k=1000 * C[:, k].mean(), rho2=L.cv_rho2(P, y, [k], len(TASKS[task]["labels"])),
                             unparsed=(P[:, k] < 0).mean()))
        # best pair
        import itertools
        best = max(itertools.combinations(range(len(labs)), 2), key=lambda p: L.cv_rho2(P, y, list(p), len(TASKS[task]["labels"])))
        rows.append(dict(task=task, labeler="+".join(labs[i] for i in best), vendor="pair",
                         acc=np.nan, cost_1k=1000 * C[:, list(best)].sum(1).mean(),
                         rho2=L.cv_rho2(P, y, list(best), len(TASKS[task]["labels"])), unparsed=np.nan))
    return pd.DataFrame(rows)


def err_corr(task, labs):
    gold, P, C = S.load_task(task, labs)
    E = (P != gold[:, None]).astype(float)
    return pd.DataFrame(np.corrcoef(E.T), index=labs, columns=labs)


def fig_motivation(labs):
    st = catalog_stats(labs); st.to_csv(os.path.join(RES, "catalog_stats.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for task, mk in zip(["imdb", "yelp", "hate"], ["o", "s", "^"]):
        d = st[(st.task == task) & (st.vendor != "pair")]
        for v, g in d.groupby("vendor"):
            ax.scatter(g.cost_1k, g.rho2, c=VCOL[v], marker=mk, s=18, edgecolors="white", linewidths=0.6, zorder=3)
        p = st[(st.task == task) & (st.vendor == "pair")].iloc[0]
        ax.scatter(p.cost_1k, p.rho2, facecolors="none", edgecolors="black", marker=mk, s=30, linewidths=0.8, zorder=4)
    ax.set_xscale("log"); ax.set_xlabel("cost per 1k tuples (USD, log scale)"); ax.set_ylabel(r"cross-validated $\rho^2$ with oracle")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=VCOL[v], marker="o", ls="", label=v) for v in VCOL]
    h += [Line2D([], [], color="gray", marker=m, ls="", label=t) for t, m in zip(["IMDB", "Yelp AVG", "Hate"], ["o", "s", "^"])]
    h += [Line2D([], [], markerfacecolor="none", markeredgecolor="black", marker="o", ls="", label="best pair")]
    ax.legend(handles=h, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)
    ax.set_title("(a) cost vs. quality (free GLM-Flash not shown)", fontsize=7.5, pad=26)
    ax = axs[1]
    E = err_corr("hate", labs)
    im = ax.imshow(E.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels([NICE[l] for l in labs], rotation=90, ha="center", fontsize=5.5)
    ax.set_yticklabels([NICE[l] for l in labs], fontsize=5.5)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{E.values[i, j]:.2f}".replace("0.", ".") if i != j else "", ha="center", va="center", fontsize=5.5,
                    color="white" if E.values[i, j] > 0.6 else "black")
    ax.set_title("(b) error correlation on Hate", fontsize=7.5)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "motivation.pdf")); plt.savefig(os.path.join(FIG, "motivation.png"), dpi=200)
    return st


def rd(name):
    return pd.read_csv(os.path.join(RES, f"exp_{name}.csv"))


def agg(df, keys):
    g = df.groupby(keys + ["method"]).agg(cost=("cost", "mean"), cov=("covered", "mean"), rel_hw=("rel_hw", "mean"),
                                          rel_err=("rel_err", "mean"), n=("cost", "size")).reset_index()
    return g


def tab_main(df):
    g = agg(df, ["query", "N"])
    piv = g.pivot_table(index=["query", "N"], columns="method", values="cost")
    cov = g.pivot_table(index=["query", "N"], columns="method", values="cov")
    met = df[df.method == "Ladder"].groupby(["query", "N"]).met.mean()
    M = ["SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shrink", "Ladder"]
    lines = []
    for N in sorted(df.N.unique()):
        if N < 50000: continue
        lines.append(r"\midrule\multicolumn{10}{l}{\emph{$N=%s$ tuples}}\\" % (f"{N:,}".replace(",", "{,}")))
        for q in S.QUERIES:
            if (q, N) not in piv.index: continue
            r = piv.loc[(q, N)]; mn = r[M].min()
            dag = met.loc[(q, N)] < 0.95
            cells = [(r"\textbf{%s}" if abs(r[m] - mn) < 1e-9 and not dag else "%s") % (f"{r[m]:.1f}" if r[m] < 100 else f"{r[m]:.0f}") for m in M]
            fixed = r[["PPI++-sub", "Strat-shrink"]]
            name = QNICE[q] + (r"$^\dagger$" if met.loc[(q, N)] < 0.95 else "")
            lines.append(f"{name} & " + " & ".join(cells) +
                         f" & {r['SRS'] / r['Ladder']:.2f}$\\times$ & {fixed.min() / r['Ladder']:.2f}$\\times$ & {fixed.max() / r['Ladder']:.2f}$\\times$ & {cov.loc[(q, N), 'Ladder']:.2f} \\\\")
    tex = r"""\begin{table*}[t]
\caption{Mean total cost (USD) to answer each query within $\pm5\%$ at 95\% confidence (every method tops up until its reported half-width meets the target), human oracle at \$0.05 per label, 200 runs per cell. PPI++-sub and Strat-shrink are the two single-family planners (\textsc{Ladder} restricted to one surrogate level without bundles, or to stratified plans without bundles). Last columns: \textsc{Ladder}'s saving over SRS, over the better and over the worse single-family planner, and its empirical coverage. $^\dagger$: stratified plans (in \textsc{Ladder} and Strat-shrink) did not reach the target within eight top-up rounds in more than 5\% of runs; excluded from the summaries in the text and not marked in bold. Bold marks the cheapest method.}\label{tab:main}
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
Query & SRS & PPI++ & Strat & PPI++-sub & Strat-shrink & \textsc{Ladder} & vs SRS & vs better & vs worse & Cov.\\
""" + "\n".join(lines).replace(r"\multicolumn{10}", r"\multicolumn{11}") + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(TAB, "main.tex"), "w").write(tex)
    return g


def fig_scale(df):
    g = agg(df, ["query", "N"])
    qs = ["imdb-pos", "yelp-avg", "agnews-biz", "hate"]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=False)
    for ax, q in zip(axs, qs):
        d = g[g["query"] == q]
        for m in ["SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shrink", "Ladder"]:
            x = d[d.method == m].sort_values("N")
            ax.plot(x.N, x.cost, color=COL[m], marker=MARK[m], ls=LS.get(m, "-"), label=m, lw=1.0 if m in LS else 1.4)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(QNICE[q], fontsize=7.5); ax.set_xlabel("table size $N$")
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5))); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].set_ylabel("mean cost (USD)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.87)); plt.savefig(os.path.join(FIG, "scale.pdf")); plt.savefig(os.path.join(FIG, "scale.pdf").replace(".pdf", ".png"), dpi=170)


def fig_factor(df, col, xlabel, fname, logx=True):
    g = df.groupby(["query", col, "method"]).cost.mean().reset_index()
    qs = sorted(g["query"].unique(), key=list(S.QUERIES).index)
    fig, axs = plt.subplots(1, len(qs), figsize=(7.0, 1.95))
    for ax, q in zip(axs, qs):
        d = g[g["query"] == q]
        srs = d[d.method == "SRS"].set_index(col).cost
        for m in ["PPI++", "Strat", "PPI++-sub", "Strat-shrink", "Ladder"]:
            x = d[d.method == m].set_index(col).cost
            ax.plot(x.index, srs.loc[x.index] / x, color=COL[m], marker=MARK[m], ls=LS.get(m, "-"), label=m, lw=1.0 if m in LS else 1.4)
        ax.axhline(1, color=COL["SRS"], ls="--", lw=0.8)
        if logx: ax.set_xscale("log")
        ax.set_title(QNICE[q], fontsize=7.5); ax.set_xlabel(xlabel)
    axs[0].set_ylabel("saving over SRS (x)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.87)); plt.savefig(os.path.join(FIG, fname)); plt.savefig(os.path.join(FIG, fname).replace(".pdf", ".png"), dpi=170)


def fig_coverage(frames):
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    allg = pd.concat([agg(d, ["query", "N", "tag", "c0", "eps"]) for d in frames])
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    for i, m in enumerate(["SRS", "PPI++", "Strat", "Ladder"]):
        v = allg[allg.method == m]["cov"].values
        ax.scatter(np.random.default_rng(0).normal(i, 0.07, len(v)), v, s=6, color=COL[m], alpha=0.6, edgecolors="none")
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color="black", lw=1)
    ax.axhline(0.95, color="gray", ls=":", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["SRS", "PPI++", "Strat", "Ladder"]); ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.8, 1.005)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "coverage.pdf")); plt.savefig(os.path.join(FIG, "coverage.pdf").replace(".pdf", ".png"), dpi=170)
    return allg


def simple_table(rows, header, caption, label, fname, colspec, wide=False, sep="3pt"):
    tex = ("\\begin{table}[t]\n\\caption{%s}\\label{%s}\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{%s}\n\\toprule\n%s\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n") % (
        caption, label, colspec, " & ".join(header), "\n".join(" & ".join(r) + " \\\\" for r in rows))
    tex = tex.replace("\\setlength{\\tabcolsep}{3pt}", "\\setlength{\\tabcolsep}{%s}" % sep)
    if wide:
        tex = tex.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
    open(os.path.join(TAB, fname), "w").write(tex)


def tab_llm(df):
    g = agg(df, ["query", "N"])
    piv = g.pivot_table(index=["query", "N"], columns="method", values="cost")
    cov = g.pivot_table(index=["query", "N"], columns="method", values="cov")
    met = df[df.method == "Ladder"].groupby(["query", "N"]).met.mean()
    M = ["SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shrink", "Ladder"]
    rows = []
    for q in S.QUERIES:
        ns = [n for (qq, n) in piv.index if qq == q and n >= 40000]
        if not ns: continue
        r = piv.loc[(q, ns[0])]; mn = r[M].min()
        dag = met.loc[(q, ns[0])] < 0.95
        name = QNICE[q] + (r"$^\dagger$" if dag else "")
        rows.append([name] + [(r"\textbf{%.2f}" if abs(r[m] - mn) < 1e-12 and not dag else "%.2f") % r[m] for m in M] +
                    ["%.2f" % cov.loc[(q, ns[0]), "Ladder"]])
    simple_table(rows, ["Query", "SRS", "PPI++", "Strat", "PPI++-sub", "Strat-shr.", r"\textsc{Ladder}", "Cov."],
                 r"LLM oracle (DeepSeek V4-Pro with reasoning; tuples it cannot label are excluded): mean cost (USD) for $\pm5\%$ at 95\%, tables of about $10^5$ tuples, 200 runs per cell. $^\dagger$ as in Table~\ref{tab:main}.",
                 "tab:llm", "llm_oracle.tex", "lrrrrrrr", sep="2.2pt")
    return g


def pct(v):
    v = round(v)
    return "0\\%" if v == 0 else ("+%d\\%%" % v if v > 0 else "$-$%d\\%%" % -v)


def tab_ablation(df):
    g = df.groupby(["query", "tag"]).cost.mean().unstack()
    base = g["abl-Ladder"]
    cols = [("abl-Ladder-nostrat", "no strat."), ("abl-Ladder-nobundle", "no bundles"), ("abl-Ladder-len1", "one level"), ("cat-DeepSeek", "DeepSeek only"),
            ("cat-Kimi", "Kimi only"), ("cat-GLM", "GLM only")]
    rows = []
    for q in S.QUERIES:
        if q not in g.index: continue
        rows.append([QNICE[q], "%.1f" % base[q]] + [pct(100 * (g.loc[q, c] / base[q] - 1)) for c, _ in cols])
    gm = lambda c: 100 * (np.exp(np.mean(np.log(g[c] / base))) - 1)
    rows.append(["Geo. mean", "--"] + [("0.0\\%" if abs(gm(c)) < 0.05 else ("+%.1f\\%%" % gm(c)) if gm(c) > 0 else ("$-$%.1f\\%%" % -gm(c))) for c, _ in cols])
    simple_table(rows, ["Query", "\\textsc{Ladder} (\\$)"] + [n for _, n in cols],
                 "Ablation ($N=10^6$; Offensive $4.3\\times10^5$; 200 runs per cell, all at equal delivered precision): cost of each variant relative to full \\textsc{Ladder} (positive = more expensive).",
                 "tab:ablation", "ablation.tex", "lrrrrrrr", wide=True, sep="6pt")
    return g


def tab_groupby(df):
    g = df.groupby(["query", "N", "method"]).agg(cost=("cost", "mean"), allcov=("all_covered", "mean"),
                                                 cov=("cov_rate", "mean"), hw=("max_rel_hw", "mean")).reset_index()
    rows = []
    for q in S.GROUPBY:
        for N in sorted(g.N.unique()):
            d = g[(g["query"] == q) & (g.N == N)].set_index("method")
            if d.empty: continue
            mn = d.cost.min()
            rows.append([GBNICE.get(q, q), f"{N:,}".replace(",", "{,}")] +
                        [("\\textbf{%.1f}" if abs(d.cost[m] - mn) < 1e-9 else "%.1f") % d.cost[m] for m in ["SRS", "PPI++", "Ladder"]] +
                        ["%.2f" % d.loc["Ladder", "cov"], "%.2f" % d.loc["Ladder", "allcov"]])
    simple_table(rows, ["Query", "$N$", "SRS", "PPI++", "\\textsc{Ladder}", "Cov.", "All"],
                 "GROUP BY COUNT with every group within $\\pm10\\%$ (95\\%), all methods with top-up: mean cost (USD), \\textsc{Ladder}'s per-group coverage and fraction of runs in which all groups are covered.",
                 "tab:groupby", "groupby.tex", "llrrrrr")
    return g


if __name__ == "__main__":
    import experiments as E
    what = sys.argv[1:] or ["motivation"]
    if "motivation" in what or "all" in what: print(fig_motivation(E.ALL9).round(3).to_string())
    if "main" in what or "all" in what:
        d = rd("main"); g = tab_main(d); fig_scale(d)
        print(g.pivot_table(index=["query", "N"], columns="method", values="cost").round(2).to_string())
        print(g.pivot_table(index=["query", "N"], columns="method", values="cov").round(3).to_string())
    if "rest" in what or "all" in what:
        print(tab_llm(rd("llm_oracle")).pivot_table(index=["query", "N"], columns="method", values="cost").round(3).to_string())
        fig_factor(rd("oracle_cost"), "c0", "oracle cost per label (USD)", "oracle_cost.pdf")
        fig_factor(rd("eps"), "eps", "relative error target", "eps.pdf")
        print(tab_ablation(rd("ablation")).round(2).to_string())
        print(tab_groupby(rd("groupby")).round(3).to_string())
        allg = fig_coverage([rd("main"), rd("llm_oracle"), rd("oracle_cost"), rd("eps")])
        print(allg.groupby("method")["cov"].describe().round(3).to_string())
        o = rd("overhead"); print(o.describe().round(3).to_string())
