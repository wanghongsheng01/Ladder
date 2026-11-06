# Ladder: artifacts for "Spend Where It Counts: Cost-Optimal Approximate Semantic Aggregation over Heterogeneous LLM Labelers"

This repository contains everything needed to reproduce every number, table and figure of the paper.

## Contents
| Path | What |
|---|---|
| `code/tasks.py` | the seven datasets, the uniform 2,000-tuple samples (fixed seed), the semantic attributes |
| `code/label.py` | the nine physical labelers (model, vendor, prompting mode, batch size), prompts, answer parsing, list prices |
| `code/llm.py` | cached OpenAI-compatible client (keys are read from `LLM_KEYS_JSON`, never stored here) |
| `code/ladder.py` | surrogates, nested control-variate estimator, closed-form allocation, ladder search, GROUP BY allocation |
| `code/simulate.py` | replay harness and baselines (SRS, PPI++, proxy-stratified sampling) |
| `code/experiments.py` | all experiments (`main`, `llm_oracle`, `oracle_cost`, `eps`, `ablation`, `pilot`, `groupby`, `overhead`) |
| `code/make_figs.py` | all figures and tables |
| `code/summarize.py` | every number quoted in the text of the paper |
| `results/labels/<task>__<labeler>.csv` | one row per tuple: `tid, pred, cost_usd, in_tok, out_tok` — the real outputs and measured costs of every labeler |
| `results/*.csv` | per-trial results of every experiment |

The texts themselves are not redistributed; `python code/fetch_data.py` downloads the public test splits (IMDB, AG News, Yelp Review Full, TweetEval hate/offensive, CARER emotion, Amazon reviews) from the Hugging Face hub and draws the same samples (seed 20261103), so `tid` values align with the label files.

The enterprise case study (Section V-H) uses de-identified customer-service notes of a company; neither the notes nor their labels can be released. Its per-run results (`results/exp_rma*.csv`) are included, and the de-identification program is `code/deidentify.py`.

## Reproduce without any API key (minutes on a laptop)
```bash
pip install numpy scipy pandas matplotlib pyarrow
cd code
python experiments.py main        # and llm_oracle, oracle_cost, eps, ablation, pilot, groupby, overhead
python make_figs.py all
python summarize.py          # every number quoted in the text (results/final_summary.txt)
```
All experiments replay sampling plans on the released labels and costs; no LLM is called. `make_figs.py` writes the figures and LaTeX tables of the paper to `05_paper/figs/` and `05_paper/tables/` next to `code/`; the tables are byte-identical to the submitted ones.

## Re-label from scratch (needs API keys)
Create a JSON file outside the repository, e.g. `~/.config/llm_keys.json` (chmod 600):
```json
{"deepseek": {"api_key": "...", "base_url": "https://api.deepseek.com/v1"},
 "glm": {"api_key": "...", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
 "glm2": {"api_key": "...", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
 "kimi": {"api_key": "...", "base_url": "https://api.moonshot.cn/v1"}}
```
then `LLM_KEYS_JSON=~/.config/llm_keys.json python label.py <labeler> [task ...]`. Model versions served by the vendors change over time; the released label files are the ones used in the paper (collected 2026-11-03).
