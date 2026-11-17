# LLM baseline experiment

**Question:** can an LLM, given only the raw data (Pokémon images and/or text descriptions) and the
same simulated feedback our models get, reach the clustering the user wants? This gives a baseline
for the projection models, scored with exactly the same metrics.

## How it works

The LLM never sees our embeddings. It builds its own representation, using one of two strategies
(`--strategy`). Both feed the same k-means, feedback and metrics, and in both the LLM can never set
an individual item's scores.

### Strategy A: feature schema (`--strategy schema`, the default)

1. **Design (round 0).** The LLM sees 40 random items and proposes 15-30 features, e.g. "dominant
   body colour is blue (0-1)" or "has wings (binary)". It is not told the hidden criterion (color),
   the labels, or how many groups k-means will use.
2. **Score.** Every item is scored on every feature, 20 items per call. Scores are cached per
   feature definition.
3. **Cluster.** Each feature is z-scored and multiplied by its weight, then grouped with the same
   k-means call Jay's script uses (`clustering.cluster_embeddings`, k = 3, seed 42).
4. **Feedback.** Jay's simulated user (`tests/triplet_by_column.py: select_feedback`, unchanged)
   picks a mis-grouped anchor and gives 32 statements: "item_A should be closer to item_P than to
   item_N".
5. **Revise.** The LLM sees its schema, this round's statements (with the items' images/text and
   current scores) and all earlier statements. It may only change the *feature set*: reweight,
   drop, or add up to 5 features. **It cannot edit any item's scores.** New features are scored for
   every item by a separate call, so test ARI measures whether the revision generalizes rather than
   memorizes.
6. Repeat 2-5 for 30 rounds, or until ARI ≥ 0.95.

### Strategy B: hypothesis + rescore (`--strategy hypothesis`, results in `..._hyp/`)

The LLM keeps an explicit written guess of the user's hidden criterion.

1. **Design.** From the same 40 items it writes a `criterion` (e.g. "group by overall body shape")
   and 3-8 `dimensions` that measure it.
2. **Score.** Every item is scored on the dimensions, and the scoring prompt includes the criterion
   as context. The criterion is part of the score-cache key, so a new criterion rescores every item.
3. **Revise.** After each round of feedback, it returns a complete new criterion and dimension set:
   it can keep, reword or replace anything.

B tests the LLM's main strength: guessing the rule from a few examples and applying it to every
item at once. A, by contrast, adjusts a large fixed feature set a little at a time. B costs more per
round (a full rescore, about 22 calls, whenever the criterion changes). `history.csv` has a
`criterion_changed` column, and `edits.json` records the criterion for every round.

### Shared by both strategies

Everything mirrors `tests/triplet_by_column.py`, so `history.csv` overlays
`results/triplet_by_column/color_3c_all/history.csv` directly:

- the same items, in the same order;
- the same 20% held-out test set and AUC pairs;
- the same rng stream for feedback;
- the same metrics (`src/utils/metrics.all_metrics`).

## Arms × models (all through Amazon Bedrock)

| inputs | Sonnet 5.5 | Opus 5.5 | GPT-6.1 Terra |
|---|---|---|---|
| image | ✓ | ✓ | ✓ |
| image+text | ✓ | ✓ | ✓ |
| text | ✓ | ✓ | ✓ |

Each cell is run with both strategies: 3 models × 3 arms × 2 strategies = 18 runs.

Rough cost per 30-round run:
- Sonnet: about $5-15 (A) or $10-20 (B).
- Opus: about twice Sonnet.
- Terra: not tracked until you pass its price.

All 18 runs together come to roughly $250-400. `-MaxUsd` caps each run. To start smaller, use
`-Models`, `-Strategies` and `-Inputs`, e.g. `.\llm_experimentation\run_all.ps1 -Models sonnet -Strategies hypothesis`.

## Running

```powershell
pip install -r llm_experimentation/requirements.txt
$env:AWS_BEARER_TOKEN_BEDROCK = "<bedrock api key>"     # never commit this

# 0. free offline check of the whole pipeline (fake LLM, random answers)
python llm_experimentation/run_experiment.py --dry-run --rounds 3 --overwrite
python llm_experimentation/tests/test_pipeline.py

# 1. confirm access + model IDs (about $0.01): one tiny call per model
python llm_experimentation/check_models.py

# 2. small real smoke run (60 items, 2 rounds): check llm_calls.jsonl and the cost
python llm_experimentation/run_experiment.py --model sonnet --inputs image --sample-size 60 --rounds 2
python llm_experimentation/run_experiment.py --model sonnet --inputs image --sample-size 60 --rounds 2 --strategy hypothesis

# 3. everything, then the comparison plot
.\llm_experimentation\run_all.ps1 -MaxUsd 25
python llm_experimentation/compare_results.py
```

The region defaults to **us-east-2** (where the key and model access are); pass `--region` /
`-Region` to change it. `$env:...` lasts for that terminal only; `setx AWS_BEARER_TOKEN_BEDROCK "<key>"`
persists it for new terminals.

**Rate limits and resuming.** Bedrock limits tokens per minute and per day for each model. Per-minute
throttling is retried automatically (`[score] ThrottlingException, retry ...`); `--workers 2` reduces
it. A "Too many tokens per day" error stops the run cleanly after the last finished round. Rerun the
same command with `--resume` (no `--overwrite`) once the quota resets. The run continues from the
saved state, so the result matches an uninterrupted run, and `--max-usd` keeps counting the whole run.

**Model IDs.** The IDs in `llm_cluster/models.py` are the expected inference-profile IDs.
`check_models.py` tests them and, if one fails, prints the matching IDs it can see in the region;
pass `--model-id` (or edit `models.py`) to use those. Terra's prices aren't filled in yet: pass
`--price-in/--price-out` (USD per 1M tokens) to get dollar tracking and an enforced `--max-usd` for
it. Token counts are always logged.

## Outputs: `results/<column>_<config>_<inputs>_<model>/`

| file | contents |
|---|---|
| `history.csv` | per round: Jay's metric columns, plus feature counts, parse failures, tokens and $ |
| `assignments.csv` | each item's dex, name, anonymous id, label, test flag, clusters before and after |
| `edits.json` | per round: anchor, feedback type, what the LLM changed, its hypothesis notes |
| `schema_round_XX.json`, `V_round_XX.npy` | the feature schema and the vectors k-means saw |
| `triplets.npy` | every feedback triplet (row indices) |
| `llm_calls.jsonl` | every prompt and reply (images shown as `[image item_xxx]`) with usage |
| `scores.jsonl` | cached item scores per feature definition |

`compare_results.py` writes `results/comparison_<config>.png` and `summary_<config>.csv`.

## Known confounds and choices

- **Recognition.** IDs are anonymous and shuffled, and every Pokémon name is redacted from the text.
  Even so, a model may recognize a Pokémon from its image and recall its official colour. That is
  world knowledge our embedding models only partly have.
- **The text arms are easy for color.** Descriptions often state the colour outright ("covered in
  yellow fur"). Our text-embedding runs get the same text, so the comparison is still fair. For a
  matching model baseline, run Jay's script with `--modality text`.
- **Reasoning settings.** Each model runs at its Bedrock default reasoning setting and gets the same
  prompts.
- **Fallbacks.** When a reply is still malformed after one retry, missing scores are filled with 0.5
  and counted in `filled_scores` / `llm_parse_failures`. Watch those columns.
- **Stretch.** DBSCAN/HDBSCAN can be run offline on the saved `V_round_XX.npy` vectors, so the number
  of clusters doesn't have to be given.
