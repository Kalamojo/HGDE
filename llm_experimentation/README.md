# LLM baseline experiment

**Question:** can an LLM, given only the raw data (Pokémon images and/or text descriptions) and the
same simulated feedback our models get, reach the clustering the user wants? This is a baseline for
the projection models (Jay's triplet model, the warp), scored with exactly the same metrics.

Three strategies, each giving the LLM one more step of control:

| `--strategy` | what the LLM controls | what stays fixed | what it tests |
|---|---|---|---|
| **`schema`** (default) | feature definitions and weights | the representation format, k-means | the LLM as a feature engineer |
| **`open`** | its own method for turning an item into numbers | k-means | the LLM as a replacement for the embedding model |
| **`direct`** | the clustering itself | only k = 3 | the LLM solving the whole problem |

In `schema` and `open` the LLM never sets an individual item's numbers, so test ARI measures
generalization. In `direct` it places every item itself, so compare `direct` by **test ARI** only.

## Files

```
run_experiment.py        one run: flags -> strategy -> harness; writes run_config.json
summarize_methods.py     what an open / direct run decided, round by round
compare_results.py       plots all runs on one item set against Jay's run
check_models.py          one tiny call per Bedrock model id
run_all.ps1              every strategy x model x inputs cell
llm_cluster/             read in this order:
  data.py                the items (Jay's selection, anonymous ids, redacted names) and Jay's feedback
  harness.py             the loop every method runs in (Strategy interface, metrics, saving, resume); no LLM code
  llm.py                 model list, Bedrock + fake clients, and LLMStrategy (what the three strategies share)
  schema_strategy.py     each strategy file holds its state, prompts and update
  open_strategy.py
  direct_strategy.py
tests/                   offline tests (see tests/README.md)
(run output goes to the repo-level results/llm_experimentation/, one folder per run; its archive/ is ignored by compare_results.py)
```

## The harness (`llm_cluster/harness.py`)

Every method runs the same loop. A method is a `Strategy` with:

| method | what it does |
|---|---|
| `start()` | the round-0 state |
| `embed(state)` | vectors `V` (n × d) for all items |
| `cluster(V)` | cluster labels; default is Jay's k-means call |
| `update(state, RoundInput)` | learn from this round's hints; returns the new state + what changed |
| `log_columns`, `summary`, `restore` | extra `history.csv` columns, a console line, counters on `--resume` |

1. **Items.** `data.select_items` repeats Jay's `archive/tests/triplet_by_column.py` selection exactly: color,
   top 3 labels, all 439 items, the same 20% held-out test set (88 items) and AUC pairs, and Jay's rng
   in the same state. Each item gets an anonymous id from a seeded permutation (dex order would leak
   evolution families), and every Pokémon name is redacted from the descriptions.
2. **Each round:** `V = embed(state)`, `pred = cluster(V)`, then `src/utils/metrics.all_metrics`
   (ARI, test ARI, test pair AUC, pairwise precision/recall, Jay's F1).
3. **Feedback.** Jay's `select_feedback`, unchanged: an anchor whose cluster is wrong, and 32
   `(anchor, positive, negative)` triplets, never using held-out items. The LLM reads each as
   `item_A should be closer to item_P than to item_N`.
4. **`update`**, then back to 2. Stops after `--rounds`, at `--goal-ari`, when nothing is left to fix,
   or when the strategy raises `StopRun` (e.g. `--max-usd`).

A hint is **satisfied** when, in the L2-normalized space, `d(A, N) − d(A, P) ≥ 0.2` (Jay's margin).

Every round is saved; `--resume` loads the last saved round (including Jay's rng state, in
`feedback_state.json`) and continues exactly as if the run had never stopped.

### Plugging in another method

The harness has the same shape as Kolade's `src/utils` framework (`ClusterState.recluster_round`):

| `src/utils` | here |
|---|---|
| `Benchmark` (items, labels, `test_mask`) | `data.Selection` |
| `sampler.sample_triplets(pred)` | `data.next_feedback` (Jay's `select_feedback`) |
| `Clusterer.run(embeddings)` | `Strategy.cluster(V)` |
| `ClusterEval.update` + `train_model` | `Strategy.update(state, RoundInput)` |
| `model(original_embeddings)` | `Strategy.embed(state)` |
| `ClusterState.log_results` / `all_metrics` | `Harness.metrics` / `all_metrics` |

A projection model fits as a `Strategy` (`start` = init the net, `embed` = `net(X)`, `update` = train
on the triplets). Two things differ and must be settled when merging: the feedback (Jay's
`select_feedback` vs `Benchmark.sample_triplets`) and the test split (Jay's vs `set_test_items`).

### What the LLM strategies share (`LLMStrategy` in `llm.py`)

- **`ask`**: one stateless call; the reply must parse as a pydantic model (and pass a check). If not,
  it is asked once more with the error.
- **Batches**: per-item calls in batches of 20 items, `--workers` at a time. Items still missing after
  two passes get a default value, counted in the `filled` column.
- **`JsonlCache`**: per-item outputs cached by a hash of the definition, so nothing unchanged is re-run.
- **`draft_loop`** (open, direct): up to `--drafts` steps per round; at each the LLM may `test` a
  version against all hints, `commit` a version, or `commit_draft` one it tested. The last step is
  applied whatever it says; if the LLM fails, its best tested draft is used.

## Strategy `schema`

The state is a `Schema`: a list of `Feature(id, definition, kind "scale" | "binary", weight 0–5)`.

1. **Design:** the LLM sees 40 random items and proposes 15–30 features judgeable for one item on its
   own (e.g. "dominant body colour is blue, 0–1"). Every weight starts at 1.
2. **Score:** every item is scored on every feature (clipped to 0–1; missing scores become 0.5),
   cached in `scores.jsonl`. A reworded feature is rescored; a reweighted one isn't.
3. **Vectorize:** each feature is z-scored across items, then multiplied by its weight → `V` → k-means.
4. **Revise:** the LLM sees its features, all hints, the share satisfied, and this round's hinted items
   with their scores. It may only reweight (0–5), drop, or add up to 5 features; it **cannot edit any
   item's scores**. A revision leaving fewer than 2 active features is refused.

Cost: about $15–20 for 30 Sonnet rounds.

## Strategy `open`

The LLM is told only the goal and how its output is used; never *how* to represent the items.

1. **The state is a `Method`:** `approach` (plain words), `instructions` for one item, `outputs` (what
   each of 1–32 numbers means), up to 12 `reference_items`, and `notes`.
2. **Executor:** separate calls apply the method to every item (with the reference items' pictures).
   The executor never sees the hints. There's no z-scoring: `V` is exactly its numbers. Cached in
   `vectors.jsonl`.
3. **Revise:** the draft loop. A tested draft is executed on every hinted item and scored against all
   hints; the LLM gets the shares satisfied and the violated hints back.

It mirrors Jay's model: a function applied to each item alone (the method), capacity to bend around
specific items (reference items), several passes over all triplets per round (test drafts).

Cost: up to about 3× schema.

## Strategy `direct`

The LLM clusters every item itself into exactly k = 3 groups (the k Jay's k-means is given).

1. **Seeing all 439 items:** contact sheets of 6 × 6 tiles with each id under its picture (about 13
   images), grouped by its current clustering; the text arm lists `item_xxx: description`.
2. **The state is a `Grouping`:** `approach`, `groups: {name: [ids]}`, `notes`.
3. **Revise:** the draft loop, showing every item by current group, all hints and the violated ones.
   A grouping must have exactly k non-empty groups and every id once; if it's still invalid after the
   retry, the current grouping stays.
4. **Vectors:** `V` is the one-hot of the groups, so metrics and feedback work unchanged. A hint is
   then satisfied exactly when A and P share a group and N doesn't.

Hinted items can be placed exactly, so all-items ARI includes memorization: compare by test ARI.

Cost: roughly $10–25 for 30 Sonnet rounds.

## Running

```powershell
pip install -r llm_experimentation/requirements.txt
$env:AWS_BEARER_TOKEN_BEDROCK = "<bedrock api key>"     # never commit this

python -m pytest llm_experimentation/tests                # free offline tests (fake LLM)
python llm_experimentation/check_models.py                # confirm the model ids (about $0.01)

# cheap smoke runs (60 items, 2 rounds), then full runs and the comparison plot
python llm_experimentation/run_experiment.py --model sonnet --sample-size 60 --rounds 2 --strategy open
.\llm_experimentation\run_all.ps1 -Models sonnet -Inputs image -MaxUsd 60
python llm_experimentation/compare_results.py
```

Main flags: `--strategy`, `--model` (ids and prices are in `MODELS` in `llm.py`), `--inputs image |
image+text | text`, `--drafts`, `--rounds`, `--max-usd`, `--workers`, `--resume`, `--dry-run`, and
Jay's item-set flags (`--column`, `--top-labels`, `--sample-size`, `--test-fraction`, `--seed`). Fix
every setting before a run; never tune one on test ARI.

Rate limits are retried automatically (`--workers 2` reduces them). If a run stops on an error or the
daily token quota, rerun the same command with `--resume`; `--max-usd` keeps counting the whole run.
One run swings a lot from round to round, so compare strategies over 2–3 `--seed`s.

## Outputs: one folder per run

| file | contents |
|---|---|
| `run_config.json` | every setting of the run |
| `history.csv` | per round: Jay's metric columns, hint satisfaction, strategy columns, tokens and $ |
| `edits.json` | per round: the anchor, and what the LLM changed / tested / committed |
| `schema_round_XX.json`, `V_round_XX.npy` | the state (Schema / Method / Grouping) and the vectors clustered |
| `assignments.csv` | each item's name, anonymous id, label, test flag, clusters before and after |
| `triplets.npy`, `feedback_state.json` | every feedback triplet; Jay's rng state (for `--resume`) |
| `llm_calls.jsonl` | every prompt and reply (images shown as `[image ...]`) |
| `scores.jsonl` / `vectors.jsonl` | schema's / open's cached per-item outputs |

Runs saved before the resume change (all of today's results) still plot, but can't be resumed. The
two `schema_momentum` runs are from the removed momentum variant and plot as their own series.

## Known confounds

- **Recognition.** Ids are anonymous and names redacted, but a model may still recognize a Pokémon
  from its picture and recall its official colour: world knowledge our embedding models only partly have.
- **The text arm is easy for color.** Descriptions often state the colour; for a matching model
  baseline run Jay's script with `--modality text`.
- **Fallbacks.** Watch the `filled` column: values the LLM never gave.
