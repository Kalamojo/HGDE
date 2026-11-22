# Tests

All offline: no Bedrock calls, no cost.

```powershell
python -m pytest llm_experimentation/tests/unit     # a few seconds
python -m pytest llm_experimentation/tests          # + the end-to-end dry runs, about 80 s
```

`helpers.py` has `make_toy_sel` (30 synthetic items, no database) and `ScriptedClient` (an LLM whose
replies the test chooses). Tests that need the real Pokémon data are skipped without it.

| file | covers |
|---|---|
| `unit/test_metrics.py` | `src/utils/metrics.py`: contingency table, pairwise P/R (= listing every pair), `sample_pairs`, `pair_auc` (= comparing every pair), `compute_metrics`, test ARI in `all_metrics` |
| `unit/test_state.py` | `src/utils/state.py`: `ClusterState`'s AUC pairs and `log_results` |
| `unit/test_benchmarks.py` | `src/utils/data/benchmarks.py`: the test split; test items never in feedback; same seed = same feedback; none once perfect |
| `unit/test_data.py` | `select_items` = run_benchmarks.py's benchmark1, name redaction, `next_feedback` |
| `unit/test_harness.py` | `Harness.run` saving and stopping, `StopRun`, `--resume` |
| `unit/test_llm.py` | `extract_json`, `JsonlCache`, the hint margin, `ask`'s retry, `draft_loop`, the budget stop |
| `unit/test_schema_strategy.py` | `vectorize`, `apply_revision`, missing scores retried then filled |
| `unit/test_open_strategy.py` | `check_method`, bad executor rows filled |
| `unit/test_direct_strategy.py` | `validate_grouping` (auto and k), groups ↔ vectors, an invalid reply keeps the grouping, auto vs k prompts |
| `integration/test_end_to_end.py` | a dry run of each strategy: resume = uninterrupted run, no test item in feedback, no Pokémon name in any prompt |
