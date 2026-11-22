"""LLM baseline for interactive clustering, built around a method-agnostic test harness.

Read in this order:
    data.py             the items and the simulated user (src/utils PokemonBenchmark and sample_triplets)
    harness.py          the loop every method runs in (the Strategy interface, metrics, saving, resume)
    llm.py              talking to the LLM, and what the three LLM strategies share
    schema_strategy.py  the LLM designs features; HDBSCAN or k-means clusters
    open_strategy.py    the LLM designs its own per-item method; HDBSCAN or k-means clusters
    direct_strategy.py  the LLM clusters every item itself
"""

from .direct_strategy import DirectStrategy
from .open_strategy import OpenStrategy
from .schema_strategy import SchemaStrategy

STRATEGIES = {"schema": SchemaStrategy, "open": OpenStrategy, "direct": DirectStrategy}
