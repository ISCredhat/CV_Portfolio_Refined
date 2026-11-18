# data/

Gitignored except this file and `manual/` seed CSVs (force-added if wanted).

- `pmdetect.duckdb`: the local store (schema in `src/pmdetect/store/schema.sql`).
- `raw_cache/`: raw API JSON keyed by sha256 of url + sorted params.
- `manual/`: hand-curated inputs (`info_type_labels.csv`, `known_cases.csv`, `news_time_overrides.csv`).
