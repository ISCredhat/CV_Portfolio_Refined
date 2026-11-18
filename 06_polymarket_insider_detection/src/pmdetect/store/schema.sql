CREATE TABLE IF NOT EXISTS markets (
    condition_id VARCHAR PRIMARY KEY,
    event_id VARCHAR,
    slug VARCHAR,
    question VARCHAR,
    category VARCHAR,
    tags VARCHAR[],
    info_type VARCHAR,
    volume_usd DOUBLE,
    liquidity_usd DOUBLE,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    closed_ts TIMESTAMP,
    uma_status VARCHAR,
    outcome VARCHAR,
    news_ts TIMESTAMP,
    news_ts_method VARCHAR,
    in_universe BOOLEAN,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tokens (
    token_id VARCHAR PRIMARY KEY,
    condition_id VARCHAR,
    outcome_label VARCHAR
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR PRIMARY KEY,
    tx_hash VARCHAR,
    condition_id VARCHAR,
    token_id VARCHAR,
    wallet VARCHAR,
    side VARCHAR,
    outcome_label VARCHAR,
    size DOUBLE,
    price DOUBLE,
    usd DOUBLE,
    ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices (
    token_id VARCHAR,
    ts TIMESTAMP,
    price DOUBLE,
    PRIMARY KEY (token_id, ts)
);

CREATE TABLE IF NOT EXISTS wallets (
    wallet VARCHAR PRIMARY KEY,
    first_seen TIMESTAMP,
    n_markets INTEGER,
    total_usd DOUBLE,
    funding_source VARCHAR,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labels (
    label_id VARCHAR PRIMARY KEY,
    condition_id VARCHAR,
    wallet VARCHAR,
    case_name VARCHAR,
    label VARCHAR,
    source_url VARCHAR,
    notes VARCHAR
);

CREATE TABLE IF NOT EXISTS fetch_log (
    endpoint VARCHAR,
    key VARCHAR,
    last_cursor VARCHAR,
    completed BOOLEAN,
    fetched_at TIMESTAMP,
    PRIMARY KEY (endpoint, key)
);
