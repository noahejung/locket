CREATE EXTENSION IF NOT EXISTS vector;

-- Task 8: bi-temporal fact store. Design steals mem0's flat-fact shape + audit history and
-- graphiti's bi-temporal fields (valid_at/invalid_at/expired_at) — see PLAN.md Task 8.

CREATE TABLE raw_items (
  id text PRIMARY KEY, source text NOT NULL, ts timestamptz,
  sender text, body text, media_path text, is_system boolean DEFAULT false, meta jsonb DEFAULT '{}'
);

CREATE TABLE entities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL, kind text NOT NULL,           -- person | place | org
  aliases text[] DEFAULT '{}',
  embedding vector(384)
);

CREATE TABLE facts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kind text NOT NULL,                                -- person|place|event|relationship|habit|preference
  body jsonb NOT NULL,                               -- the typed Fact payload
  statement text NOT NULL,                           -- NL rendering, what gets embedded
  confidence real NOT NULL,
  happened_at text,                                  -- ISO date/range from ExtractedFact.happened_at (timeline + FactRow need it as a column)
  entity_ids uuid[] DEFAULT '{}',
  provenance text[] NOT NULL,                        -- raw_items.id values — every fact cites sources
  valid_at timestamptz, invalid_at timestamptz, expired_at timestamptz,   -- bi-temporal (graphiti pattern)
  created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
  hash text UNIQUE,                                  -- md5(statement) pre-insert dedup (mem0 pattern)
  embedding vector(384)
);

CREATE TABLE fact_history (
  id bigserial PRIMARY KEY, fact_id uuid NOT NULL, event text NOT NULL,   -- ADD|UPDATE|EXPIRE
  prev jsonb, next jsonb, at timestamptz DEFAULT now()
);

CREATE INDEX ON facts USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON entities USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON raw_items (ts);
