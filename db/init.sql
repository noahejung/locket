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

-- Task 14: tiered entity resolution's confirm queue. Not listed in the plan's Task 14
-- "Files" section (which names only resolution.py + its test), but resolution.py must not
-- talk to Postgres directly — "store.py is the only module that talks to Postgres" is a
-- hard boundary stated in PLAN.md's File Structure section — so persisting proposed merges
-- for the CLI's y/n confirm queue (Task 19) belongs here, alongside the other tables.
CREATE TABLE merge_proposals (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  mention text NOT NULL,
  candidate_entity_id uuid NOT NULL REFERENCES entities(id),
  evidence text,
  score real NOT NULL,
  status text NOT NULL DEFAULT 'pending',            -- pending|confirmed|rejected
  created_at timestamptz DEFAULT now()
);

-- Task 17/18: versioned profile snapshots. Not listed in Task 17's plan "Files" section,
-- but the MCP server's get_profile tool (Task 17) reads what profile.py's synthesize()
-- (Task 18) writes, and store.py is the only module allowed to talk to Postgres -- same
-- rationale as Task 14's merge_proposals addition above. Every synthesize() call that finds
-- new facts inserts a new row (versioned, never mutated in place); get_profile always reads
-- the newest one.
CREATE TABLE profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  body text NOT NULL,
  fact_count integer NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Fix-wave-1 item 8b: pipeline-run idempotency watermark. A window (a specific
-- chronological slice of raw_items, identified by extraction.graph.window_hash) that has
-- already been sent through extraction -- successfully or given-up-on, either way the
-- pipeline already spent up to MAX_ATTEMPTS model calls reaching a terminal state for it --
-- is recorded here so a second `pipeline run` over the same corpus doesn't re-call the
-- model (and re-bill) for it. Same "store.py is the only module that talks to Postgres"
-- rationale as merge_proposals/profiles above.
CREATE TABLE extracted_windows (
  window_hash text PRIMARY KEY,
  created_at timestamptz DEFAULT now()
);
