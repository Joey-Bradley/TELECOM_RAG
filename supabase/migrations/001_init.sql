-- ============================================================
-- Telecom RAG Assistant — Supabase Database Setup
-- Run this in the Supabase SQL Editor (supabase.com → SQL Editor)
-- ============================================================

-- 1. Enable pgvector extension
create extension if not exists vector;


-- ============================================================
-- 2. Telecom docs table (shared knowledge base)
-- ============================================================
create table if not exists telecom_docs (
  id        bigserial primary key,
  content   text        not null,
  source    text        not null,
  embedding vector(384)           -- all-MiniLM-L6-v2 produces 384-dim vectors
);

-- IVFFlat index for fast approximate nearest-neighbor search
-- Rebuild after ingesting large document sets for best performance
create index if not exists telecom_docs_embedding_idx
  on telecom_docs
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- RLS: authenticated users can read; writes only via service role key
alter table telecom_docs enable row level security;

create policy "Authenticated users can read docs"
  on telecom_docs for select
  to authenticated
  using (true);


-- ============================================================
-- 3. Vector similarity search function
-- ============================================================
create or replace function match_telecom_docs(
  query_embedding vector(384),
  match_count     int default 5
)
returns table (
  id         bigint,
  content    text,
  source     text,
  similarity float
)
language sql stable
as $$
  select
    id,
    content,
    source,
    1 - (embedding <=> query_embedding) as similarity
  from telecom_docs
  order by embedding <=> query_embedding
  limit match_count;
$$;


-- ============================================================
-- 4. Per-user chat history
-- ============================================================
create table if not exists chat_history (
  id         bigserial    primary key,
  user_id    uuid         not null references auth.users(id) on delete cascade,
  session_id text         not null,
  role       text         not null check (role in ('user', 'assistant')),
  content    text         not null,
  sources    jsonb        not null default '[]',
  created_at timestamptz  not null default now()
);

create index if not exists chat_history_user_session_idx
  on chat_history (user_id, session_id, created_at);

-- RLS: users can only see and write their own history
alter table chat_history enable row level security;

create policy "Users can view own history"
  on chat_history for select
  using (auth.uid() = user_id);

create policy "Users can insert own history"
  on chat_history for insert
  with check (auth.uid() = user_id);

create policy "Users can delete own history"
  on chat_history for delete
  using (auth.uid() = user_id);


-- ============================================================
-- Done! Next steps:
--   1. Copy your Project URL and anon key → .env (SUPABASE_URL, SUPABASE_KEY)
--   2. Copy your service role key        → .env (SUPABASE_SERVICE_KEY)
--   3. Run: python ingest.py
--   4. Run: streamlit run app.py
-- ============================================================
