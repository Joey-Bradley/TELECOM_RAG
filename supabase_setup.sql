-- Run this in the Supabase SQL Editor before ingesting documents

-- Enable pgvector extension
create extension if not exists vector;

-- Create the documents table
create table if not exists telecom_docs (
  id bigserial primary key,
  content text not null,
  source text not null,
  embedding vector(384)  -- all-MiniLM-L6-v2 outputs 384 dimensions
);

-- Create index for fast similarity search
create index if not exists telecom_docs_embedding_idx
  on telecom_docs
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Create the match function used by app.py
create or replace function match_telecom_docs(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id bigint,
  content text,
  source text,
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
