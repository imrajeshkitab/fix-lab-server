create table public.summaries (
  id uuid not null default extensions.uuid_generate_v4 (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  cover text null,
  source_id text null,
  author text null,
  title text null,
  feedback jsonb null,
  content jsonb null,
  audio jsonb null,
  source text null,
  tags jsonb null,
  linear_identifier text null,
  constraint summaries_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_summaries_source_id on public.summaries using btree (source_id) TABLESPACE pg_default;