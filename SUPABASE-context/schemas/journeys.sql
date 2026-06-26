create table public.journeys (
  id uuid not null default extensions.uuid_generate_v4 (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  cover text null,
  source_id text null,
  author text null,
  title text null,
  content jsonb null,
  audio jsonb null,
  source text null,
  content_chapterwise jsonb null,
  tags jsonb null,
  constraint journeys_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_journeys_source_id on public.journeys using btree (source_id) TABLESPACE pg_default;