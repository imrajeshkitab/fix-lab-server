create table public.bites (
  id uuid not null default extensions.uuid_generate_v4 (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  cover text null,
  source_id text null,
  author text null,
  title text null,
  audio jsonb null,
  content jsonb null,
  source text null,
  tags jsonb null,
  audio_version jsonb null default '{"en": 1, "hi": 1}'::jsonb,
  linear_identifier text null,
  difficulty text null,
  category text null,
  author_bilingual jsonb null,
  title_bilingual jsonb null,
  constraint bites_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_bites_source_id on public.bites using btree (source_id) TABLESPACE pg_default;