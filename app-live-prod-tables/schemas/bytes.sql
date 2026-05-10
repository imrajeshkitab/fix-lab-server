create table public.bytes (
  created_at timestamp with time zone not null default now(),
  source_id text null,
  cover_page text not null default ''::text,
  audio text not null default ''::text,
  duration text not null,
  content text not null,
  title text not null,
  id uuid not null default gen_random_uuid (),
  category text null,
  author text not null,
  priority boolean not null default false,
  published boolean not null default false,
  updated_at timestamp with time zone not null default now(),
  source text null,
  language text null,
  difficulty text null,
  affiliate_links text null default 'https://heartfulness.org/magazine/editions'::text,
  constraint bytes_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_bytes_filters on public.bytes using btree (source, published, language, category) TABLESPACE pg_default;

create index IF not exists idx_bytes_created_at on public.bytes using btree (created_at desc) TABLESPACE pg_default;