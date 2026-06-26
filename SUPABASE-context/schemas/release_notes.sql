create table public.release_notes (
  id uuid not null default gen_random_uuid (),
  version text not null,
  title text not null,
  release_date date not null,
  summary text null,
  is_published boolean null default false,
  created_at timestamp with time zone null default now(),
  constraint release_notes_pkey primary key (id),
  constraint release_notes_version_key unique (version)
) TABLESPACE pg_default;