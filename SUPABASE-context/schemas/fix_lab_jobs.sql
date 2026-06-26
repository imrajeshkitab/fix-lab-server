create table public.fix_lab_jobs (
  id uuid not null default gen_random_uuid (),
  status text not null default 'running'::text,
  total integer null default 0,
  completed integer null default 0,
  failed integer null default 0,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint fix_lab_jobs_pkey primary key (id)
) TABLESPACE pg_default;