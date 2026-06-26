create table public.profiles (
  id uuid not null default gen_random_uuid (),
  email text null,
  full_name text null,
  role text null,
  created_at timestamp with time zone null default now(),
  constraint profiles_pkey primary key (id)
) TABLESPACE pg_default;