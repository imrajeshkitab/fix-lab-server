create table public.content_assignments (
  id uuid not null default extensions.uuid_generate_v4 (),
  reviewer_id uuid null,
  content_type text null,
  content_id uuid null,
  status text null,
  assigned_by uuid null,
  assigned_at timestamp with time zone null default now(),
  completed_at timestamp with time zone null,
  assigned_languages jsonb null,
  iteration_count integer null default 0,
  updated_at timestamp with time zone not null default now(),
  constraint content_assignments_pkey primary key (id),
  constraint content_assignments_assigned_by_fkey foreign KEY (assigned_by) references profiles (id),
  constraint content_assignments_reviewer_id_fkey foreign KEY (reviewer_id) references profiles (id)
) TABLESPACE pg_default;

create index IF not exists idx_assignments_reviewer on public.content_assignments using btree (reviewer_id) TABLESPACE pg_default;

create index IF not exists idx_assignments_content on public.content_assignments using btree (content_id) TABLESPACE pg_default;