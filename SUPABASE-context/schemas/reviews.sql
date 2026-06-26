create table public.reviews (
  id uuid not null default extensions.uuid_generate_v4 (),
  assignment_id uuid null,
  reviewer_id uuid null,
  rating integer null,
  is_verified boolean null default false,
  created_at timestamp with time zone null default now(),
  feedback_details jsonb null,
  constraint reviews_pkey primary key (id),
  constraint reviews_assignment_id_fkey foreign KEY (assignment_id) references content_assignments (id),
  constraint reviews_reviewer_id_fkey foreign KEY (reviewer_id) references profiles (id)
) TABLESPACE pg_default;

create index IF not exists idx_reviews_assignment on public.reviews using btree (assignment_id) TABLESPACE pg_default;