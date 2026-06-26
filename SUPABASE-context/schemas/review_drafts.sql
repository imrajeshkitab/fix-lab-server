create table public.review_drafts (
  id uuid not null default extensions.uuid_generate_v4 (),
  assignment_id uuid null,
  reviewer_id uuid null,
  rating integer null,
  feedback_items jsonb null,
  verification_status text null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint review_drafts_pkey primary key (id),
  constraint review_drafts_assignment_reviewer_unique unique (assignment_id, reviewer_id),
  constraint review_drafts_assignment_id_fkey foreign KEY (assignment_id) references content_assignments (id),
  constraint review_drafts_reviewer_id_fkey foreign KEY (reviewer_id) references profiles (id)
) TABLESPACE pg_default;

create index IF not exists idx_review_drafts_assignment on public.review_drafts using btree (assignment_id) TABLESPACE pg_default;

create index IF not exists idx_review_drafts_reviewer on public.review_drafts using btree (reviewer_id) TABLESPACE pg_default;

create trigger trigger_review_drafts_updated_at BEFORE
update on review_drafts for EACH row
execute FUNCTION update_review_draft_timestamp ();