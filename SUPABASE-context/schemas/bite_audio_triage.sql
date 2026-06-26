create table public.bite_audio_triage (
  id uuid not null default gen_random_uuid (),
  bite_id uuid not null,
  language text not null,
  assignment_id uuid null,
  decision text not null,
  confidence double precision null,
  reasoning text null,
  segments_to_regen jsonb null default '[]'::jsonb,
  feedback_classification jsonb null default '[]'::jsonb,
  paragraph_timings jsonb null default '[]'::jsonb,
  model_used text null,
  cost_input_tokens integer null default 0,
  cost_output_tokens integer null default 0,
  status text null default 'completed'::text,
  admin_action text null,
  admin_notes text null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint bite_audio_triage_pkey primary key (id),
  constraint bite_audio_triage_assignment_id_fkey foreign KEY (assignment_id) references content_assignments (id) on delete set null,
  constraint bite_audio_triage_bite_id_fkey foreign KEY (bite_id) references bites (id) on delete CASCADE,
  constraint bite_audio_triage_language_check check ((language = any (array['en'::text, 'hi'::text]))),
  constraint bite_audio_triage_admin_action_check check (
    (
      (admin_action is null)
      or (
        admin_action = any (
          array[
            'approved'::text,
            'rejected'::text,
            'modified'::text
          ]
        )
      )
    )
  ),
  constraint bite_audio_triage_status_check check (
    (
      status = any (
        array[
          'completed'::text,
          'failed'::text,
          'expired'::text
        ]
      )
    )
  ),
  constraint bite_audio_triage_confidence_check check (
    (
      (confidence >= (0)::double precision)
      and (confidence <= (1)::double precision)
    )
  ),
  constraint bite_audio_triage_decision_check check (
    (
      decision = any (
        array[
          'full'::text,
          'partial'::text,
          'skip'::text,
          'escalate'::text
        ]
      )
    )
  )
) TABLESPACE pg_default;

create index IF not exists idx_bite_audio_triage_bite_lang on public.bite_audio_triage using btree (bite_id, language) TABLESPACE pg_default;

create index IF not exists idx_bite_audio_triage_assignment on public.bite_audio_triage using btree (assignment_id) TABLESPACE pg_default
where
  (assignment_id is not null);

create index IF not exists idx_bite_audio_triage_decision on public.bite_audio_triage using btree (decision, status) TABLESPACE pg_default;

create index IF not exists idx_bite_audio_triage_pending_review on public.bite_audio_triage using btree (created_at desc) TABLESPACE pg_default
where
  (
    (admin_action is null)
    and (status = 'completed'::text)
  );

create trigger trg_bite_audio_triage_updated_at BEFORE
update on bite_audio_triage for EACH row
execute FUNCTION update_bite_audio_triage_updated_at ();
