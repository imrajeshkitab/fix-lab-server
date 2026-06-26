-- ============================================================================
-- Add triage_id link to fix_lab_job_items
-- ============================================================================
-- When a regeneration is driven by an approved triage, the job item links
-- back to the bite_audio_triage row that produced the plan. This gives the
-- UI an audit trail (which triage decision led to which audio change).
--
-- Nullable: existing /api/fix-lab/regenerate flow doesn't use triage and
-- continues to work (triage_id stays null for those rows).
-- ============================================================================

ALTER TABLE fix_lab_job_items
    ADD COLUMN IF NOT EXISTS triage_id UUID REFERENCES bite_audio_triage(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_fix_lab_job_items_triage
    ON fix_lab_job_items (triage_id)
    WHERE triage_id IS NOT NULL;

COMMENT ON COLUMN fix_lab_job_items.triage_id IS
    'Optional link to the bite_audio_triage row that triggered this regen. NULL for items started via direct /regenerate.';
