-- ============================================================================
-- Bite Audio Triage — Phase 1
-- ============================================================================
-- Stores AI triage analysis for each bite+language combination.
-- Each row represents one triage run: paragraph alignment (from Whisper STT),
-- feedback classification, and fix decision (from Gemini text-only call).
--
-- Admin can approve/reject/modify the AI's recommendation via admin_action.
-- Once approved, Phase 2+ will use this to drive audio regeneration.
--
-- Re-triage behavior: when a new triage runs for the same (bite_id, language),
-- previous rows are marked status='expired' (handled in fix-lab-server).
-- ============================================================================

CREATE TABLE IF NOT EXISTS bite_audio_triage (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,

    -- ── Target ────────────────────────────────────────────────────────────
    bite_id         UUID NOT NULL REFERENCES bites(id) ON DELETE CASCADE,
    language        TEXT NOT NULL CHECK (language IN ('en', 'hi')),
    assignment_id   UUID REFERENCES content_assignments(id) ON DELETE SET NULL,

    -- ── AI Decision ───────────────────────────────────────────────────────
    decision        TEXT NOT NULL CHECK (decision IN ('full', 'partial', 'skip', 'escalate')),
    confidence      FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    reasoning       TEXT,

    -- ── Structured Output (JSONB) ─────────────────────────────────────────
    -- segments_to_regen: array of {paragraph_index, issue, tts_hints, t_start, t_end, paragraph_text}
    segments_to_regen       JSONB DEFAULT '[]'::jsonb,
    -- feedback_classification: array of {feedback_index, is_actionable, category, mapped_paragraph, summary}
    feedback_classification JSONB DEFAULT '[]'::jsonb,
    -- paragraph_timings: array of {paragraph_index, t_start, t_end} from Whisper alignment
    paragraph_timings       JSONB DEFAULT '[]'::jsonb,

    -- ── Model Metadata ────────────────────────────────────────────────────
    model_used          TEXT,                -- e.g. "gemini-2.0-flash"
    cost_input_tokens   INT DEFAULT 0,
    cost_output_tokens  INT DEFAULT 0,

    -- ── Status ────────────────────────────────────────────────────────────
    -- completed: triage finished, current latest for (bite, lang)
    -- failed:    triage threw an error
    -- expired:   superseded by a newer triage run for the same (bite, lang)
    status          TEXT DEFAULT 'completed' CHECK (status IN ('completed', 'failed', 'expired')),

    -- ── Admin Review ──────────────────────────────────────────────────────
    admin_action    TEXT CHECK (admin_action IS NULL OR admin_action IN ('approved', 'rejected', 'modified')),
    admin_notes     TEXT,

    -- ── Timestamps ────────────────────────────────────────────────────────
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- ── Indexes ───────────────────────────────────────────────────────────────

-- Fast lookup by bite + language (most common query)
CREATE INDEX IF NOT EXISTS idx_bite_audio_triage_bite_lang
    ON bite_audio_triage (bite_id, language);

-- Filter by assignment
CREATE INDEX IF NOT EXISTS idx_bite_audio_triage_assignment
    ON bite_audio_triage (assignment_id)
    WHERE assignment_id IS NOT NULL;

-- Dashboard queries: filter by decision/status
CREATE INDEX IF NOT EXISTS idx_bite_audio_triage_decision
    ON bite_audio_triage (decision, status);

-- Admin review queue: latest non-expired pending reviews
CREATE INDEX IF NOT EXISTS idx_bite_audio_triage_pending_review
    ON bite_audio_triage (created_at DESC)
    WHERE admin_action IS NULL AND status = 'completed';

-- ── Auto-update updated_at ────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_bite_audio_triage_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_bite_audio_triage_updated_at ON bite_audio_triage;
CREATE TRIGGER trg_bite_audio_triage_updated_at
    BEFORE UPDATE ON bite_audio_triage
    FOR EACH ROW
    EXECUTE FUNCTION update_bite_audio_triage_updated_at();

-- ── RLS ───────────────────────────────────────────────────────────────────
-- fix-lab-server uses service role (bypasses RLS automatically).
-- Explicit policy lets admins query directly from the gallery if needed.

ALTER TABLE bite_audio_triage ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admins can view bite audio triage"
    ON bite_audio_triage FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE profiles.id = auth.uid()
            AND profiles.role = 'admin'
        )
    );

-- ── Comments ──────────────────────────────────────────────────────────────

COMMENT ON TABLE bite_audio_triage IS
    'AI triage analysis for bite voice-over audio issues. Each row = one triage run.';

COMMENT ON COLUMN bite_audio_triage.decision IS
    'AI recommendation: full (regen all paras), partial (regen some), skip (no VO issue), escalate (needs human)';

COMMENT ON COLUMN bite_audio_triage.segments_to_regen IS
    'JSONB array of paragraphs to regenerate with timing, issue description, and TTS hints';

COMMENT ON COLUMN bite_audio_triage.paragraph_timings IS
    'JSONB array mapping each source paragraph to its audio time range (from Whisper STT alignment)';

COMMENT ON COLUMN bite_audio_triage.status IS
    'completed = current latest; failed = errored; expired = superseded by newer run for same (bite, lang)';

COMMENT ON COLUMN bite_audio_triage.admin_action IS
    'Admin decision on the AI recommendation: approved → proceed to regen, rejected → ignore, modified → manual override';
