-- ============================================================================
-- Daily Reviewer Progress Report — schema
-- ============================================================================
-- Two tables + one trigger. The RPC that builds the report payload lives in
-- migrations/rpc_functions/get_reviewer_daily_report.sql.
--
-- This migration is idempotent.
-- ============================================================================

-- ─── Recipients ────────────────────────────────────────────────────────────
-- Who receives the daily email. Admin UI manages this.

CREATE TABLE IF NOT EXISTS report_recipients (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    name        TEXT,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_recipients_enabled
    ON report_recipients (enabled)
    WHERE enabled = TRUE;

-- Trigger to bump updated_at
CREATE OR REPLACE FUNCTION update_report_recipients_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_report_recipients_updated_at ON report_recipients;
CREATE TRIGGER trg_report_recipients_updated_at
    BEFORE UPDATE ON report_recipients
    FOR EACH ROW EXECUTE FUNCTION update_report_recipients_updated_at();


-- ─── Run history ───────────────────────────────────────────────────────────
-- Audit trail of past report runs (manual + cron).

CREATE TABLE IF NOT EXISTS report_runs (
    id                 UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    type               TEXT NOT NULL DEFAULT 'daily_summary'
                       CHECK (type IN ('daily_summary')),
    status             TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'sent', 'failed')),
    triggered_by       TEXT,                              -- 'cron' | 'manual'
    recipients_count   INT NOT NULL DEFAULT 0,
    error              TEXT,
    payload_summary    JSONB,                             -- small headline stats
    sent_at            TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_runs_created
    ON report_runs (created_at DESC);


-- ─── Seed initial recipients ──────────────────────────────────────────────
-- INSERT ... ON CONFLICT DO NOTHING so re-running the migration is safe.

INSERT INTO report_recipients (email, name, enabled) VALUES
    ('kr.rajesh117@gmail.com', 'Rajesh Kumar', TRUE),
    ('mightyr443@gmail.com',   'Mighty R',    TRUE)
ON CONFLICT (email) DO NOTHING;


COMMENT ON TABLE report_recipients IS
    'Who receives the daily Kitab progress report email. Admin UI manages this.';
COMMENT ON TABLE report_runs IS
    'Audit trail of report send attempts. Used by admin Reports tab history view.';
