-- ============================================================================
-- Reports cron schedule — admin-editable from the UI
-- ============================================================================
-- Two wrapper RPCs (SECURITY DEFINER) so the backend can read + update the
-- pg_cron job for the daily report without needing direct cron-schema
-- privileges from the service role.
--
-- The webhook URL and secret stay hard-coded inside the wrapper — admins
-- only change the time via these endpoints. If the URL or secret ever
-- need to rotate, re-run this migration with the new values.
--
-- Run this in Supabase SQL Editor ONCE after deploying the backend changes.
-- 👉 BEFORE RUNNING 👈
--   Edit the two literals in update_report_cron_schedule() below if your
--   fix-lab-server URL or REPORTS_WEBHOOK_SECRET differ from the defaults.
-- ============================================================================

-- ─── get_report_cron_schedule ────────────────────────────────────────────────
-- Returns the current cron expression for the daily report job.
-- NULL if the job doesn't exist yet (run migration 007 first).

DROP FUNCTION IF EXISTS get_report_cron_schedule();

CREATE OR REPLACE FUNCTION get_report_cron_schedule()
RETURNS JSONB AS $$
DECLARE
    v_row cron.job%ROWTYPE;
BEGIN
    SELECT * INTO v_row FROM cron.job WHERE jobname = 'kitab-daily-report' LIMIT 1;
    IF NOT FOUND THEN
        RETURN jsonb_build_object('exists', false);
    END IF;
    RETURN jsonb_build_object(
        'exists',   true,
        'jobname',  v_row.jobname,
        'schedule', v_row.schedule,
        'active',   v_row.active
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION get_report_cron_schedule() TO authenticated, service_role;


-- ─── update_report_cron_schedule ─────────────────────────────────────────────
-- Replaces the existing job with a new schedule. Caller passes the cron
-- expression directly (e.g. '30 4 * * *' for 04:30 UTC = 10:00 IST).
-- Returns the new schedule on success.

DROP FUNCTION IF EXISTS update_report_cron_schedule(TEXT);

CREATE OR REPLACE FUNCTION update_report_cron_schedule(p_cron TEXT)
RETURNS JSONB AS $$
DECLARE
    v_url     TEXT := 'https://fix-lab-server-1.onrender.com/api/reports/run-daily';
    v_secret  TEXT := '_lB3F6p88TJbWyOvn7OX1_0ct-bi7ndsgVbSchMO_F4';
    v_command TEXT;
BEGIN
    -- Build the cron command body. Inline values so the job is self-contained.
    v_command := format($job$
        SELECT net.http_post(
            url     := %L,
            headers := jsonb_build_object(
                'Content-Type',     'application/json',
                'x-reports-secret', %L
            ),
            body    := '{}'::jsonb
        );
    $job$, v_url, v_secret);

    -- Drop any prior copy (idempotent re-runs).
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-daily-report') THEN
        PERFORM cron.unschedule('kitab-daily-report');
    END IF;

    -- Schedule the new job. cron.schedule raises on invalid expressions.
    PERFORM cron.schedule('kitab-daily-report', p_cron, v_command);

    RETURN jsonb_build_object(
        'exists',   true,
        'jobname',  'kitab-daily-report',
        'schedule', p_cron,
        'active',   true
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION update_report_cron_schedule(TEXT) TO authenticated, service_role;


-- ─── Sanity check ───────────────────────────────────────────────────────────
-- After running, you can verify with:
--   SELECT get_report_cron_schedule();
--   SELECT * FROM cron.job WHERE jobname = 'kitab-daily-report';

COMMENT ON FUNCTION get_report_cron_schedule IS
    'Returns current pg_cron schedule for the daily report job.';
COMMENT ON FUNCTION update_report_cron_schedule IS
    'Updates the pg_cron schedule for the daily report. Caller provides a cron expression.';
