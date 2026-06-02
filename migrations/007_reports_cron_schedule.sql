-- ============================================================================
-- pg_cron schedule for daily report — run ONCE in Supabase SQL Editor
-- ============================================================================
-- Prerequisites:
--   1. Enable `pg_cron` extension     (Dashboard → Database → Extensions)
--   2. Enable `pg_net` extension      (same place)
--   3. Apply migration 006_reports.sql first (creates the tables)
--
-- Before running this file:
--   - Replace <FIX_LAB_SERVER_URL> with the deployed Render URL
--     e.g. 'https://fix-lab-server-1.onrender.com'
--   - The webhook secret is read from a DB-level setting (set below) so it
--     never appears in plain text in the cron job definition.
--
-- To inspect or change the cron job later:
--   SELECT * FROM cron.job;                                     -- list jobs
--   SELECT cron.unschedule('kitab-daily-report');               -- remove
--   SELECT cron.schedule(...);                                  -- recreate
-- ============================================================================

-- 1. Store the webhook secret at the database level. Visible only to
--    superusers / db owner, not in cron.job definitions.
ALTER DATABASE postgres SET app.reports_webhook_secret
    TO '_lB3F6p88TJbWyOvn7OX1_0ct-bi7ndsgVbSchMO_F4';

-- 2. Store the fix-lab-server URL similarly so we don't hardcode it.
--    👉 EDIT THIS to your actual Render URL before running 👈
ALTER DATABASE postgres SET app.fix_lab_server_url
    TO 'https://fix-lab-server-1.onrender.com';

-- 3. Drop any prior copy of the job (idempotent re-runs of this file).
SELECT cron.unschedule('kitab-daily-report')
WHERE EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-daily-report');

-- 4. Schedule: 04:30 UTC = 10:00 IST, every day.
--    cron expression: 'minute hour day-of-month month day-of-week'
SELECT cron.schedule(
    'kitab-daily-report',
    '30 4 * * *',
    $$
    SELECT net.http_post(
        url     := current_setting('app.fix_lab_server_url') || '/api/reports/run-daily',
        headers := jsonb_build_object(
            'Content-Type',     'application/json',
            'x-reports-secret', current_setting('app.reports_webhook_secret')
        ),
        body    := '{}'::jsonb
    );
    $$
);

-- 5. Verify
SELECT jobname, schedule, active FROM cron.job WHERE jobname = 'kitab-daily-report';

-- ─── Manual trigger (handy for testing without waiting for 10am) ─────────────
-- Uncomment + run this block to fire the report right now:
--
-- SELECT net.http_post(
--     url     := current_setting('app.fix_lab_server_url') || '/api/reports/run-daily',
--     headers := jsonb_build_object(
--         'Content-Type', 'application/json',
--         'x-reports-secret', current_setting('app.reports_webhook_secret')
--     ),
--     body    := '{}'::jsonb
-- );
