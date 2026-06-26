-- ============================================================================
-- pg_cron schedule for daily report — run ONCE in Supabase SQL Editor
-- ============================================================================
-- Prerequisites:
--   1. Enable `pg_cron` extension     (Dashboard → Database → Extensions)
--   2. Enable `pg_net` extension      (same place)
--   3. Apply migration 006_reports.sql first (creates the tables)
--
-- 👉 BEFORE RUNNING THIS FILE 👈
-- Replace the two placeholders below with your actual values:
--   <FIX_LAB_SERVER_URL>   → e.g.  https://fix-lab-server-1.onrender.com
--   <REPORTS_WEBHOOK_SECRET> → the long random string from your .env
--                             (must match REPORTS_WEBHOOK_SECRET on Render)
--
-- Why inline instead of ALTER DATABASE?
--   Supabase's SQL Editor user is not a superuser, so `ALTER DATABASE ... SET`
--   is blocked with "permission denied to set parameter". Inlining the values
--   into cron.schedule's command is the standard workaround. The cron.job
--   table is only readable by privileged roles (service_role / postgres) so
--   anon/authenticated users can't see the secret.
--
-- To inspect or change the cron job later:
--   SELECT * FROM cron.job;                              -- list jobs
--   SELECT cron.unschedule('kitab-daily-report');        -- remove
--   SELECT cron.schedule(...);                           -- recreate
-- ============================================================================

-- 1. Remove any prior copy of the job (idempotent re-runs).
SELECT cron.unschedule('kitab-daily-report')
WHERE EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-daily-report');

-- 2. Schedule: 04:30 UTC = 10:00 IST, every day.
--    cron expression: 'minute hour day-of-month month day-of-week'
SELECT cron.schedule(
    'kitab-daily-report',
    '30 4 * * *',
    $$
    SELECT net.http_post(
        url     := 'https://fix-lab-server-1.onrender.com/api/reports/run-daily',
        headers := jsonb_build_object(
            'Content-Type',     'application/json',
            'x-reports-secret', '_lB3F6p88TJbWyOvn7OX1_0ct-bi7ndsgVbSchMO_F4'
        ),
        body    := '{}'::jsonb
    );
    $$
);

-- 3. Verify
SELECT jobname, schedule, active FROM cron.job WHERE jobname = 'kitab-daily-report';


-- ─── Manual trigger (handy for testing without waiting for 10am) ─────────────
-- Uncomment + run this block to fire the report right now:
--
-- SELECT net.http_post(
--     url     := 'https://fix-lab-server-1.onrender.com/api/reports/run-daily',
--     headers := jsonb_build_object(
--         'Content-Type', 'application/json',
--         'x-reports-secret', '_lB3F6p88TJbWyOvn7OX1_0ct-bi7ndsgVbSchMO_F4'
--     ),
--     body    := '{}'::jsonb
-- );
