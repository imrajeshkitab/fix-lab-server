-- ============================================================================
-- Render warmer — kicks the sleeping fix-lab-server container awake
-- ============================================================================
-- Render free tier suspends idle services after ~50s. First request after
-- sleep takes 30-60s to cold-start. Without a warmer, the daily report
-- cron at 04:30 UTC would hit a cold container; pg_net's default timeout
-- (5s) gives up before Render wakes, even though the request still
-- completes server-side. Result: confusing "timeout" in net._http_response
-- even though the email actually landed.
--
-- This migration adds a second cron that runs 5 minutes BEFORE the report:
--   04:25 UTC → warm_fix_lab_server() loops 10×, pinging GET / with a
--              30-second pg_sleep between attempts. Total runtime ~5 min.
--              First ping wakes the container; rest just keep it warm.
--   04:30 UTC → report cron fires against an already-warm server.
--
-- The warmer schedule is AUTO-COUPLED to the report schedule. When an
-- admin updates the report time via the UI, update_report_cron_schedule()
-- re-schedules the warmer to fire 5 min earlier. Two crons stay in sync
-- without manual intervention.
--
-- Idempotent — safe to re-run.
-- ============================================================================

-- ─── 1. Warmer function ─────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION warm_fix_lab_server()
RETURNS VOID AS $$
DECLARE
    v_url        TEXT := 'https://fix-lab-server-1.onrender.com/';
    v_iterations INT  := 10;       -- 10 pings × 30s ≈ 5 minutes total
    v_sleep_sec  INT  := 30;
    i            INT;
BEGIN
    FOR i IN 1..v_iterations LOOP
        BEGIN
            -- Fire-and-forget. We don't care about per-call success here;
            -- the first ping that lands starts Render's wake-up sequence.
            PERFORM net.http_post(
                url     := v_url,
                headers := jsonb_build_object('Content-Type', 'application/json'),
                body    := '{}'::jsonb,
                timeout_milliseconds := 20000
            );
        EXCEPTION WHEN OTHERS THEN
            -- Swallow per-iteration errors; keep pinging.
            NULL;
        END;
        -- Don't sleep after the final iteration
        IF i < v_iterations THEN
            PERFORM pg_sleep(v_sleep_sec);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION warm_fix_lab_server() TO authenticated, service_role;


-- ─── 2. Replace update_report_cron_schedule() to manage warmer too ──────────
-- The new version:
--   • Schedules the report job (same as before)
--   • If expression is a simple daily ('M H * * *'), schedules a warmer
--     cron 5 minutes earlier. Wraps to previous hour / 23:55 if needed.
--   • If expression is non-daily (e.g. '0 */6 * * *'), unschedules the
--     warmer — its single-time semantics don't fit multi-fire schedules.

DROP FUNCTION IF EXISTS update_report_cron_schedule(TEXT);

CREATE OR REPLACE FUNCTION update_report_cron_schedule(p_cron TEXT)
RETURNS JSONB AS $$
DECLARE
    v_url       TEXT := 'https://fix-lab-server-1.onrender.com/api/reports/run-daily';
    v_secret    TEXT := '_lB3F6p88TJbWyOvn7OX1_0ct-bi7ndsgVbSchMO_F4';
    v_command   TEXT;
    v_parts     TEXT[];
    v_min       INT;
    v_hour      INT;
    v_warm_min  INT;
    v_warm_hour INT;
    v_warm_cron TEXT;
    v_warm_scheduled BOOLEAN := false;
BEGIN
    -- ── 2a. Schedule the report ──
    v_command := format($job$
        SELECT net.http_post(
            url     := %L,
            headers := jsonb_build_object(
                'Content-Type',     'application/json',
                'x-reports-secret', %L
            ),
            body    := '{}'::jsonb,
            timeout_milliseconds := 90000
        );
    $job$, v_url, v_secret);

    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-daily-report') THEN
        PERFORM cron.unschedule('kitab-daily-report');
    END IF;
    PERFORM cron.schedule('kitab-daily-report', p_cron, v_command);

    -- ── 2b. Schedule the warmer iff daily expression ──
    v_parts := regexp_split_to_array(trim(p_cron), '\s+');

    IF array_length(v_parts, 1) = 5
       AND v_parts[3] = '*'
       AND v_parts[4] = '*'
       AND v_parts[5] = '*'
       AND v_parts[1] ~ '^\d+$'
       AND v_parts[2] ~ '^\d+$'
    THEN
        v_min  := v_parts[1]::INT;
        v_hour := v_parts[2]::INT;
        -- warmer = report time − 5 minutes
        IF v_min >= 5 THEN
            v_warm_min  := v_min - 5;
            v_warm_hour := v_hour;
        ELSE
            v_warm_min  := v_min + 55;        -- wraps to previous hour
            v_warm_hour := (v_hour + 23) % 24;
        END IF;
        v_warm_cron := v_warm_min || ' ' || v_warm_hour || ' * * *';

        IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-warm-render') THEN
            PERFORM cron.unschedule('kitab-warm-render');
        END IF;
        PERFORM cron.schedule(
            'kitab-warm-render',
            v_warm_cron,
            $w$ SELECT warm_fix_lab_server(); $w$
        );
        v_warm_scheduled := true;
    ELSE
        -- Non-daily — warmer semantics don't fit. Drop any prior warmer.
        IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'kitab-warm-render') THEN
            PERFORM cron.unschedule('kitab-warm-render');
        END IF;
    END IF;

    RETURN jsonb_build_object(
        'exists',          true,
        'jobname',         'kitab-daily-report',
        'schedule',        p_cron,
        'active',          true,
        'warmer_scheduled', v_warm_scheduled,
        'warmer_schedule', CASE WHEN v_warm_scheduled THEN v_warm_cron ELSE NULL END
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION update_report_cron_schedule(TEXT) TO authenticated, service_role;


-- ─── 3. Backfill: re-run update_report_cron_schedule for the existing job ───
-- Re-invoking the function on the current schedule will (a) recreate the
-- report job with the new 90s timeout baked in, and (b) schedule the
-- warmer for the first time.

DO $$
DECLARE
    v_cur TEXT;
BEGIN
    SELECT schedule INTO v_cur FROM cron.job WHERE jobname = 'kitab-daily-report' LIMIT 1;
    IF v_cur IS NOT NULL THEN
        PERFORM update_report_cron_schedule(v_cur);
        RAISE NOTICE 'Backfilled report (%) and warmer schedule', v_cur;
    ELSE
        RAISE NOTICE 'No existing report cron found — skipping backfill';
    END IF;
END $$;


-- ─── 4. Sanity check ────────────────────────────────────────────────────────
-- After running, verify with:
--   SELECT jobname, schedule, active FROM cron.job
--   WHERE jobname IN ('kitab-daily-report', 'kitab-warm-render') ORDER BY schedule;
-- Expected for 04:30 UTC report:
--   kitab-warm-render   | 25 4 * * *   | t
--   kitab-daily-report  | 30 4 * * *   | t


COMMENT ON FUNCTION warm_fix_lab_server IS
    'Fires 10 fire-and-forget pings to fix-lab-server with 30s gaps between. '
    'Used by the kitab-warm-render cron 5 min before the daily report to cold-start Render.';
