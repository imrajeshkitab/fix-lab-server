"""
Daily Reviewer Progress Report
================================
Builds the report payload via Supabase RPC, renders an HTML email,
sends via Resend HTTPS API (Render free tier blocks outbound SMTP,
so we use a transactional service over port 443). Idempotent —
safe to re-run manually.

Public functions:
    build_report_payload(sb_rpc_fn) -> dict
    render_html(payload) -> str
    send_email(http_client, to_addresses, subject, html_body) -> None

Used by main.py endpoints:
    POST /api/reports/run-daily       — triggered by Supabase pg_cron at 10am IST
    GET  /api/reports/preview         — admin UI preview before sending
"""

import html as html_lib
import logging
import os
from typing import List

logger = logging.getLogger("fix-lab.reports")


# ────────────────────────────── Env / config ─────────────────────────────────
# Brevo (HTTPS, port 443 — works on Render free tier)

BREVO_API_KEY       = os.getenv("BREVO_API_KEY")
REPORTS_FROM_EMAIL  = os.getenv("REPORTS_FROM_EMAIL", "rajesh.kumar@kitab.com")
REPORTS_FROM_NAME   = os.getenv("REPORTS_FROM_NAME", "Kitab RMS Bot")
REPORTS_TIMEZONE    = os.getenv("REPORTS_TIMEZONE", "Asia/Kolkata")
BREVO_API_URL       = "https://api.brevo.com/v3/smtp/email"
SMTP_USER           = os.getenv("SMTP_USER")


# ─────────────────────────── Payload construction ────────────────────────────

async def build_report_payload(sb_rpc) -> dict:
    """Call get_reviewer_daily_report() RPC and return the JSON payload."""
    data = await sb_rpc("get_reviewer_daily_report", {"p_hours": 24})
    # PostgREST returns the JSONB as the response body directly when the
    # function returns a single JSONB value.
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    return data or {}


def headline_summary(payload: dict) -> dict:
    """Small summary kept on report_runs.payload_summary for the UI history list."""
    delta = payload.get("delta") or {}
    return {
        "completed_24h":       delta.get("completed_24h", 0),
        "corrected_24h":       delta.get("corrected_24h", 0),
        "rereview_24h":        delta.get("rereview_24h", 0),
        "reviewed_24h":        delta.get("reviewed_24h", 0),
        "new_assignments_24h": delta.get("new_assignments_24h", 0),
        "net_backlog_change":  delta.get("net_backlog_change", 0),
    }


# ──────────────────────────── HTML rendering ─────────────────────────────────

_CSS = """
    body { font-family: -apple-system, system-ui, sans-serif; color: #1f2937; line-height: 1.5; max-width: 760px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: 18px; margin: 0 0 4px; color: #0f172a; }
    .subtitle { color: #6b7280; font-size: 13px; margin-bottom: 24px; }
    h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; margin: 28px 0 12px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.4px; color: #64748b; margin: 16px 0 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 8px; }
    th { background: #f8fafc; text-align: left; padding: 8px 10px; font-weight: 600; color: #475569; border-bottom: 1px solid #e5e7eb; }
    td { padding: 8px 10px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    td.dim { color: #94a3b8; }
    tr.total td { font-weight: 700; background: #f8fafc; border-top: 2px solid #cbd5e1; }
    tr.inactive td { color: #94a3b8; background: #fafafa; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-amber { background: #fef3c7; color: #92400e; }
    .pill-red { background: #fee2e2; color: #991b1b; }
    .pill-blue { background: #dbeafe; color: #1e40af; }
    .pill-slate { background: #f1f5f9; color: #475569; }
    ul { padding-left: 20px; margin: 8px 0; }
    .available-row { color: #166534; font-weight: 500; }
    .footer { color: #9ca3af; font-size: 11px; margin-top: 32px; padding-top: 16px; border-top: 1px solid #f1f5f9; }
    .delta-positive { color: #b91c1c; }
    .delta-negative { color: #15803d; }
    .delta-zero { color: #6b7280; }
    .spark { font-family: monospace; white-space: pre; font-size: 12px; line-height: 1.2; }
    .kpi-row { font-size: 13px; margin: 4px 0 16px; }
    .kpi-row .pill { margin-right: 6px; }
    .bar-cell { padding: 4px 10px; }
    .bar-track { background:#f1f5f9; border-radius: 4px; height: 14px; width: 140px; position: relative; overflow: hidden; }
    .bar-fill { background: #3b82f6; height: 100%; border-radius: 4px; }
    .bar-fill.warn { background: #f59e0b; }
    .bar-fill.danger { background: #ef4444; }
    .bar-fill.muted { background: #cbd5e1; }
    .twocol { width: 100%; }
    .twocol td { vertical-align: top; padding: 0; border: none; }
    .twocol td.left { padding-right: 12px; }
    .twocol td.right { padding-left: 12px; }
"""


def _esc(s) -> str:
    return html_lib.escape(str(s)) if s is not None else ""


def _fmt_lang(lang: str) -> str:
    return "English" if lang == "en" else ("Hindi" if lang == "hi" else lang.upper())


def _fmt_content_type(ct: str) -> str:
    return ct.title() if ct else "—"


def _render_progress_table(progress: list, window_label: str = "last 24 hours") -> str:
    """Progress matrix for a given window (24h or 7d)."""
    if not progress:
        return f"<p style='color:#6b7280'>No reviewer activity in the {window_label}.</p>"

    rows = sorted(progress, key=lambda r: (r.get("content_type", ""), r.get("language", "")))
    total_approved = sum(r.get("approved", 0) for r in rows)
    total_corrected = sum(r.get("corrected", 0) for r in rows)

    body = """
    <table>
      <thead>
        <tr>
          <th>Content</th>
          <th>Language</th>
          <th class="num">✅ Approved</th>
          <th class="num">🔧 Corrected</th>
          <th class="num">Total moved</th>
        </tr>
      </thead>
      <tbody>
    """
    for r in rows:
        body += f"""
        <tr>
          <td>{_esc(_fmt_content_type(r.get('content_type')))}</td>
          <td>{_esc(_fmt_lang(r.get('language')))}</td>
          <td class="num">{_esc(r.get('approved', 0))}</td>
          <td class="num">{_esc(r.get('corrected', 0))}</td>
          <td class="num">{_esc(r.get('total', 0))}</td>
        </tr>"""
    body += f"""
        <tr class="total">
          <td colspan="2">Total</td>
          <td class="num">{total_approved}</td>
          <td class="num">{total_corrected}</td>
          <td class="num">{total_approved + total_corrected}</td>
        </tr>
      </tbody>
    </table>
    """
    return body


def _render_progress_side_by_side(progress_24h: list, progress_7d: list) -> str:
    """Two progress tables side-by-side: 24h vs 7d."""
    return f"""
    <table class="twocol"><tr>
      <td class="left">
        <h3>Last 24 hours</h3>
        {_render_progress_table(progress_24h, "last 24 hours")}
      </td>
      <td class="right">
        <h3>Last 7 days</h3>
        {_render_progress_table(progress_7d, "last 7 days")}
      </td>
    </tr></table>
    """


def _render_snapshot_table(snapshot: list) -> str:
    """Section 2 — full backlog & status snapshot."""
    if not snapshot:
        return "<p style='color:#6b7280'>No data.</p>"

    rows = sorted(snapshot, key=lambda r: (r.get("content_type", ""), r.get("language", "")))
    tot = {
        "total_on_rms": 0, "assigned": 0, "approved": 0,
        "sent_for_correction": 0, "under_rereview": 0,
    }

    body = """
    <table>
      <thead>
        <tr>
          <th>Content</th>
          <th>Language</th>
          <th class="num">Total on RMS</th>
          <th class="num">Assigned</th>
          <th class="num">✅ Approved</th>
          <th class="num">🔧 Sent for correction</th>
          <th class="num">🔁 Under re-review</th>
        </tr>
      </thead>
      <tbody>
    """
    for r in rows:
        for k in tot:
            tot[k] += int(r.get(k) or 0)
        body += f"""
        <tr>
          <td>{_esc(_fmt_content_type(r.get('content_type')))}</td>
          <td>{_esc(_fmt_lang(r.get('language')))}</td>
          <td class="num">{_esc(r.get('total_on_rms', 0))}</td>
          <td class="num">{_esc(r.get('assigned', 0))}</td>
          <td class="num">{_esc(r.get('approved', 0))}</td>
          <td class="num">{_esc(r.get('sent_for_correction', 0))}</td>
          <td class="num">{_esc(r.get('under_rereview', 0))}</td>
        </tr>"""
    body += f"""
        <tr class="total">
          <td colspan="2">Total</td>
          <td class="num">{tot['total_on_rms']}</td>
          <td class="num">{tot['assigned']}</td>
          <td class="num">{tot['approved']}</td>
          <td class="num">{tot['sent_for_correction']}</td>
          <td class="num">{tot['under_rereview']}</td>
        </tr>
      </tbody>
    </table>
    """
    return body


def _render_delta(delta: dict) -> str:
    if not delta:
        return ""
    completed = int(delta.get("completed_24h", 0))
    corrected = int(delta.get("corrected_24h", 0))
    rereview  = int(delta.get("rereview_24h", 0))
    reviewed  = int(delta.get("reviewed_24h", 0)) or (completed + corrected)
    new_a     = int(delta.get("new_assignments_24h", 0))
    net       = int(delta.get("net_backlog_change", 0))
    cls = "delta-positive" if net > 0 else ("delta-negative" if net < 0 else "delta-zero")
    arrow = "▲" if net > 0 else ("▼" if net < 0 else "▶")
    return f"""
    <p style="font-size:14px; margin-bottom: 4px;">
      <strong>Backlog change:</strong>
      <span class="{cls}">{arrow} {net:+d}</span>
    </p>
    <ul style="font-size: 13px; color: #475569; margin-top: 4px; padding-left: 18px;">
      <li><strong>{reviewed}</strong> reviewed
          &nbsp;<span style="color:#6b7280;">({completed} approved + {corrected} sent for correction)</span>
      </li>
      <li><strong>{rereview}</strong> sent for re-review</li>
      <li><strong>{new_a}</strong> new assignments</li>
    </ul>
    """


def _bar(value: int, max_value: int, kind: str = "") -> str:
    """Inline CSS horizontal bar (email-safe). kind ∈ {'', 'warn', 'danger', 'muted'}."""
    if max_value <= 0:
        pct = 0
    else:
        pct = min(100, int(round((value / max_value) * 100)))
    cls = f"bar-fill {kind}".strip()
    return (
        f'<div class="bar-track">'
        f'<div class="{cls}" style="width:{pct}%"></div>'
        f'</div>'
    )


def _render_activity_kpis(reviewers: dict) -> str:
    """Compact KPI line: active 24h vs 7d counts."""
    a24 = reviewers.get("active_24h") or {}
    a7 = reviewers.get("active_7d") or {}
    available = reviewers.get("available") or []
    return f"""
    <div class="kpi-row">
      <span class="pill pill-green">Active 24h · {int(a24.get('count') or 0)}</span>
      <span class="pill pill-blue">Active 7d · {int(a7.get('count') or 0)}</span>
      <span class="pill pill-slate">Available · {len(available)}</span>
    </div>
    """


def _render_near_finish(near_finish: list) -> str:
    """Reviewers about to wrap up (pending ≤ 5 AND active in last 7d)."""
    if not near_finish:
        return "<p style='color:#6b7280'>No reviewers are near finish right now.</p>"
    items = "".join(
        f"<li><strong>{_esc(r.get('name'))}</strong> — "
        f"<span class='pill pill-amber'>{int(r.get('pending') or 0)} left</span> "
        f"<span style='color:#94a3b8;font-size:12px;'>· {int(r.get('done_7d') or 0)} done in 7d</span></li>"
        for r in near_finish
    )
    return f"""
    <p style="font-size:13px;color:#475569;margin:0 0 6px;">
      Reviewers with ≤5 pending items and recent activity — consider assigning a fresh batch.
    </p>
    <ul>{items}</ul>
    """


def _render_leaderboard(leaderboard: list) -> str:
    """Per-reviewer table with HTML/CSS Done-7d bars."""
    if not leaderboard:
        return "<p style='color:#6b7280'>No reviewer activity to show.</p>"

    max_done_7d = max((int(r.get("done_7d") or 0) for r in leaderboard), default=0) or 1

    rows_html = ""
    for r in leaderboard:
        done_24h = int(r.get("done_24h") or 0)
        done_7d = int(r.get("done_7d") or 0)
        corrected_7d = int(r.get("corrected_7d") or 0)
        pending = int(r.get("pending") or 0)
        is_active_7d = bool(r.get("active_7d"))
        row_cls = "" if is_active_7d else "inactive"
        # Pending pill colour: amber if pending>0, slate otherwise
        pending_html = (
            f'<span class="pill pill-amber">{pending}</span>' if pending > 0
            else '<span class="pill pill-slate">0</span>'
        )
        bar_kind = "" if is_active_7d else "muted"
        rows_html += f"""
        <tr class="{row_cls}">
          <td>{_esc(r.get('name'))}</td>
          <td class="num">{done_24h}</td>
          <td class="num">{done_7d}</td>
          <td class="bar-cell">{_bar(done_7d, max_done_7d, bar_kind)}</td>
          <td class="num">{corrected_7d}</td>
          <td class="num">{pending_html}</td>
        </tr>"""

    return f"""
    <table>
      <thead>
        <tr>
          <th>Reviewer</th>
          <th class="num">Done 24h</th>
          <th class="num">Done 7d</th>
          <th>&nbsp;</th>
          <th class="num">🔧 Corrected 7d</th>
          <th class="num">Pending</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="font-size:11px;color:#94a3b8;margin-top:4px;">
      Rows in grey have no activity in the last 7 days (still hold pending items).
    </p>
    """


def _render_available(reviewers: dict) -> str:
    available = reviewers.get("available") or []
    if not available:
        return "<p style='color:#6b7280'>No reviewers are fully available right now.</p>"
    items = "".join(f"<li class='available-row'>🟢 {_esc(n)}</li>" for n in available)
    return f"<ul>{items}</ul>"


def _render_throughput(throughput: list) -> str:
    if not throughput:
        return ""
    max_val = max(int(d.get("completed") or 0) for d in throughput) or 1
    lines = []
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    # Parse dates manually to avoid datetime import overhead
    for d in throughput:
        date_str = d.get("date", "")
        c = int(d.get("completed") or 0)
        bar = "█" * max(1, int((c / max_val) * 24)) if c > 0 else " "
        # Get weekday: use datetime here, simple
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            wday = weekday_map.get(dt.weekday(), "?")
        except Exception:
            wday = "?"
        lines.append(f"  {wday} {bar} {c}")
    return f"""
    <div class="spark">{"<br/>".join(_esc(line) for line in lines)}</div>
    """


def render_html(payload: dict) -> str:
    """Produce the full HTML email body for the report."""
    from datetime import datetime, timezone, timedelta
    # IST = UTC+5:30 (good enough for display, even if REPORTS_TIMEZONE is something else)
    ist = timezone(timedelta(hours=5, minutes=30))
    now_local = datetime.now(ist)
    date_str = now_local.strftime("%a, %d %b %Y · %H:%M IST")

    progress_24h = payload.get("progress_24h") or []
    progress_7d = payload.get("progress_7d") or []
    snapshot = payload.get("snapshot") or []
    delta = payload.get("delta") or {}
    reviewers = payload.get("reviewers") or {}
    leaderboard = reviewers.get("leaderboard") or []
    near_finish = reviewers.get("near_finish") or []
    throughput = payload.get("throughput_7d") or []

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_CSS}</style></head><body>
  <h1>📊 Kitab Reviews — Daily Progress Report</h1>
  <div class="subtitle">{_esc(date_str)} · window: last 24 hours</div>

  <h2>1. Movement summary</h2>
  {_render_progress_side_by_side(progress_24h, progress_7d)}
  {_render_delta(delta)}

  <h2>2. Backlog &amp; Status snapshot</h2>
  {_render_snapshot_table(snapshot)}

  <h2>3. Reviewer activity</h2>
  {_render_activity_kpis(reviewers)}

  <h3>Near finish (≤ 5 pending)</h3>
  {_render_near_finish(near_finish)}

  <h3>Reviewer leaderboard</h3>
  {_render_leaderboard(leaderboard)}

  <h3>Available reviewers (no pending work)</h3>
  {_render_available(reviewers)}

  <h2>4. Throughput — last 7 days</h2>
  {_render_throughput(throughput)}

  <div class="footer">
    Auto-generated by Kitab RMS · sent from {_esc(SMTP_USER or 'rajesh.kumar@kitab.com')}<br/>
    To stop receiving this, ask an admin to remove your address from the Reports tab in the admin dashboard.
  </div>
</body></html>"""


# ──────────────────────────── Email sending ──────────────────────────────────

async def send_email(
    http_client,
    to_addresses: List[str],
    subject: str,
    html_body: str,
) -> None:
    """
    Send via Brevo HTTPS API. Raises on failure so the worker can record
    it in report_runs.error.
    """
    if not to_addresses:
        raise RuntimeError("send_email called with empty recipient list")
    if not BREVO_API_KEY:
        raise RuntimeError("BREVO_API_KEY not set — configure in .env / Render env vars")

    payload = {
        "sender": {
            "name":  REPORTS_FROM_NAME,
            "email": REPORTS_FROM_EMAIL,
        },
        "to":      [{"email": email} for email in to_addresses],
        "subject": subject,
        "htmlContent": html_body,
    }

    headers = {
        "api-key":      BREVO_API_KEY,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    logger.info(
        f"Reports: POST to Brevo for {len(to_addresses)} recipient(s) "
        f"from <{REPORTS_FROM_EMAIL}>"
    )

    r = await http_client.post(BREVO_API_URL, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201, 202):
        # Surface the response body so report_runs.error is actionable
        body = (r.text or "")[:500]
        raise RuntimeError(f"Brevo API HTTP {r.status_code}: {body}")

    # Brevo returns { "messageId": "..." } on success
    result = r.json() if r.content else {}
    logger.info(
        f"Reports: email sent ✅ to {len(to_addresses)} recipient(s) "
        f"(message_id={result.get('messageId', '?')})"
    )


def report_subject(payload: dict) -> str:
    """Build the email Subject line."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")
    delta = payload.get("delta") or {}
    completed = int(delta.get("completed_24h", 0))
    return f"Kitab Reviews — Daily Report ({today}) · {completed} reviews in 24h"
