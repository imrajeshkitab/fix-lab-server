"""
Daily Reviewer Progress Report
================================
Builds the report payload via Supabase RPC, renders an HTML email,
sends via Gmail SMTP. Idempotent — safe to re-run manually.

Public functions:
    build_report_payload(sb_rpc_fn) -> dict
    render_html(payload) -> str
    send_email(to_addresses, subject, html_body) -> None

Used by main.py endpoints:
    POST /api/reports/run-daily       — triggered by Supabase pg_cron at 10am IST
    GET  /api/reports/preview         — admin UI preview before sending
"""

import html as html_lib
import logging
import os
from email.message import EmailMessage
from typing import Callable, List, Optional

logger = logging.getLogger("fix-lab.reports")


# ────────────────────────────── Env / config ─────────────────────────────────

SMTP_HOST          = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT          = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER          = os.getenv("SMTP_USER")
SMTP_APP_PASSWORD  = os.getenv("SMTP_APP_PASSWORD")
REPORTS_FROM_NAME  = os.getenv("REPORTS_FROM_NAME", "Kitab RMS Bot")
REPORTS_TIMEZONE   = os.getenv("REPORTS_TIMEZONE", "Asia/Kolkata")


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
        "new_assignments_24h": delta.get("new_assignments_24h", 0),
        "net_backlog_change":  delta.get("net_backlog_change", 0),
    }


# ──────────────────────────── HTML rendering ─────────────────────────────────

_CSS = """
    body { font-family: -apple-system, system-ui, sans-serif; color: #1f2937; line-height: 1.5; max-width: 720px; margin: 24px auto; padding: 0 16px; }
    h1 { font-size: 18px; margin: 0 0 4px; color: #0f172a; }
    .subtitle { color: #6b7280; font-size: 13px; margin-bottom: 24px; }
    h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; margin: 28px 0 12px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 8px; }
    th { background: #f8fafc; text-align: left; padding: 8px 10px; font-weight: 600; color: #475569; border-bottom: 1px solid #e5e7eb; }
    td { padding: 8px 10px; border-bottom: 1px solid #f1f5f9; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr.total td { font-weight: 700; background: #f8fafc; border-top: 2px solid #cbd5e1; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-amber { background: #fef3c7; color: #92400e; }
    .pill-red { background: #fee2e2; color: #991b1b; }
    .pill-blue { background: #dbeafe; color: #1e40af; }
    ul { padding-left: 20px; margin: 8px 0; }
    .available-row { color: #166534; font-weight: 500; }
    .footer { color: #9ca3af; font-size: 11px; margin-top: 32px; padding-top: 16px; border-top: 1px solid #f1f5f9; }
    .delta-positive { color: #b91c1c; }
    .delta-negative { color: #15803d; }
    .delta-zero { color: #6b7280; }
    .spark { font-family: monospace; white-space: pre; font-size: 12px; line-height: 1.2; }
"""


def _esc(s) -> str:
    return html_lib.escape(str(s)) if s is not None else ""


def _fmt_lang(lang: str) -> str:
    return "English" if lang == "en" else ("Hindi" if lang == "hi" else lang.upper())


def _fmt_content_type(ct: str) -> str:
    return ct.title() if ct else "—"


def _render_progress_table(progress: list) -> str:
    """Section 1 — progress matrix for the last 24 hours."""
    if not progress:
        return "<p style='color:#6b7280'>No reviewer activity in the last 24 hours.</p>"

    # Group by (content_type, language) for stable ordering
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
          <th class="num">🔧 Sent for correction</th>
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
    new_a     = int(delta.get("new_assignments_24h", 0))
    net       = int(delta.get("net_backlog_change", 0))
    cls = "delta-positive" if net > 0 else ("delta-negative" if net < 0 else "delta-zero")
    arrow = "▲" if net > 0 else ("▼" if net < 0 else "▶")
    return f"""
    <p style="font-size:14px;">
      <strong>Backlog change:</strong>
      <span class="{cls}">{arrow} {net:+d}</span>
      &nbsp;&nbsp;
      ({completed} completed, {new_a} new assignments)
    </p>
    """


def _render_reviewers(reviewers: dict) -> str:
    if not reviewers:
        return "<p style='color:#6b7280'>No reviewer data.</p>"

    available = reviewers.get("available") or []
    active = reviewers.get("active") or []

    html_out = ""

    # Available
    if available:
        items = "".join(f"<li class='available-row'>🟢 {_esc(n)}</li>" for n in available)
        html_out += f"""
        <p><strong>Available reviewers</strong> (no pending work):</p>
        <ul>{items}</ul>
        """
    else:
        html_out += "<p style='color:#6b7280'>No reviewers are fully available right now.</p>"

    # Active
    if active:
        rows_html = ""
        for r in active:
            avg = r.get("avg_rating")
            avg_str = f"{float(avg):.1f}" if avg is not None else "—"
            rows_html += f"""
            <tr>
              <td>{_esc(r.get('name'))}</td>
              <td class="num">{_esc(r.get('done_24h', 0))}</td>
              <td class="num">{_esc(r.get('pending', 0))}</td>
              <td class="num">{avg_str}</td>
            </tr>"""
        html_out += f"""
        <p style="margin-top:16px;"><strong>Active reviewers</strong>:</p>
        <table>
          <thead>
            <tr>
              <th>Reviewer</th>
              <th class="num">Done (24h)</th>
              <th class="num">Pending</th>
              <th class="num">Avg ⭐ given</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """

    return html_out


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

    progress = payload.get("progress_24h") or []
    snapshot = payload.get("snapshot") or []
    delta = payload.get("delta") or {}
    reviewers = payload.get("reviewers") or {}
    throughput = payload.get("throughput_7d") or []

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{_CSS}</style></head><body>
  <h1>📊 Kitab Reviews — Daily Progress Report</h1>
  <div class="subtitle">{_esc(date_str)} · window: last 24 hours</div>

  <h2>1. Progress (last 24 hours)</h2>
  {_render_progress_table(progress)}
  {_render_delta(delta)}

  <h2>2. Backlog &amp; Status snapshot</h2>
  {_render_snapshot_table(snapshot)}

  <h2>3. Reviewer activity</h2>
  {_render_reviewers(reviewers)}

  <h2>4. Throughput — last 7 days</h2>
  {_render_throughput(throughput)}

  <div class="footer">
    Auto-generated by Kitab RMS · sent from {_esc(SMTP_USER or 'rajesh.kumar@kitab.com')}<br/>
    To stop receiving this, ask an admin to remove your address from the Reports tab in the admin dashboard.
  </div>
</body></html>"""


# ──────────────────────────── Email sending ──────────────────────────────────

async def send_email(
    to_addresses: List[str],
    subject: str,
    html_body: str,
) -> None:
    """
    Send via Gmail SMTP using aiosmtplib. Raises on failure so the worker can
    record it in report_runs.error.
    """
    if not to_addresses:
        raise RuntimeError("send_email called with empty recipient list")
    if not SMTP_USER or not SMTP_APP_PASSWORD:
        raise RuntimeError(
            "SMTP not configured — set SMTP_USER and SMTP_APP_PASSWORD in .env"
        )

    import aiosmtplib

    msg = EmailMessage()
    msg["From"] = f"{REPORTS_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = ", ".join(to_addresses)
    msg["Subject"] = subject
    # Plain-text fallback (most clients won't show this, but it's required for spam filters)
    msg.set_content(
        "This is the daily Kitab progress report. Your email client is showing the plain-text fallback; "
        "open in an HTML-capable client (Gmail, Outlook, Apple Mail) to see the full report."
    )
    msg.add_alternative(html_body, subtype="html")

    logger.info(
        f"Reports: sending to {len(to_addresses)} recipient(s) via {SMTP_HOST}:{SMTP_PORT}"
    )

    await aiosmtplib.send(
        msg,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USER,
        password=SMTP_APP_PASSWORD,
        start_tls=True,                  # Gmail requires STARTTLS on port 587
        timeout=30,
    )

    logger.info(f"Reports: email sent ✅ to {len(to_addresses)} recipient(s)")


def report_subject(payload: dict) -> str:
    """Build the email Subject line."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")
    delta = payload.get("delta") or {}
    completed = int(delta.get("completed_24h", 0))
    return f"Kitab Reviews — Daily Report ({today}) · {completed} reviews in 24h"
