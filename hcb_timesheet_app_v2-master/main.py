# app.py
import streamlit as st
import pandas as pd
import numpy as np
import bisect
import datetime
import math
import tempfile
import os
import base64
from io import StringIO

# PDF / styling
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER

# =========================
# Font setup (graceful fallback)
# =========================
try:
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    pdfmetrics.registerFont(TTFont("SourceSansPro", os.path.join(font_dir, "SourceSansPro-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSansPro-Bold", os.path.join(font_dir, "SourceSansPro-Bold.ttf")))
    registerFontFamily(
        "SourceSansPro",
        normal="SourceSansPro",
        bold="SourceSansPro-Bold",
        italic="SourceSansPro",
        boldItalic="SourceSansPro-Bold",
    )
    PDF_FONT = "SourceSansPro"
    PDF_FONT_BOLD = "SourceSansPro-Bold"
except Exception:
    st.warning("Source Sans Pro fonts not found. Using default PDF fonts.")
    PDF_FONT = "Helvetica"
    PDF_FONT_BOLD = "Helvetica-Bold"

# =========================
# Helpers
# =========================
r_q_h = lambda h: math.ceil(float(h) * 4) / 4 if pd.notnull(h) and h != "" else 0.0
get_mon = lambda d: d - datetime.timedelta(days=d.weekday())
find_weeks = lambda dates: sorted(list(set(get_mon(d) for d in dates)))

def _norm_client(x: str) -> str:
    return (x or "").strip().casefold()

def round_to_quarter(dt: datetime.datetime) -> datetime.datetime:
    """
    Round a datetime to the nearest 15 minutes.
    Half-quarters (7.5 min) round up.
    """
    if pd.isna(dt):
        return dt
    base = dt.replace(second=0, microsecond=0, minute=0)
    mins = dt.minute + dt.second / 60.0
    q = int(round(mins / 15.0))  # 0..4
    return base + datetime.timedelta(minutes=15 * q)


# =========================
# Data processing
# =========================
def dedup_files(files):
    seen = set()
    return [f for f in files if (f.name, f.size) not in seen and not seen.add((f.name, f.size))]

def load_clean(f):
    c = f.read().decode("utf-8")
    try:
        h_row = next(i for i, l in enumerate(c.splitlines()) if "START_DATE*" in l)
    except StopIteration:
        st.error(f"Header not found in {f.name}.")
        st.stop()
    df = pd.read_csv(StringIO(c), skiprows=h_row)
    df = df.loc[
        df["START_DATE*"].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}")
    ].copy()
    drops = [col for col in df.columns if any(p in col for p in ["CATEGORY", "RATE", "MILES"])]
    return df.drop(columns=drops, errors="ignore")

def parse_ts(df):
    return df.assign(
        start=pd.to_datetime(df["START_DATE*"], format="%m/%d/%Y %H:%M"),
        end=pd.to_datetime(df["END_DATE*"], format="%m/%d/%Y %H:%M"),
    )

def extract_sites(df):
    o = df["START*"].astype(str).str.rsplit("|", n=1, expand=True)
    d = df["STOP*"].astype(str).str.rsplit("|", n=1, expand=True)
    df = df.assign(on=o[0].str.strip(), ot=o[1].str.strip(), dn=d[0].str.strip(), dt=d[1].str.strip())
    df = df.assign(
        origin=np.where(df["ot"] == "Homeowner", df["on"], df["ot"]),
        destin=np.where(df["dt"] == "Homeowner", df["dn"], df["dt"]),
    )
    return df.sort_values("start").reset_index(drop=True)

def clamp(df):
    df = df.assign(date=df["start"].dt.normalize())
    starts = df.loc[df["ot"].eq("Homeowner") | df["origin"].eq("Office")].groupby("date")["start"].min()
    ends = df.loc[df["dt"].eq("Homeowner") | df["destin"].eq("Office")].groupby("date")["end"].max()
    df = df.join(starts.rename("day_start"), on="date").join(ends.rename("day_end"), on="date")
    df = df.assign(cs=df[["start", "day_start"]].max(axis=1), ce=df[["end", "day_end"]].min(axis=1))
    df = df[df["ce"] > df["cs"]].copy()
    return df.assign(dur_hr=(df["ce"] - df["cs"]).dt.total_seconds() / 3600)

def build_segs(df):
    """
    Build clamped event segments + same-day gaps between events.
    NEW: For each day, if the first clamped segment starts AFTER 08:00,
    insert a leading gap [08:00 -> first_cs] and mark it to force owner='Other'.
    """
    if df.empty:
        return []

    # Ensure sorted and group by calendar day of the clamped start
    df = df.sort_values("cs").copy()
    df["_day"] = df["cs"].dt.date

    segs = []
    for day, g in df.groupby("_day"):
        recs = list(g.to_dict("records"))
        if not recs:
            continue

        # --- NEW: pre-8am → first drive gap (forced 'Other') ---
        eight = datetime.datetime.combine(day, datetime.time(8, 0))
        first_cs = recs[0]["cs"]
        if first_cs > eight:
            segs.append({
                "s": eight,
                "e": first_cs,
                "dur": (first_cs - eight).total_seconds() / 3600.0,
                "ot": None, "dt": None, "on": None, "dn": None,
                "force_other": True   # <— flag picked up by allocator
            })

        # --- Normal segments + in-day gaps ---
        for p, c in zip(recs, recs[1:]):
            # event segment
            segs.append({
                "s": p["cs"], "e": p["ce"], "dur": p["dur_hr"],
                "ot": p["ot"], "dt": p["dt"], "on": p["origin"], "dn": p["destin"],
                "force_other": False
            })
            # gap between same-day events
            if p["ce"].date() == c["cs"].date() and c["cs"] > p["ce"]:
                segs.append({
                    "s": p["ce"], "e": c["cs"],
                    "dur": (c["cs"] - p["ce"]).total_seconds() / 3600.0,
                    "ot": p["dt"], "dt": c["ot"], "on": p["destin"], "dn": c["origin"],
                    "force_other": False
                })

        # last event of the day
        l = recs[-1]
        segs.append({
            "s": l["cs"], "e": l["ce"], "dur": l["dur_hr"],
            "ot": l["ot"], "dt": l["dt"], "on": l["origin"], "dn": l["destin"],
            "force_other": False
        })

    return segs


def alloc_detailed(segs):
    """
    Returns a flat table:
      Day of week | Date | Start Time | End Time | Client Name | Client Hours
    - Start/End are rounded to the nearest 0.25h.
    - Client Hours = exact (End - Start) in hours (no rounding).
    - Segments marked with force_other=True are always 'Other'.
    """
    cols = ["Day of week", "Date", "Start Time", "End Time", "Client Name", "Client Hours"]
    if not segs:
        return pd.DataFrame(columns=cols)

    dep = sorted([s for s in segs if s.get("ot") == "Homeowner"], key=lambda x: x["s"])
    arr = sorted([s for s in segs if s.get("dt") == "Homeowner"], key=lambda x: x["e"])
    s_ts = [s["s"] for s in dep]
    e_ts = [s["e"] for s in arr]

    rows = []
    for s in segs:
        # owner inference (unchanged, except forced 'Other' honored)
        if s.get("force_other"):
            owner = "Other"
        else:
            if any("sittler" in str(n).lower() for n in [s.get("on"), s.get("dn"), s.get("ot"), s.get("dt")]):
                owner = "Other"
            elif s.get("ot") == "Homeowner":
                owner = s.get("on")
            elif s.get("dt") == "Homeowner":
                owner = s.get("dn")
            else:
                i = bisect.bisect_right(e_ts, s["s"])
                h1 = arr[i - 1]["dn"] if i > 0 else None
                j = bisect.bisect_left(s_ts, s["e"])
                h2 = dep[j]["on"] if j < len(s_ts) else None
                owner = h1 or h2 or "Other"

        # round start/end to nearest quarter-hour
        rs = round_to_quarter(s["s"])
        re = round_to_quarter(s["e"])
        if re <= rs:
            continue  # very short segment collapsed by rounding

        hours = (re - rs).total_seconds() / 3600.0  # exact (no rounding)

        rows.append(
            {
                "Day of week": rs.strftime("%A"),
                "Date": rs.date(),
                "Start Time": rs.time(),
                "End Time": re.time(),
                "Client Name": owner,
                "Client Hours": hours,
            }
        )

    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["Date", "Start Time"]).reset_index(drop=True)
    return df



# =========================
# NEW: Consolidate contiguous same-client rows
# =========================
def consolidate_contiguous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge adjacent rows when:
      - Same Date
      - Same Client Name (case/space-insensitive)
      - Previous End Time == Next Start Time
    After merging, recompute Client Hours as exact difference (no rounding).
    """
    if df.empty:
        return df

    work = df.dropna(subset=["Start Time", "End Time", "Client Name"]).copy()
    if work.empty:
        return df

    work["_client_norm"] = work["Client Name"].apply(_norm_client)
    work = work.sort_values(["Date", "_client_norm", "Start Time"]).reset_index(drop=True)

    blocks = []
    for _, row in work.iterrows():
        if not blocks:
            blocks.append(row.to_dict()); continue

        prev = blocks[-1]
        same_day = row["Date"] == prev["Date"]
        same_client = row["_client_norm"] == prev["_client_norm"]
        contiguous = row["Start Time"] == prev["End Time"]

        if same_day and same_client and contiguous:
            prev["End Time"] = row["End Time"]  # extend block
        else:
            blocks.append(row.to_dict())

    cons = pd.DataFrame(blocks)

    def _hours(r):
        start_dt = datetime.datetime.combine(r["Date"], r["Start Time"])
        end_dt   = datetime.datetime.combine(r["Date"], r["End Time"])
        return (end_dt - start_dt).total_seconds() / 3600.0

    cons["Client Hours"] = cons.apply(_hours, axis=1)
    cons["Day of week"] = cons["Date"].apply(lambda d: d.strftime("%A"))
    cons = cons.drop(columns=["_client_norm"], errors="ignore")
    cons = cons[["Day of week", "Date", "Start Time", "End Time", "Client Name", "Client Hours"]]

    # bring back placeholder rows (if any)
    non_merge = df[df["Start Time"].isna() | df["End Time"].isna() | df["Client Name"].isna()]
    out = pd.concat([cons, non_merge], ignore_index=True)
    out = out.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)
    return out


# =========================
# Styled PDF (Detailed Rows)
# =========================
    def export_pdf_detailed(df_week: pd.DataFrame, week_monday: datetime.date, employee_name: str = "Chad Barlow"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            doc = SimpleDocTemplate(
                tmp.name,
                pagesize=landscape(letter),
                leftMargin=0.5 * inch,
                rightMargin=0.5 * inch,
                topMargin=0.5 * inch,
                bottomMargin=0.5 * inch,
            )
    
            # Styles
            h_style = ParagraphStyle(
                "H",
                fontName=PDF_FONT_BOLD,
                fontSize=18,
                alignment=TA_CENTER,
                spaceAfter=28,
                textColor=colors.HexColor("#31333f"),
            )
            l_style = ParagraphStyle(
                "L",
                fontName=PDF_FONT,
                fontSize=10,
                spaceAfter=10,
                textColor=colors.HexColor("#31333f"),
            )
    
            total_hrs = float(df_week["Client Hours"].fillna(0).sum())
            th = int(total_hrs) if total_hrs == int(total_hrs) else round(total_hrs, 2)
    
            elems = [
                Paragraph("HCB TIMESHEET", h_style),
                Paragraph(f"Employee: <b>{employee_name}</b>", l_style),
                Paragraph(f"Week of: <b>{week_monday:%B %-d, %Y}</b>", l_style),
                Paragraph(
                    f'Total Hours: <b><font backcolor="#fffac1" color="#373737">{th}</font></b>',
                    l_style,
                ),
                Spacer(1, 0.18 * inch),
            ]
    
            display_df = df_week.copy()
    
            def fmt_time(t):
                if pd.isna(t):
                    return ""
                return f"{int(t.hour):02d}:{int(t.minute):02d}"
    
            if not display_df.empty:
                display_df["Date"] = display_df["Date"].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "")
                display_df["Start Time"] = display_df["Start Time"].apply(fmt_time)
                display_df["End Time"] = display_df["End Time"].apply(fmt_time)
                display_df["Client Hours"] = display_df["Client Hours"].apply(
                    lambda x: (int(x) if (pd.notna(x) and float(x) == int(x)) else ("" if pd.isna(x) else x))
                )
    
            headers = ["Day", "Date", "Start", "End", "Client", "Hours"]
            data = [headers] + display_df.rename(
                columns={
                    "Day of week": "Day",
                    "Client Name": "Client",
                    "Client Hours": "Hours",
                    "Start Time": "Start",
                    "End Time": "End",
                }
            )[headers].values.tolist()
    
            # helper: add vertical spans for contiguous equal values; blank duplicates
            def _add_vertical_spans(data_matrix, col_idx, first_data_row, style_list):
                if len(data_matrix) <= first_data_row:
                    return
                r0 = first_data_row
                vals = [row[col_idx] for row in data_matrix[first_data_row:]]
                i = 0
                while i < len(vals):
                    j = i + 1
                    while j < len(vals) and vals[j] == vals[i]:
                        j += 1
                    if (j - i) > 1 and str(vals[i]) != "":
                        style_list.append(("SPAN", (col_idx, r0 + i), (col_idx, r0 + j - 1)))
                        for k in range(i + 1, j):
                            data_matrix[first_data_row + k][col_idx] = ""
                    i = j
    
            total_width = landscape(letter)[0] - doc.leftMargin - doc.RightMargin if hasattr(doc, "RightMargin") else landscape(letter)[0] - doc.leftMargin - doc.rightMargin
            col_widths = [
                1.1 * inch,  # Day
                1.3 * inch,  # Date
                1.0 * inch,  # Start
                1.0 * inch,  # End
                total_width - (1.1 + 1.3 + 1.0 + 1.0 + 0.9) * inch,  # Client
                0.9 * inch,  # Hours
            ]
    
            tbl = Table(data, colWidths=col_widths, repeatRows=1)
            style = [
                # header
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#31333f")),
                ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
    
                # body defaults
                ("FONTNAME", (0, 1), (-1, -1), PDF_FONT),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#31333f")),
                ("TOPPADDING", (0, 1), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
    
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),
                ("ALIGN", (2, 1), (3, -1), "RIGHT"),
                ("ALIGN", (5, 1), (5, -1), "RIGHT"),
                ("VALIGN", (0, 1), (1, -1), "MIDDLE"),  # center Day/Date merged cells vertically
            ]
            # NOTE: No ROWBACKGROUNDS and no per-block backgrounds
    
            # apply vertical merges for Day (col 0) and Date (col 1)
            FIRST_DATA_ROW = 1  # header row is 0
            _add_vertical_spans(data, col_idx=0, first_data_row=FIRST_DATA_ROW, style_list=style)  # Day
            _add_vertical_spans(data, col_idx=1, first_data_row=FIRST_DATA_ROW, style_list=style)  # Date
    
            tbl.setStyle(TableStyle(style))
            elems.append(tbl)
    
            doc.build(elems)
            with open(tmp.name, "rb") as f:
                pdf_bytes = f.read()
        os.remove(tmp.name)
        return pdf_bytes


        # ---- helpers ----
        def _add_vertical_spans(data_matrix, col_idx, first_data_row, style_list):
            """Add SPANs for contiguous identical values in the given column; blank duplicates."""
            if len(data_matrix) <= first_data_row:
                return
            r0 = first_data_row
            vals = [row[col_idx] for row in data_matrix[first_data_row:]]
            i = 0
            while i < len(vals):
                j = i + 1
                while j < len(vals) and vals[j] == vals[i]:
                    j += 1
                if (j - i) > 1 and str(vals[i]) != "":
                    style_list.append(("SPAN", (col_idx, r0 + i), (col_idx, r0 + j - 1)))
                    for k in range(i + 1, j):
                        data_matrix[first_data_row + k][col_idx] = ""  # show value only once
                i = j

        def _date_runs(data_matrix, date_col_idx, first_data_row):
            """Return [(row_start, row_end), ...] for contiguous identical Date values."""
            runs = []
            if len(data_matrix) <= first_data_row:
                return runs
            r = first_data_row
            cur = data_matrix[r][date_col_idx]
            start = r
            for rr in range(r + 1, len(data_matrix)):
                v = data_matrix[rr][date_col_idx]
                if v != cur:
                    runs.append((start, rr - 1))
                    cur = v
                    start = rr
            runs.append((start, len(data_matrix) - 1))
            return runs

        total_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
        col_widths = [
            1.1 * inch,  # Day
            1.3 * inch,  # Date
            1.0 * inch,  # Start
            1.0 * inch,  # End
            total_width - (1.1 + 1.3 + 1.0 + 1.0 + 0.9) * inch,  # Client
            0.9 * inch,  # Hours
        ]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        style = [
            # header
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#31333f")),
            ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),

            # body defaults
            ("FONTNAME", (0, 1), (-1, -1), PDF_FONT),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#31333f")),
            ("TOPPADDING", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),

            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),
            ("ALIGN", (2, 1), (3, -1), "RIGHT"),
            ("ALIGN", (5, 1), (5, -1), "RIGHT"),
            ("VALIGN", (0, 1), (1, -1), "MIDDLE"),  # merged Day/Date cells centered vertically
        ]
        # NOTE: intentionally NO ("ROWBACKGROUNDS", ...) here

        # ---- apply vertical merges for Day (col 0) and Date (col 1) ----
        FIRST_DATA_ROW = 1  # header is row 0
        _add_vertical_spans(data, col_idx=0, first_data_row=FIRST_DATA_ROW, style_list=style)  # Day
        _add_vertical_spans(data, col_idx=1, first_data_row=FIRST_DATA_ROW, style_list=style)  # Date

        # ---- apply group-level zebra by Date runs (uniform fill per merged block) ----
        alt_colors = [colors.white, colors.HexColor("#f7f8fb")]
        runs = _date_runs(data, date_col_idx=1, first_data_row=FIRST_DATA_ROW)
        for idx, (r_start, r_end) in enumerate(runs):
            style.append(("BACKGROUND", (0, r_start), (-1, r_end), alt_colors[idx % 2]))

        tbl.setStyle(TableStyle(style))
        elems.append(tbl)

        doc.build(elems)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp.name)
    return pdf_bytes



# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("MileIQ Billables ➜ Detailed Weekly Schedule (Merged Blocks + Styled PDF)")

employee_name = st.text_input("Employee name (for PDF header)", value="Chad Barlow")

files = st.file_uploader(
    "Upload MileIQ CSVs (duplicates ignored)", type=["csv"], accept_multiple_files=True
)

if files:
    unique_files = dedup_files(files)
    if len(unique_files) < len(files):
        st.warning("Duplicate files ignored.")

    # Build full segment list across all files
    all_segs = []
    for f in unique_files:
        df = load_clean(f)
        df = parse_ts(df)
        df = extract_sites(df)
        df = clamp(df)
        all_segs.extend(build_segs(df))

    detailed_all = alloc_detailed(all_segs)

    if detailed_all.empty:
        st.warning("No valid data found in uploaded files.")
        st.stop()

    # Determine available weeks from data
    weeks = find_weeks(detailed_all["Date"].unique())
    default_week = [w for w in [get_mon(datetime.date.today())] if w in weeks] or (weeks[-1:])

    sel_weeks = st.multiselect(
        "Select week(s) to export",
        options=weeks,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        default=default_week,
    )
    if not sel_weeks:
        st.info("Select at least one week to proceed.")
        st.stop()

    enable_edit = st.checkbox(
        "Enable inline edits (Client / Hours / Work Performed)",  # <— UPDATED label
        value=False,
        help="Start/End times derive from segments and are not editable here.",
    )

    for wk in sorted(sel_weeks):
        st.markdown(f"### Week of {wk:%B %d, %Y}")

        # Mon–Fri window
        days = [wk + datetime.timedelta(days=i) for i in range(5)]  # Monday..Friday
        mask = detailed_all["Date"].between(days[0], days[-1])
        df_week = detailed_all.loc[mask].copy()

        # ---- consolidate contiguous same-client blocks before placeholders ----
        df_week = consolidate_contiguous(df_week)

        # --- NEW: ensure Work Performed column exists (blank by default) ---
        if "Work Performed" not in df_week.columns:
            df_week["Work Performed"] = ""

        # Ensure at least one placeholder row per weekday (if empty)
        present = set(df_week["Date"].dropna().unique())
        for d in days:
            if d not in present:
                df_week = pd.concat(
                    [
                        df_week,
                        pd.DataFrame(
                            [
                                {
                                    "Day of week": d.strftime("%A"),
                                    "Date": d,
                                    "Start Time": pd.NaT,
                                    "End Time": pd.NaT,
                                    "Client Name": np.nan,
                                    "Client Hours": np.nan,
                                    "Work Performed": ""   # <— include new column in placeholder
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        # Order for display
        df_week = df_week.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)

        # Editor or read-only
        if enable_edit:
            edited = st.data_editor(
                df_week,
                key=f"edit_{wk}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Client Hours": st.column_config.NumberColumn(label="Client Hours", min_value=0.0, step=0.25),
                    "Work Performed": st.column_config.TextColumn(width="large"),  # <— EDITABLE
                },
                disabled=["Day of week", "Date", "Start Time", "End Time", "Client Name"],
            )
            # Normalize hours rounding up to quarter-hour (unchanged)
            edited["Client Hours"] = edited["Client Hours"].apply(lambda x: r_q_h(x) if pd.notna(x) and x != "" else x)
            render_df = edited
        else:
            render_df = df_week

        # Totals
        total_h = float(render_df["Client Hours"].fillna(0).sum())
        th_str = int(total_h) if total_h == int(total_h) else round(total_h, 2)
        st.markdown(
            f"**Total Hours:** "
            f"<span style='background:#fffac1;color:#373737;padding:2px 6px;border-radius:4px;'>{th_str}</span>",
            unsafe_allow_html=True,
        )

        # Display
        st.dataframe(render_df, use_container_width=True)

        # Downloads: CSV + PDF + inline preview
        csv_bytes = render_df.to_csv(index=False).encode("utf-8")  # includes Work Performed
        st.download_button(
            label=f"Download CSV (Week of {wk:%Y-%m-%d})",
            data=csv_bytes,
            file_name=f"Detailed_Segments_{wk:%Y-%m-%d}.csv",
            mime="text/csv",
            key=f"csv_{wk}",
        )

        pdf_bytes = export_pdf_detailed(render_df.drop(columns=["Work Performed"], errors="ignore"), wk, employee_name=employee_name)
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"HCB_Timesheet_Detailed_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"pdf_{wk}",
        )

        # Inline embed
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f"""
            <object data="data:application/pdf;base64,{b64}" type="application/pdf" width="100%" height="800px">
                <embed src="data:application/pdf;base64,{b64}" type="application/pdf"/>
            </object>
            """,
            unsafe_allow_html=True,
        )
