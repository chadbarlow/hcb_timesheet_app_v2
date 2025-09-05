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
    """Round a datetime to the nearest 15 minutes. Half-quarters (7.5 min) round up."""
    if pd.isna(dt):
        return dt
    base = dt.replace(second=0, microsecond=0, minute=0)
    mins = dt.minute + dt.second / 60.0
    q = int(round(mins / 15.0))  # 0..4
    return base + datetime.timedelta(minutes=15 * q)

# =========================
# Data processing (MileIQ path)
# =========================
def load_clean_from_text(file_name: str, text: str) -> pd.DataFrame:
    """Load MileIQ CSV from full text. Finds the header row with START_DATE*."""
    try:
        lines = text.splitlines()
        h_row = next(i for i, l in enumerate(lines) if "START_DATE*" in l)
    except StopIteration:
        raise ValueError(f"Header not found in {file_name}.")
    df = pd.read_csv(StringIO(text), skiprows=h_row)
    df = df.loc[df["START_DATE*"].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}")].copy()
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
    """Build clamped event segments + same-day gaps. Also add pre-8am gap if first segment starts after 08:00."""
    if df.empty:
        return []
    df = df.sort_values("cs").copy()
    df["_day"] = df["cs"].dt.date
    segs = []
    for day, g in df.groupby("_day"):
        recs = list(g.to_dict("records"))
        if not recs:
            continue
        eight = datetime.datetime.combine(day, datetime.time(8, 0))
        first_cs = recs[0]["cs"]
        if first_cs > eight:
            segs.append({"s": eight, "e": first_cs, "dur": (first_cs - eight).total_seconds()/3600.0,
                         "ot": None, "dt": None, "on": None, "dn": None, "force_other": True})
        for p, c in zip(recs, recs[1:]):
            segs.append({"s": p["cs"], "e": p["ce"], "dur": p["dur_hr"],
                         "ot": p["ot"], "dt": p["dt"], "on": p["origin"], "dn": p["destin"], "force_other": False})
            if p["ce"].date() == c["cs"].date() and c["cs"] > p["ce"]:
                segs.append({"s": p["ce"], "e": c["cs"], "dur": (c["cs"] - p["ce"]).total_seconds()/3600.0,
                             "ot": p["dt"], "dt": c["ot"], "on": p["destin"], "dn": c["origin"], "force_other": False})
        l = recs[-1]
        segs.append({"s": l["cs"], "e": l["ce"], "dur": l["dur_hr"],
                     "ot": l["ot"], "dt": l["dt"], "on": l["origin"], "dn": l["destin"], "force_other": False})
    return segs

def alloc_detailed(segs):
    """Produce detailed rows from segments (rounded to nearest 0.25h for bounds; hours = exact diff)."""
    cols = ["Day of week", "Date", "Start Time", "End Time", "Client Name", "Client Hours"]
    if not segs:
        return pd.DataFrame(columns=cols)
    dep = sorted([s for s in segs if s.get("ot") == "Homeowner"], key=lambda x: x["s"])
    arr = sorted([s for s in segs if s.get("dt") == "Homeowner"], key=lambda x: x["e"])
    s_ts = [s["s"] for s in dep]
    e_ts = [s["e"] for s in arr]
    rows = []
    for s in segs:
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
        rs = round_to_quarter(s["s"])
        re = round_to_quarter(s["e"])
        if re <= rs:
            continue
        hours = (re - rs).total_seconds() / 3600.0
        rows.append({"Day of week": rs.strftime("%A"), "Date": rs.date(),
                     "Start Time": rs.time(), "End Time": re.time(),
                     "Client Name": owner, "Client Hours": hours})
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["Date", "Start Time"]).reset_index(drop=True)
    return df

# =========================
# Consolidate contiguous same-client rows
# =========================
def consolidate_contiguous(df: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent rows when same date/client and prev End == next Start."""
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
        if (row["Date"] == prev["Date"]
            and row["_client_norm"] == prev["_client_norm"]
            and row["Start Time"] == prev["End Time"]):
            prev["End Time"] = row["End Time"]
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
    non_merge = df[df["Start Time"].isna() | df["End Time"].isna() | df["Client Name"].isna()]
    out = pd.concat([cons, non_merge], ignore_index=True)
    out = out.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)
    return out

# =========================
# Revised CSV normalization ➜ convert to internal schema
# =========================
def _normalize_revised_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers to: Day, Date, Start, End, Client, Work Performed, Hours."""
    variants = {
        "day of week": "Day", "day": "Day", "dow": "Day",
        "date": "Date",
        "start time": "Start", "start": "Start",
        "end time": "End", "end": "End",
        "client name": "Client", "client": "Client",
        "client hours": "Hours", "hours": "Hours", "hrs": "Hours",
        "work performed": "Work Performed", "notes": "Work Performed", "work": "Work Performed",
    }
    rename = {}
    for c in df_raw.columns:
        key = c.strip().lower()
        if key in variants:
            rename[c] = variants[key]
    df = df_raw.rename(columns=rename).copy()
    for col in ["Day", "Date", "Start", "End", "Client", "Work Performed", "Hours"]:
        if col not in df.columns:
            df[col] = ""
    return df[["Day", "Date", "Start", "End", "Client", "Work Performed", "Hours"]].copy()

def _parse_time_to_timeobj(v):
    if pd.isna(v) or v == "":
        return pd.NaT
    if isinstance(v, datetime.time):
        return v
    try:
        ts = pd.to_datetime(str(v), errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return ts.time()
    except Exception:
        return pd.NaT

def _parse_date_to_dateobj(v):
    if pd.isna(v) or v == "":
        return pd.NaT
    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
        return v
    try:
        ts = pd.to_datetime(str(v), errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return ts.date()
    except Exception:
        return pd.NaT

def detailed_from_revised_text(text: str) -> pd.DataFrame:
    """Read a non-MileIQ CSV and return internal detailed schema."""
    try:
        df_raw = pd.read_csv(StringIO(text))
    except Exception:
        df_raw = pd.read_csv(StringIO(text), encoding="latin-1")
    df_norm = _normalize_revised_csv(df_raw)
    # Convert to internal schema columns
    day = df_norm["Day"].astype(str).fillna("")
    date = df_norm["Date"].apply(_parse_date_to_dateobj)
    start_t = df_norm["Start"].apply(_parse_time_to_timeobj)
    end_t   = df_norm["End"].apply(_parse_time_to_timeobj)
    client = df_norm["Client"].astype(str)
    # Hours numeric if possible
    def _h(x):
        try:
            v = float(x)
            return v
        except Exception:
            return np.nan
    hours = df_norm["Hours"].apply(_h)
    work = df_norm["Work Performed"].fillna("").astype(str)

    # If day is missing/empty, regenerate from date
    def _dow(d, fallback):
        if pd.isna(d):
            return fallback or ""
        return d.strftime("%A")
    day_full = [(_dow(d, day.iloc[i] if day.iloc[i] not in ["", "nan"] else "")) for i, d in enumerate(date)]

    out = pd.DataFrame({
        "Day of week": day_full,
        "Date": date,
        "Start Time": start_t,
        "End Time": end_t,
        "Client Name": client,
        "Client Hours": hours,
        "Work Performed": work,
    })
    # Drop rows without a real Date or Start/End if both missing
    out = out[~out["Date"].isna()].copy()
    out = out.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)
    return out

# =========================
# Universal ingestion: MileIQ or Revised
# =========================
def ingest_file_to_detailed(file_obj) -> pd.DataFrame:
    """Return a detailed DataFrame from an uploaded file, regardless of format."""
    # Read the entire upload to text once (robust to encoding)
    try:
        content = file_obj.read()
    finally:
        try:
            file_obj.seek(0)
        except Exception:
            pass
    # Try utf-8 first, then latin-1
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin-1")

    # Heuristic: MileIQ if any line contains START_DATE*
    if "START_DATE*" in text:
        try:
            df = load_clean_from_text(file_obj.name, text)
        except ValueError:
            # If oddly formatted, fall back to revised path
            return detailed_from_revised_text(text)
        df = parse_ts(df)
        df = extract_sites(df)
        df = clamp(df)
        segs = build_segs(df)
        detailed = alloc_detailed(segs)
        # ensure Work Performed exists
        if "Work Performed" not in detailed.columns:
            detailed["Work Performed"] = ""
        return detailed

    # Otherwise treat as Revised/Already-Detailed CSV
    detailed = detailed_from_revised_text(text)
    # Fill missing Work Performed
    if "Work Performed" not in detailed.columns:
        detailed["Work Performed"] = ""
    return detailed

# =========================
# Styled PDF (merged Day/Date, no zebra, Work Performed 50%, short Mon / MM-DD)
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
        h_style = ParagraphStyle("H", fontName=PDF_FONT_BOLD, fontSize=18, alignment=TA_CENTER,
                                 spaceAfter=28, textColor=colors.HexColor("#31333f"))
        l_style = ParagraphStyle("L", fontName=PDF_FONT, fontSize=10, spaceAfter=10,
                                 textColor=colors.HexColor("#31333f"))
        wp_style = ParagraphStyle("WP", fontName=PDF_FONT, fontSize=9, leading=11,
                                  textColor=colors.HexColor("#31333f"))

        total_hrs = float(df_week["Client Hours"].fillna(0).sum())
        th = int(total_hrs) if total_hrs == int(total_hrs) else round(total_hrs, 2)

        elems = [
            Paragraph("HCB TIMESHEET", h_style),
            Paragraph(f"Employee: <b>{employee_name}</b>", l_style),
            Paragraph(f"Week of: <b>{week_monday:%B %-d, %Y}</b>", l_style),
            Paragraph(f'Total Hours: <b><font backcolor="#fffac1" color="#373737">{th}</font></b>', l_style),
            Spacer(1, 0.18 * inch),
        ]

        display_df = df_week.copy()
        if "Work Performed" not in display_df.columns:
            display_df["Work Performed"] = ""

        def fmt_time(t):
            if pd.isna(t):
                return ""
            return f"{int(t.hour):02d}:{int(t.minute):02d}"

        if not display_df.empty:
            display_df["_DayShort"] = display_df["Date"].apply(lambda d: d.strftime("%a") if pd.notna(d) else "")
            display_df["_DateShort"] = display_df["Date"].apply(lambda d: d.strftime("%m-%d") if pd.notna(d) else "")
            display_df["Start Time"] = display_df["Start Time"].apply(fmt_time)
            display_df["End Time"] = display_df["End Time"].apply(fmt_time)
            display_df["Client Hours"] = display_df["Client Hours"].apply(
                lambda x: (int(x) if (pd.notna(x) and float(x) == int(x)) else ("" if pd.isna(x) else x))
            )
            display_df["Work Performed"] = display_df["Work Performed"].fillna("").astype(str)

        headers = ["Day", "Date", "Start", "End", "Client", "Work Performed", "Hours"]
        df_for_pdf = pd.DataFrame({
            "Day": display_df["_DayShort"] if "_DayShort" in display_df else "",
            "Date": display_df["_DateShort"] if "_DateShort" in display_df else "",
            "Start": display_df["Start Time"],
            "End": display_df["End Time"],
            "Client": display_df["Client Name"],
            "Work Performed": display_df["Work Performed"],
            "Hours": display_df["Client Hours"],
        })
        mapped = df_for_pdf[headers].values.tolist()
        for row in mapped:
            row[5] = Paragraph(str(row[5]), wp_style)
        data = [headers] + mapped

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

        total_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
        work_w = total_width * 0.5
        other_w = total_width - work_w

        base_day   = 1.1 * inch
        base_date  = 1.3 * inch
        base_start = 1.0 * inch
        base_end   = 1.0 * inch
        base_hours = 0.9 * inch
        base_client = total_width - (base_day + base_date + base_start + base_end + base_hours)
        scale = other_w / (base_day + base_date + base_start + base_end + base_client + base_hours)
        day_w, date_w = base_day*scale, base_date*scale
        start_w, end_w = base_start*scale, base_end*scale
        client_w, hours_w = base_client*scale, base_hours*scale
        col_widths = [day_w, date_w, start_w, end_w, client_w, work_w, hours_w]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#31333f")),
            ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),

            ("FONTNAME", (0, 1), (-1, -1), PDF_FONT),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#31333f")),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),

            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),
            ("ALIGN", (2, 1), (3, -1), "RIGHT"),
            ("ALIGN", (6, 1), (6, -1), "RIGHT"),
            ("VALIGN", (0, 1), (1, -1), "MIDDLE"),
            ("VALIGN", (5, 1), (5, -1), "TOP"),
        ]
        FIRST_DATA_ROW = 1
        _add_vertical_spans(data, 0, FIRST_DATA_ROW, style)
        _add_vertical_spans(data, 1, FIRST_DATA_ROW, style)

        tbl.setStyle(TableStyle(style))
        elems.append(tbl)
        doc.build(elems)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    os.remove(tmp.name)
    return pdf_bytes

# =========================
# UI & shared render
# =========================
st.set_page_config(layout="wide")
st.title("MileIQ Billables ➜ Detailed Weekly Schedule (Merged Blocks + Styled PDF)")

employee_name = st.text_input("Employee name (for PDF header)", value="Chad Barlow")

def _render_pipeline(file_list, section_key_prefix=""):
    """Shared render for both 'usual' and 'revised' sections; auto-detects file type."""
    if not file_list:
        return

    # Build combined detailed rows across all files (MileIQ or Revised)
    detailed_frames = []
    for f in file_list:
        try:
            detailed_frames.append(ingest_file_to_detailed(f))
        except Exception as e:
            st.error(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    if not detailed_frames:
        st.warning("No valid data found in uploaded files.")
        return

    detailed_all = pd.concat(detailed_frames, ignore_index=True)
    # Ensure dtypes and columns
    for col in ["Day of week", "Date", "Start Time", "End Time", "Client Name", "Client Hours"]:
        if col not in detailed_all.columns:
            detailed_all[col] = pd.NA
    if "Work Performed" not in detailed_all.columns:
        detailed_all["Work Performed"] = ""

    # Determine available weeks
    weeks = find_weeks(detailed_all["Date"].dropna().unique())
    if not weeks:
        st.warning("Could not infer week(s) from dates in the uploaded files.")
        return
    default_week = [w for w in [get_mon(datetime.date.today())] if w in weeks] or (weeks[-1:])

    sel_weeks = st.multiselect(
        "Select week(s) to export",
        options=weeks,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        default=default_week,
        key=f"{section_key_prefix}weeks"
    )
    if not sel_weeks:
        st.info("Select at least one week to proceed.")
        return

    enable_edit = st.checkbox(
        "Enable inline edits (Client / Hours / Work Performed)",
        value=False,
        help="Start/End times derive from segments and are not editable here.",
        key=f"{section_key_prefix}edit_toggle"
    )

    for wk in sorted(sel_weeks):
        st.markdown(f"### Week of {wk:%B %d, %Y}")

        days = [wk + datetime.timedelta(days=i) for i in range(5)]  # Mon..Fri
        mask = detailed_all["Date"].between(days[0], days[-1])
        df_week = detailed_all.loc[mask].copy()

        # Consolidate contiguous same-client blocks before placeholders
        df_week = consolidate_contiguous(df_week)

        # Ensure Work Performed column exists (blank by default)
        if "Work Performed" not in df_week.columns:
            df_week["Work Performed"] = ""

        # Ensure at least one placeholder row per weekday (if empty)
        present = set(df_week["Date"].dropna().unique())
        for d in days:
            if d not in present:
                df_week = pd.concat(
                    [
                        df_week,
                        pd.DataFrame([{
                            "Day of week": d.strftime("%A"), "Date": d,
                            "Start Time": pd.NaT, "End Time": pd.NaT,
                            "Client Name": np.nan, "Client Hours": np.nan,
                            "Work Performed": ""
                        }])
                    ],
                    ignore_index=True,
                )

        df_week = df_week.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)

        if enable_edit:
            edited = st.data_editor(
                df_week,
                key=f"{section_key_prefix}edit_{wk}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Client Hours": st.column_config.NumberColumn(label="Client Hours", min_value=0.0, step=0.25),
                    "Work Performed": st.column_config.TextColumn(width="large"),
                },
                disabled=["Day of week", "Date", "Start Time", "End Time", "Client Name"],
            )
            edited["Client Hours"] = edited["Client Hours"].apply(lambda x: r_q_h(x) if pd.notna(x) and x != "" else x)
            render_df = edited
        else:
            render_df = df_week

        total_h = float(render_df["Client Hours"].fillna(0).sum())
        th_str = int(total_h) if total_h == int(total_h) else round(total_h, 2)
        st.markdown(
            f"**Total Hours:** "
            f"<span style='background:#fffac1;color:#373737;padding:2px 6px;border-radius:4px;'>{th_str}</span>",
            unsafe_allow_html=True,
        )

        st.dataframe(render_df, use_container_width=True)

        csv_bytes = render_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download CSV (Week of {wk:%Y-%m-%d})",
            data=csv_bytes,
            file_name=f"Detailed_Segments_{wk:%Y-%m-%d}.csv",
            mime="text/csv",
            key=f"{section_key_prefix}csv_{wk}",
        )

        pdf_bytes = export_pdf_detailed(render_df, wk, employee_name=employee_name)
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"HCB_Timesheet_Detailed_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"{section_key_prefix}pdf_{wk}",
        )

        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f"""
            <object data="data:application/pdf;base64,{b64}" type="application/pdf" width="100%" height="800px">
                <embed src="data:application/pdf;base64,{b64}" type="application/pdf"/>
            </object>
            """,
            unsafe_allow_html=True,
        )

# =========================
# UI sections
# =========================
st.set_page_config(layout="wide")

# Main (usual) uploader — now accepts MileIQ or Revised
st.subheader("Upload MileIQ CSVs (or detailed CSVs); duplicates ignored")
files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True, key="main_files")
if files:
    _render_pipeline(files, section_key_prefix="main_")

# Revised section — same processing
st.markdown("---")
st.header("Revised CSV — Process & Format (same as usual)")
revised_files = st.file_uploader("Upload revised CSVs", type=["csv"], accept_multiple_files=True, key="revised_files")
if revised_files:
    _render_pipeline(revised_files, section_key_prefix="rev_")
