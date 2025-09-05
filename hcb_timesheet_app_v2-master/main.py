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
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

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
    segs, recs = [], df.to_dict("records")
    if not recs:
        return []
    for p, c in zip(recs, recs[1:]):
        # clamped event segment
        segs.append(
            {
                "s": p["cs"],
                "e": p["ce"],
                "dur": p["dur_hr"],
                "ot": p["ot"],
                "dt": p["dt"],
                "on": p["origin"],
                "dn": p["destin"],
            }
        )
        # same-day gap segment (covers travel/idle)
        if p["ce"].date() == c["cs"].date() and c["cs"] > p["ce"]:
            segs.append(
                {
                    "s": p["ce"],
                    "e": c["cs"],
                    "dur": (c["cs"] - p["ce"]).total_seconds() / 3600,
                    "ot": p["dt"],
                    "dt": c["ot"],
                    "on": p["destin"],
                    "dn": c["origin"],
                }
            )
    l = recs[-1]
    segs.append(
        {
            "s": l["cs"],
            "e": l["ce"],
            "dur": l["dur_hr"],
            "ot": l["ot"],
            "dt": l["dt"],
            "on": l["origin"],
            "dn": l["destin"],
        }
    )
    return segs

def alloc_detailed(segs):
    """
    Returns a flat table with columns:
    Day of week | Date | Start Time | End Time | Client Name | Client Hours
    - Client Hours are rounded UP to the nearest 0.25h
    - Start/End times remain as actual clamped event/gap boundaries
    """
    cols = ["Day of week", "Date", "Start Time", "End Time", "Client Name", "Client Hours"]
    if not segs:
        return pd.DataFrame(columns=cols)

    dep = sorted([s for s in segs if s["ot"] == "Homeowner"], key=lambda x: x["s"])
    arr = sorted([s for s in segs if s["dt"] == "Homeowner"], key=lambda x: x["e"])
    s_ts = [s["s"] for s in dep]
    e_ts = [s["e"] for s in arr]

    rows = []
    for s in segs:
        # owner selection
        if any("sittler" in str(n).lower() for n in [s["on"], s["dn"], s["ot"], s["dt"]]):
            owner = "Other"
        elif s["ot"] == "Homeowner":
            owner = s["on"]
        elif s["dt"] == "Homeowner":
            owner = s["dn"]
        else:
            i = bisect.bisect_right(e_ts, s["s"])
            h1 = arr[i - 1]["dn"] if i > 0 else None
            j = bisect.bisect_left(s_ts, s["e"])
            h2 = dep[j]["on"] if j < len(s_ts) else None
            owner = h1 or h2 or "Other"

        if s["e"] > s["s"]:
            dur = r_q_h((s["e"] - s["s"]).total_seconds() / 3600.0)
            rows.append(
                {
                    "Day of week": s["s"].strftime("%A"),
                    "Date": s["s"].date(),
                    "Start Time": s["s"].time(),
                    "End Time": s["e"].time(),
                    "Client Name": owner,
                    "Client Hours": dur,
                }
            )

    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["Date", "Start Time"]).reset_index(drop=True)
    return df

# =========================
# Styled PDF (Detailed Rows)
# =========================
def export_pdf_detailed(df_week: pd.DataFrame, week_monday: datetime.date, employee_name: str = "Chad Barlow"):
    """
    Build a landscape PDF with a detailed, per-row schedule:
    Day | Date | Start | End | Client | Hours
    """
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

        # Compute total hours (ignoring NaNs)
        total_hrs = float(df_week["Client Hours"].fillna(0).sum())
        th = int(total_hrs) if total_hrs == int(total_hrs) else round(total_hrs, 2)

        # Header
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

        # Table data
        # Ensure columns and friendly formatting
        display_df = df_week.copy()
        # format date and time for table
        def fmt_time(t):
            if pd.isna(t):
                return ""
            return f"{int(t.hour):02d}:{int(t.minute):02d}"

        if not display_df.empty:
            display_df["Date"] = display_df["Date"].apply(lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else "")
            display_df["Start Time"] = display_df["Start Time"].apply(fmt_time)
            display_df["End Time"] = display_df["End Time"].apply(fmt_time)
            # Pretty print hours (strip .0)
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

        # Column widths
        total_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
        col_widths = [
            1.1 * inch,  # Day
            1.3 * inch,  # Date
            1.0 * inch,  # Start
            1.0 * inch,  # End
            total_width - (1.1 + 1.3 + 1.0 + 1.0 + 0.9) * inch,  # Client (flex)
            0.9 * inch,  # Hours
        ]

        tbl = Table(data, colWidths=col_widths, repeatRows=1)

        base_style = [
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
            ("TOPPADDING", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),

            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e5e8")),

            ("ALIGN", (0, 1), (0, -1), "LEFT"),   # Day
            ("ALIGN", (1, 1), (1, -1), "LEFT"),   # Date
            ("ALIGN", (2, 1), (3, -1), "RIGHT"),  # Start/End
            ("ALIGN", (4, 1), (4, -1), "LEFT"),   # Client
            ("ALIGN", (5, 1), (5, -1), "RIGHT"),  # Hours
        ]
        # Zebra rows
        base_style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [None, colors.HexColor("#f7f8fb")]))

        tbl.setStyle(TableStyle(base_style))
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
st.title("MileIQ Billables ➜ Detailed Weekly Schedule (with Styled PDF)")

# Optional header config
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

    enable_edit = st.checkbox("Enable inline edits (Client / Hours)", value=False, help="Start/End times derive from travel/visit segments and are not editable here.")

    for wk in sorted(sel_weeks):
        st.markdown(f"### Week of {wk:%B %d, %Y}")

        # Mon–Fri window (to match your sample output)
        days = [wk + datetime.timedelta(days=i) for i in range(5)]  # Monday..Friday
        mask = detailed_all["Date"].between(days[0], days[-1])
        df_week = detailed_all.loc[mask].copy()

        # Ensure one placeholder row per weekday if empty
        present = set(df_week["Date"].unique())
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
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        # Order for display
        df_week = df_week.sort_values(["Date", "Start Time"], na_position="last").reset_index(drop=True)

        # Show editor or read-only table
        if enable_edit:
            # Configure editor: allow editing Client Name & Hours; lock Day/Date/Times
            edited = st.data_editor(
                df_week,
                key=f"edit_{wk}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Client Hours": st.column_config.NumberColumn(label="Client Hours", min_value=0.0, step=0.25),
                },
                disabled=["Day of week", "Date", "Start Time", "End Time"],
            )
            # Normalize hours rounding up to quarter-hour
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

        # Display table
        st.dataframe(render_df, use_container_width=True)

        # Downloads: CSV + PDF + inline preview
        csv_bytes = render_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download CSV (Week of {wk:%Y-%m-%d})",
            data=csv_bytes,
            file_name=f"Detailed_Segments_{wk:%Y-%m-%d}.csv",
            mime="text/csv",
            key=f"csv_{wk}",
        )

        pdf_bytes = export_pdf_detailed(render_df, wk, employee_name=employee_name)
        st.download_button(
            label=f"Download PDF (Week of {wk:%Y-%m-%d})",
            data=pdf_bytes,
            file_name=f"HCB_Timesheet_Detailed_{wk:%Y-%m-%d}.pdf",
            mime="application/pdf",
            key=f"pdf_{wk}",
        )

        # Inline embed (desktop-friendly <object> + <embed> fallback)
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f"""
            <object data="data:application/pdf;base64,{b64}" type="application/pdf" width="100%" height="800px">
                <embed src="data:application/pdf;base64,{b64}" type="application/pdf"/>
            </object>
            """,
            unsafe_allow_html=True,
        )
