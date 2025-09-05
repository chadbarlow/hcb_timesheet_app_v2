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
    return df.assign(dur_hr=(df["ce"] - df_
