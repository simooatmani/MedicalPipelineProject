import hashlib
import io
import os
import re
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="Medical Data Cleaning & Versioning Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed",
)

WORKDIR = "pipeline_data"
os.makedirs(WORKDIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# SESSION STATE
DEFAULTS = {
    "stage": "input",  # input -> versioned -> rejected -> done
    "raw_df": None,
    "clean_df": None,
    "final_df": None,
    "dq_results": None,
    "version_tag": None,
    "logs": [],
    "source_url": "",
    "raw_path": None,
    "clean_path": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def log(msg: str):
    st.session_state.logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")


def reset_pipeline():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# --------------------------------------------------------------------------------------
# GIT / DVC
def run_cmd(cmd: str):
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            log(f"⚠️ `{cmd}` failed: {result.stderr.strip()[:200]}")
            return False, result.stderr
        return True, result.stdout
    except FileNotFoundError as e:
        log(f"⚠️ Command not found ({cmd}): {e}")
        return False, str(e)
    except Exception as e:
        log(f"⚠️ Error running `{cmd}`: {e}")
        return False, str(e)

def ensure_git_dvc_repo(local_remote_path: str):
    if not os.path.exists(os.path.join(WORKDIR, ".git")):
        run_cmd("git init -q")
        log("🔧 Initialized Git repository")
    if not os.path.exists(os.path.join(WORKDIR, ".dvc")):
        ok, _ = run_cmd("dvc init -q")
        if ok:
            log("🔧 Initialized DVC repository")
    # I used just local remote storage
    abs_remote = os.path.abspath(local_remote_path)
    os.makedirs(abs_remote, exist_ok=True)
    ok, _ = run_cmd(f"dvc remote add -d -f localremote {abs_remote}")
    if ok:
        log(f"🔧 DVC remote (local) set to {abs_remote}")

def dvc_track(filename: str, message: str):
    ok1, _ = run_cmd(f"dvc add {filename}")
    run_cmd(f"git add {filename}.dvc .gitignore")
    run_cmd(f'git commit -q -m "{message}"')
    if ok1:
        log(f"📦 DVC tracked: {filename}")
    return ok1

def dvc_push():
    ok, _ = run_cmd("dvc push -q")
    if ok:
        log("💾 Pushed data to local DVC remote")
    else:
        log("💾 Push to local remote skipped — tracked locally only")
    return ok

def dvc_tag(version_tag: str):
    run_cmd(f"git tag -f {version_tag}")
    log(f"🏷️ Tagged version: {version_tag}")

# --------------------------------------------------------------------------------------
# STAGE 1: DATA COLLECTION & SCRAPING
def collect_data(url: str) -> pd.DataFrame:
    log(f"🌐 Fetching data from {url}")
    headers = {"User-Agent": "Mozilla/5.0 (MedicalDataPipeline/1.0)"}

    if url.lower().split("?")[0].endswith(".csv"):
        df = pd.read_csv(url)
        log(f"✅ Loaded CSV directly ({df.shape[0]} rows, {df.shape[1]} cols)")
        return df

    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()

    if "csv" in content_type:
        df = pd.read_csv(io.StringIO(resp.text))
        log("✅ Parsed HTTP response as CSV")
        return df

    if "json" in content_type:
        df = pd.read_json(io.StringIO(resp.text))
        log("✅ Parsed HTTP response as JSON")
        return df

    try:
        tables = pd.read_html(io.StringIO(resp.text))
        if tables:
            df = max(tables, key=lambda t: t.shape[0] * max(t.shape[1], 1))
            log(f"✅ Extracted largest HTML table via BeautifulSoup ({df.shape[0]} rows, {df.shape[1]} cols)")
            return df
    except ValueError:
        pass

    soup = BeautifulSoup(resp.text, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.lower().split("?")[0].endswith((".csv", ".json")):
            full_url = href if href.startswith("http") else requests.compat.urljoin(url, href)
            log(f"🔗 Found linked dataset: {full_url}")
            if full_url.lower().endswith(".csv"):
                return pd.read_csv(full_url)
            return pd.read_json(full_url)

    raise ValueError("No tabular CSV/JSON/HTML-table data could be extracted from this URL.")

# --------------------------------------------------------------------------------------
# STAGE 2: ANONYMIZATION & PROCESSING
PHI_PATTERNS = [
    "name", "ssn", "social_security", "mrn", "medical_record", "address", "phone",
    "email", "dob", "date_of_birth", "patient_id", "fullname", "first_name", "last_name",
]

def anonymize_phi(df: pd.DataFrame, k_level: int = 5) -> pd.DataFrame:
    df = df.copy()
    hashed = []
    for col in df.columns:
        cl = col.lower().strip()
        if any(p in cl for p in PHI_PATTERNS):
            df[col] = df[col].apply(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:10])
            hashed.append(col)
    if hashed:
        log(f"🔒 De-identified PHI columns (one-way hashed): {hashed}")

    for col in list(df.columns):
        if "age" in col.lower():
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                bins = list(range(0, 121, k_level))
                df[col + "_bucket"] = pd.cut(numeric, bins=bins, right=False)
                log(f"🔒 Generalized '{col}' into {k_level}-year buckets (k-anonymity)")
            except Exception:
                pass
    return df

def standardize_icd_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    icd_pattern = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")
    for col in df.columns:
        cl = col.lower()
        if "icd" in cl or "diagnosis_code" in cl:
            df[col] = df[col].astype(str).str.upper().str.strip()
            valid_pct = df[col].str.match(icd_pattern, na=False).mean() * 100
            log(f"🩺 Standardized ICD codes in '{col}' ({valid_pct:.1f}% valid ICD-10 format)")
    return df


UNIT_CONVERSIONS = {# mg/dL -> mmol/L
    "glucose": lambda x: x / 18.0,
    "cholesterol": lambda x: x / 38.67,
}

def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        cl = col.lower()
        if "mmol" in cl:
            continue
        for key, fn in UNIT_CONVERSIONS.items():
            if key in cl:
                try:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    if numeric.dropna().median() > 30:  # looks like mg/dL scale
                        df[col] = numeric.apply(lambda x: round(fn(x), 2) if pd.notna(x) else x)
                        log(f"⚗️ Converted '{col}' from mg/dL to mmol/L")
                except Exception:
                    pass
    return df

NA_MARKERS = ["n/a", "na", "null", "none", "unknown", "-", "--", ""]

def handle_inconsistent_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower().isin(NA_MARKERS), col] = np.nan
    log("🧹 Standardized missing-value markers and trimmed whitespace")
    return df

def clean_pipeline(df: pd.DataFrame, k_level: int) -> pd.DataFrame:
    df = handle_inconsistent_values(df)
    df = standardize_icd_codes(df)
    df = normalize_units(df)
    df = anonymize_phi(df, k_level)
    return df

# --------------------------------------------------------------------------------------
# STAGE 3: DATA QUALITY VERIFICATION
def run_dq_checks(df: pd.DataFrame, missing_threshold_pct: float = 15.0):
    results = {}

    miss_pct = df.isna().mean() * 100
    bad_cols = miss_pct[miss_pct > missing_threshold_pct]
    results["missing_value_threshold"] = {
        "status": "pass" if bad_cols.empty else "warn",
        "detail": (f"{len(bad_cols)} feature(s) exceed {missing_threshold_pct:.0f}% missing"
                   if not bad_cols.empty else "All features within missing-value threshold"),
        "critical": False,
    }

    dup_count = int(df.duplicated().sum())
    results["duplicate_record_check"] = {
        "status": "pass" if dup_count == 0 else "warn",
        "detail": f"{dup_count} duplicate row(s) found",
        "critical": False,
    }

    domain_issues = []
    for col in df.columns:
        cl = col.lower()
        if "age" in cl and "_bucket" not in cl:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = int(((vals < 0) | (vals > 120)).sum())
            if bad:
                domain_issues.append(f"{col}: {bad} out-of-range")
        if any(k in cl for k in ["lab", "glucose", "cholesterol", "weight", "height", "bp"]):
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = int((vals < 0).sum())
            if bad:
                domain_issues.append(f"{col}: {bad} negative value(s)")
    results["domain_type_validity"] = {
        "status": "pass" if not domain_issues else "fail",
        "detail": "; ".join(domain_issues) if domain_issues else "All values within expected domain (age 0-120, labs >= 0)",
        "critical": True,
    }

    diag_col = next((c for c in df.columns if "diagnos" in c.lower() and "date" in c.lower()), None)
    treat_col = next((c for c in df.columns if "treat" in c.lower() and "date" in c.lower()), None)
    if diag_col and treat_col:
        d = pd.to_datetime(df[diag_col], errors="coerce")
        t = pd.to_datetime(df[treat_col], errors="coerce")
        bad = int((t < d).sum())
        results["cross_feature_consistency"] = {
            "status": "pass" if bad == 0 else "fail",
            "detail": (f"{bad} row(s) where treatment date precedes diagnosis date"
                       if bad else "Treatment dates consistent with diagnosis dates"),
            "critical": True,
        }
    else:
        results["cross_feature_consistency"] = {
            "status": "pass", "detail": "No diagnosis/treatment date pair found — check skipped", "critical": False,
        }

    outlier_report = []
    for col in df.select_dtypes(include=np.number).columns:
        vals = df[col].dropna()
        if len(vals) > 5 and vals.std() > 0:
            z = (vals - vals.mean()) / vals.std()
            rate = (z.abs() > 3).mean() * 100
            if rate > 5:
                outlier_report.append(f"{col}: {rate:.1f}% extreme outliers")
    results["distribution_consistency"] = {
        "status": "pass" if not outlier_report else "warn",
        "detail": "; ".join(outlier_report) if outlier_report else "Distributions look consistent",
        "critical": False,
    }

    overall_pass = not any(r["critical"] and r["status"] == "fail" for r in results.values())
    return results, overall_pass

# --------------------------------------------------------------------------------------
# STAGE 4: DATA TRANSFORMATION
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

    for col in df.select_dtypes(include="object").columns:
        if 1 < df[col].nunique(dropna=True) <= 15:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                le = LabelEncoder()
                df.loc[non_null.index, col + "_encoded"] = le.fit_transform(non_null.astype(str))

    num_cols = [c for c in df.select_dtypes(include=np.number).columns]
    if num_cols:
        scaler = StandardScaler()
        filled = df[num_cols].apply(lambda c: c.fillna(c.median()))
        scaled = scaler.fit_transform(filled)
        for i, c in enumerate(num_cols):
            df[c + "_scaled"] = scaled[:, i]

    log("🔄 Encoded categoricals, scaled numeric features, formatted column names")
    return df

# --------------------------------------------------------------------------------------
# SIDEBAR — LOCAL STORAGE & SETTINGS
with st.sidebar:
    st.header("⚙️ Local Storage")
    local_remote = st.text_input(
        "Local DVC remote path",
        value="./dvc_store",
        help="A local directory used as the DVC remote. Relative paths are resolved from the working directory.",
    )
    enable_dvc = st.checkbox("Enable DVC version control", value=True,
                              help="Requires `dvc` installed (`pip install dvc`).")
    st.caption("Data is versioned with DVC and stored in the local directory above.")

    st.divider()
    st.header("🔧 Pipeline Settings")
    k_anon_level = st.slider("k-anonymity age bucket width (years)", 2, 10, 5)
    missing_threshold = st.slider("Missing-value threshold per feature (%)", 5, 30, 15)

    st.divider()
    if st.session_state.logs:
        with st.expander("📜 Pipeline Logs", expanded=False):
            for entry in st.session_state.logs[-80:]:
                st.text(entry)

# --------------------------------------------------------------------------------------
# HEADER + STAGE TRACKER
st.title("Medical Data Cleaning & Version Control Pipeline")
st.caption("URL → Scrape → DVC Version (local) → Clean & Anonymize → Quality Gate → Transform → Clean Dataset")

STAGE_LABELS = ["Input", "Collection", "DVC Init", "Cleaning", "DQ Verification", "Transform", "Output"]
STAGE_INDEX = {
    "input": 0, "versioned": 2, "cleaned": 3, "rejected": 4, "done": 6,
}
current_idx = STAGE_INDEX.get(st.session_state.stage, 0)
cols = st.columns(len(STAGE_LABELS))
for i, (c, name) in enumerate(zip(cols, STAGE_LABELS)):
    if st.session_state.stage == "rejected" and i == 4:
        c.error(f"🚫 {name}")
    elif i < current_idx:
        c.success(f"✅ {name}")
    elif i == current_idx:
        c.info(f"▶ {name}")
    else:
        c.markdown(f"<div style='text-align:center;color:#999'>{name}</div>", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------------------
# STAGE: INPUT
if st.session_state.stage == "input":
    st.subheader("📥 Input: Medical Data Source URL")
    with st.form("url_form", clear_on_submit=False):
        url = st.text_input(
            "Enter a medical data source URL (e.g. anonymized dataset link, public database, direct CSV link)",
            value=st.session_state.source_url,
            placeholder="https://example.com/public-dataset.csv",
        )
        submitted = st.form_submit_button("▶ Run Pipeline", use_container_width=True)

    if submitted:
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            st.session_state.source_url = url.strip()
            with st.status("Running pipeline...", expanded=True) as status:
                try:
                    st.write("**Stage 1 — Data Collection & Scraping**")
                    raw_df = collect_data(st.session_state.source_url)
                    st.session_state.raw_df = raw_df
                    raw_path = "raw_data.csv"
                    raw_df.to_csv(os.path.join(WORKDIR, raw_path), index=False)
                    st.session_state.raw_path = raw_path
                    st.write(f"Collected **{raw_df.shape[0]} rows × {raw_df.shape[1]} columns**")

                    st.write("**Stage 2 — Data Versioning (DVC → local)**")
                    if enable_dvc:
                        ensure_git_dvc_repo(local_remote)
                        dvc_track(raw_path, "Add raw medical dataset")
                        dvc_push()
                    else:
                        log("DVC disabled by user — skipping versioning for raw data")
                    st.session_state.stage = "versioned"

                    st.write("**Stage 3 — Cleaning & Anonymization (PHI de-identification, ICD, units)**")
                    clean_df = clean_pipeline(raw_df, k_anon_level)
                    st.session_state.clean_df = clean_df
                    st.write(f"Cleaned dataset: **{clean_df.shape[0]} rows × {clean_df.shape[1]} columns**")

                    st.write("**Stage 4 — Data Quality Verification (DQV)**")
                    dq_results, overall_pass = run_dq_checks(clean_df, missing_threshold)
                    st.session_state.dq_results = dq_results
                    for name, r in dq_results.items():
                        icon = "✅" if r["status"] == "pass" else ("⚠️" if r["status"] == "warn" else "❌")
                        st.write(f"{icon} {name.replace('_', ' ').title()}: {r['detail']}")

                    if not overall_pass:
                        st.session_state.stage = "rejected"
                        status.update(label="🚫 Data Quality Gate REJECTED — pipeline halted", state="error")
                    else:
                        st.write("**Stage 5 — Data Transformation (encoding, scaling, formatting)**")
                        final_df = transform_data(clean_df)
                        st.session_state.final_df = final_df

                        st.write("**Stage 6 — DVC Commit & Tag (clean version)**")
                        version_tag = f"v1.0-anonymized-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        clean_filename = "clean_data.csv"
                        final_df.to_csv(os.path.join(WORKDIR, clean_filename), index=False)
                        st.session_state.clean_path = clean_filename
                        if enable_dvc:
                            dvc_track(clean_filename, f"Add cleaned dataset {version_tag}")
                            dvc_tag(version_tag)
                            dvc_push()
                        st.session_state.version_tag = version_tag
                        st.session_state.stage = "done"
                        status.update(label="✅ Pipeline completed successfully", state="complete")
                except Exception as e:
                    log(f"❌ Pipeline error: {e}")
                    status.update(label=f"❌ Pipeline failed: {e}", state="error")
                    st.error(f"Pipeline failed: {e}")
                    st.session_state.stage = "input"
            st.rerun()

# --------------------------------------------------------------------------------------
# STAGE: REJECTED
elif st.session_state.stage == "rejected":
    st.error("🚫 Data Quality Gate: REJECTED")
    st.write("The cleaned dataset failed one or more critical quality checks:")
    for name, r in st.session_state.dq_results.items():
        icon = "✅" if r["status"] == "pass" else ("⚠️" if r["status"] == "warn" else "❌")
        st.write(f"{icon} **{name.replace('_', ' ').title()}** — {r['detail']}")
    st.info("Retry with another URL or view previous versions via DVC tags / `git log`.")
    if st.button("🔄 Retry with another URL", use_container_width=True):
        reset_pipeline()
        st.rerun()

# --------------------------------------------------------------------------------------
# STAGE: OUTPUT & DOWNLOAD
elif st.session_state.stage == "done":
    local_tag_msg = (
        f"✅ Dataset cleaned, verified, and version-tagged as `{st.session_state.version_tag}` "
        f"(DVC-tracked, stored in `{local_remote}`)"
        if enable_dvc else
        f"✅ Dataset cleaned and verified — tag `{st.session_state.version_tag}` (DVC disabled)"
    )
    st.success(local_tag_msg)

    with st.expander("📋 Data Quality Verification Summary", expanded=False):
        for name, r in st.session_state.dq_results.items():
            icon = "✅" if r["status"] == "pass" else "⚠️"
            st.write(f"{icon} **{name.replace('_', ' ').title()}** — {r['detail']}")

    st.subheader("📊 Clean Data Output — Dataset Ready for Modeling")

    final_df = st.session_state.final_df
    if not isinstance(final_df, pd.DataFrame):
        st.error("Session state lost — please re-run the pipeline.")
        if st.button("🔄 Restart"):
            reset_pipeline()
            st.rerun()
        st.stop()

    st.dataframe(final_df, use_container_width=True)

    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download CSV file",
            data=csv_bytes,
            file_name=f"clean_medical_data_{st.session_state.version_tag}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("↩️ Return", use_container_width=True):
            reset_pipeline()
            st.rerun()