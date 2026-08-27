# ============================================================
# ❄️ LOTTO AI V.MAX COLD/DEAD ENSEMBLE
# ============================================================
# PURPOSE:
#   วิเคราะห์ "เลขดับ" โดยตรง
#
# AI:
#   RandomForest + ExtraTrees + HistGradientBoosting
#
# STAT:
#   Frequency + Transition + Pattern
#
# EQUATION:
#   Direct Lag แยกจาก Equation
#
# VALIDATION:
#   Strict Walk-Forward
#   Cold Stability
#   Recent Stability
#
# FEATURES:
#   ใช้เฉพาะข้อมูลย้อนหลัง
#   ไม่มี DOW / Month / Calendar เป็น Feature
#
# OUTPUT:
#   COLD-5 แยกทุกหลัก
#   COLD-5 บน
#   COLD-5 ล่าง
#   Cold Score
#   Cold Stability
#   ประวัติย้อนหลัง 10 งวด
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import warnings

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. HELPER FUNCTION & CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX COLD/DEAD",
    page_icon="❄️",
    layout="centered"
)

def render_html(html_str):
    """
    ฟังก์ชันลบช่องว่างด้านหน้าทุกบรรทัด 
    ป้องกัน Streamlit เรนเดอร์เป็น Code Block
    """
    clean_html = "\n".join([line.strip() for line in html_str.split("\n")])
    st.markdown(clean_html, unsafe_allow_html=True)

# ============================================================
# 1. LOTTERY SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "1. หวยไทย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",

    "2. หวยธกส.":
        "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",

    "3. หวยออมสิน":
        "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",

    "4. หวยลาว":
        "https://suksan18190.blogspot.com/2026/07/blog-post.html",

    "5. หวยฮานอย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",

    "6. หวยมาเลย์":
        "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",

    "7. หวยหุ้นไทยเย็น":
        "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",

    "8. หวยหุ้นนิเคอิบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",

    "9. หวยหุ้นฮั่งเส็งบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",

    "10. หวยหุ้นจีนบ่าย":
        "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}

# ============================================================
# 2. CSS
# ============================================================

render_html(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 4px;
        color: #0d47a1;
    }
    .sub-title {
        font-size: 14px;
        text-align: center;
        color: #546e7a;
        margin-bottom: 15px;
    }
    .position-title {
        font-size: 18px;
        font-weight: 900;
        margin-top: 18px;
        margin-bottom: 8px;
        color: #1565c0;
    }
    .cold-card {
        padding: 15px;
        border-radius: 14px;
        border: 1px solid #bbdefb;
        background-color: #e3f2fd;
        margin: 8px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .cold-card-strong {
        padding: 16px;
        border-radius: 14px;
        border: 2px solid #90caf9;
        background-color: #e3f2fd;
        margin: 10px 0;
    }
    .number-highlight {
        font-size: 27px;
        font-weight: 900;
        padding: 4px 8px;
        color: #0d47a1;
    }
    .dot-sep {
        color: #90caf9;
        margin: 0 4px;
    }
    .score {
        font-size: 14px;
        color: #546e7a;
    }
    .good {
        color: #2e7d32;
        font-weight: 800;
    }
    .warn {
        color: #ef6c00;
        font-weight: 800;
    }
    .bad {
        color: #c62828;
        font-weight: 800;
    }
    .info-row {
        padding: 5px 0;
        font-size: 14px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 7px;
        font-weight: 700;
        margin-right: 4px;
    }
    .badge-ai {
        background-color: #fff3e0;
        border: 1px solid #ffcc80;
        color: #e65100;
    }
    .badge-stat {
        background-color: #e8f5e9;
        border: 1px solid #a5d6a7;
        color: #2e7d32;
    }
    .badge-eq {
        background-color: #f3e5f5;
        border: 1px solid #ce93d8;
        color: #6a1b9a;
    }
    .history-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        background: #fafafa;
    }
    </style>
    """
)

# ============================================================
# 3. FETCH DATA
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_clean_data(url):

    try:

        headers = {
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        main = soup.find(
            "div",
            class_=re.compile(
                r"post-body|entry-content|post-content|content"
            )
        )

        if main is None:
            main = soup

        lines = main.get_text(
            separator="\n"
        ).split("\n")

        date_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}|"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        )

        num_pattern = re.compile(
            r"\b(\d{3})\b.*?\b(\d{2})\b|"
            r"\b(\d{5,6})\b.*?\b(\d{2})\b"
        )

        current_date = pd.Timestamp(
            datetime.now()
        )

        rows = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            dm = date_pattern.search(line)

            if dm:

                try:

                    d = pd.to_datetime(
                        dm.group(1),
                        errors="coerce"
                    )

                    if not pd.isna(d):
                        current_date = d

                except:
                    pass

            nm = num_pattern.search(line)

            if not nm:
                continue

            if nm.group(1):

                r3 = nm.group(1)
                r2 = nm.group(2)

            elif nm.group(3):

                r3 = nm.group(3)[-3:]
                r2 = nm.group(4)

            else:
                continue

            rows.append(
                {
                    "Date": current_date,
                    "Result_3D": str(r3).zfill(3),
                    "Result_2D": str(r2).zfill(2)
                }
            )

        if len(rows) < 10:
            raise ValueError(
                "ข้อมูลน้อยเกินไป"
            )

        df = pd.DataFrame(rows)

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = (
            df
            .dropna()
            .drop_duplicates()
            .sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        st.error(
            f"❌ ดึงข้อมูลไม่ได้: {e}"
        )

        return pd.DataFrame()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    lags=(1, 2, 3, 5),
    rolls=(3, 5, 10)
):

    x = df.copy()

    r3 = x["Result_3D"].astype(str)
    r2 = x["Result_2D"].astype(str)

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)

    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    # --------------------------------------------------------
    # Previous information
    # --------------------------------------------------------

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = ph + pt + po

    x["PrevOdd"] = (
        (ph % 2) +
        (pt % 2) +
        (po % 2)
    )

    x["DistHT"] = (ph - pt).abs()
    x["DistTO"] = (pt - po).abs()

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

    for pos in ["H", "T", "O", "T2", "O2"]:

        s = x[pos]
        prev = s.shift(1)

        x[f"Odd_{pos}"] = (
            prev % 2
        )

        x[f"High_{pos}"] = (
            prev >= 5
        ).astype(np.int8)

        x[f"Prime_{pos}"] = (
            prev.isin([2, 3, 5, 7])
        ).astype(np.int8)

        x[f"Mod3_{pos}"] = (
            prev % 3
        )

        x[f"Mod4_{pos}"] = (
            prev % 4
        )

        # ----------------------------------------------------
        # Lags
        # ----------------------------------------------------

        for lag in lags:

            x[f"L{lag}_{pos}"] = (
                s.shift(lag)
            )

        # ----------------------------------------------------
        # Rolling
        # ----------------------------------------------------

        for w in rolls:

            x[f"RM{w}_{pos}"] = (
                prev
                .rolling(
                    w,
                    min_periods=1
                )
                .mean()
            )

            x[f"RSTD{w}_{pos}"] = (
                prev
                .rolling(
                    w,
                    min_periods=1
                )
                .std()
                .fillna(0)
            )

        # ----------------------------------------------------
        # Skip / Gap
        # ----------------------------------------------------

        arr = s.to_numpy()

        raw_skip = np.zeros(
            len(arr),
            dtype=np.float32
        )

        last = np.full(
            10,
            -1,
            dtype=np.int32
        )

        for i, val in enumerate(arr):

            v = int(val)

            if last[v] < 0:
                raw_skip[i] = i
            else:
                raw_skip[i] = (
                    i - last[v]
                )

            last[v] = i

        x[f"Skip_{pos}"] = (
            pd.Series(
                raw_skip,
                index=x.index
            ).shift(1)
        )

    return (
        x
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(-1)
    )


# ============================================================
# 5. FREQUENCY ENGINE
# ============================================================

class FrequencyEngine:

    def analyze(
        self,
        df,
        pos
    ):

        s = df[pos].astype(int)

        if len(s) == 0:
            return np.ones(10) / 10

        r10 = (
            s.tail(10)
            .value_counts(
                normalize=True
            )
        )

        r20 = (
            s.tail(20)
            .value_counts(
                normalize=True
            )
        )

        r50 = (
            s.tail(50)
            .value_counts(
                normalize=True
            )
        )

        all_f = (
            s.value_counts(
                normalize=True
            )
        )

        score = np.array(
            [
                r10.get(d, 0) * 0.50
                +
                r20.get(d, 0) * 0.30
                +
                r50.get(d, 0) * 0.15
                +
                all_f.get(d, 0) * 0.05

                for d in range(10)
            ]
        )

        score = score + 0.01

        return score / score.sum()


# ============================================================
# 6. TRANSITION ENGINE
# ============================================================

class TransitionEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 6:
            return np.ones(10) / 10

        current = int(
            df[pos].iloc[-1]
        )

        subset = df[
            df[pos].shift(1) == current
        ]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
        )

        score = np.array(
            [
                freq.get(d, 0)
                for d in range(10)
            ]
        )

        score += 0.01

        return score / score.sum()


# ============================================================
# 7. PATTERN ENGINE
# ============================================================

class PatternEngine:

    def analyze(
        self,
        df,
        pos
    ):

        if len(df) < 7:
            return np.ones(10) / 10

        a = int(
            df[pos].iloc[-1]
        )

        b = int(
            df[pos].iloc[-2]
        )

        subset = df[
            (df[pos].shift(1) == a)
            &
            (df[pos].shift(2) == b)
        ]

        if len(subset) < 2:

            subset = df[
                df[pos].shift(1) == a
            ]

        if len(subset) < 1:
            return np.ones(10) / 10

        freq = (
            subset[pos]
            .value_counts(
                normalize=True
            )
        )

        score = np.array(
            [
                freq.get(d, 0)
                for d in range(10)
            ]
        )

        score += 0.01

        return score / score.sum()


# ============================================================
# 8. EQUATION ENGINE
# ============================================================

class ColdEquationEngine:

    def discover(
        self,
        X_hist,
        y_hist,
        X_next,
        pos,
        bt=10
    ):

        default = {
            "prob": np.ones(10) / 10,
            "cold": [],
            "strength": 0.0,
            "stable": 0,
            "total": 0,
            "equations": []
        }

        n = len(X_hist)

        if n < 50:
            return default

        required = [
            f"L1_{pos}",
            f"L2_{pos}",
            f"L3_{pos}",
            f"L5_{pos}"
        ]

        for col in required:

            if col not in X_hist.columns:
                return default

        L1 = X_hist[
            f"L1_{pos}"
        ].values

        L2 = X_hist[
            f"L2_{pos}"
        ].values

        L3 = X_hist[
            f"L3_{pos}"
        ].values

        L5 = X_hist[
            f"L5_{pos}"
        ].values

        actual = y_hist.values

        nL1 = X_next[
            f"L1_{pos}"
        ].values[0]

        nL2 = X_next[
            f"L2_{pos}"
        ].values[0]

        nL3 = X_next[
            f"L3_{pos}"
        ].values[0]

        nL5 = X_next[
            f"L5_{pos}"
        ].values[0]

        # ----------------------------------------------------
        # Direct Lag
        # ----------------------------------------------------

        direct_hist = {
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "L5": L5
        }

        direct_next = {
            "L1": nL1,
            "L2": nL2,
            "L3": nL3,
            "L5": nL5
        }

        # ----------------------------------------------------
        # Actual equations
        # ----------------------------------------------------

        equation_hist = {

            "L1+L2":
                L1 + L2,

            "L1-L2":
                np.abs(L1 - L2),

            "L1+L3":
                L1 + L3,

            "L1-L3":
                np.abs(L1 - L3),

            "L2+L3":
                L2 + L3,

            "L1+L5":
                L1 + L5,

            "L3+L5":
                L3 + L5,

            "L1+L2+L3":
                L1 + L2 + L3,

            "2L1+L2":
                2 * L1 + L2,

            "L1*L2":
                L1 * L2,

            "2L1":
                L1 * 2,

            "L1+5":
                L1 + 5,

            "L2+L5":
                L2 + L5,

            "L2-L5":
                np.abs(L2 - L5)
        }

        equation_next = {

            "L1+L2":
                nL1 + nL2,

            "L1-L2":
                abs(nL1 - nL2),

            "L1+L3":
                nL1 + nL3,

            "L1-L3":
                abs(nL1 - nL3),

            "L2+L3":
                nL2 + nL3,

            "L1+L5":
                nL1 + nL5,

            "L3+L5":
                nL3 + nL5,

            "L1+L2+L3":
                nL1 + nL2 + nL3,

            "2L1+L2":
                2 * nL1 + nL2,

            "L1*L2":
                nL1 * nL2,

            "2L1":
                nL1 * 2,

            "L1+5":
                nL1 + 5,

            "L2+L5":
                nL2 + nL5,

            "L2-L5":
                abs(nL2 - nL5)
        }

        candidates = []

        start = max(
            35,
            n - bt
        )

        recent_start = max(
            0,
            n - 5
        )

        # ----------------------------------------------------
        # Evaluate Direct Lag
        # ----------------------------------------------------

        for name, arr in direct_hist.items():

            pred = (
                np.floor(
                    np.nan_to_num(
                        arr,
                        nan=-999
                    )
                ).astype(int)
                % 10
            )

            hit = np.mean(
                pred[start:n] == actual[start:n]
            )

            recent = np.mean(
                pred[recent_start:n]
                ==
                actual[recent_start:n]
            )

            if (
                hit >= 0.10
                and recent >= 0.10
            ):

                score = (
                    0.60 * hit
                    +
                    0.40 * recent
                )

                n_pred = int(
                    np.floor(
                        np.nan_to_num(
                            direct_next[name],
                            nan=-999
                        )
                    )
                ) % 10

                candidates.append(
                    {
                        "name": name,
                        "type": "Direct",
                        "score": score,
                        "hit": hit,
                        "recent": recent,
                        "pred": n_pred
                    }
                )

        # ----------------------------------------------------
        # Evaluate Equations
        # ----------------------------------------------------

        for name, arr in equation_hist.items():

            pred = (
                np.floor(
                    np.nan_to_num(
                        arr,
                        nan=-999
                    )
                ).astype(int)
                % 10
            )

            hit = np.mean(
                pred[start:n] == actual[start:n]
            )

            recent = np.mean(
                pred[recent_start:n]
                ==
                actual[recent_start:n]
            )

            if (
                hit >= 0.12
                and recent >= 0.15
            ):

                score = (
                    0.65 * hit
                    +
                    0.35 * recent
                )

                n_pred = int(
                    np.floor(
                        np.nan_to_num(
                            equation_next[name],
                            nan=-999
                        )
                    )
                ) % 10

                candidates.append(
                    {
                        "name": name,
                        "type": "Equation",
                        "score": score,
                        "hit": hit,
                        "recent": recent,
                        "pred": n_pred
                    }
                )

        if not candidates:
            return default

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        selected = candidates[:10]

        # ----------------------------------------------------
        # Equation probability
        # ----------------------------------------------------

        prob = np.zeros(10)

        total_weight = 0

        for r in selected:

            if r["type"] == "Equation":
                base = 1.00
            else:
                base = 0.70

            weight = (
                base
                +
                r["score"]
            )

            prob[r["pred"]] += weight
            total_weight += weight

        if total_weight <= 0:
            prob = np.ones(10) / 10
        else:
            prob /= total_weight
            prob += 0.01
            prob /= prob.sum()

        cold = [
            (
                int(i),
                float(prob[i])
            )
            for i in np.argsort(prob)[:5]
        ]

        return {
            "prob": prob,
            "cold": cold,
            "strength": float(
                np.mean(
                    [
                        r["hit"]
                        for r in selected
                    ]
                )
            ),
            "stable": len(selected),
            "total":
                len(
                    direct_hist
                )
                +
                len(
                    equation_hist
                ),
            "equations": selected
        }


# ============================================================
# 9. AI ENGINE
# ============================================================

class ColdAI:

    def __init__(
        self,
        trees=80,
        weights=(0.40, 0.35, 0.25)
    ):

        self.trees = trees
        self.weights = weights

    def predict(
        self,
        X,
        y,
        X_next,
        fast_mode=False
    ):

        rf_w, et_w, hgb_w = (
            self.weights
        )

        result = np.zeros(10)

        total_w = 0

        trees = (
            max(15, self.trees // 2)
            if fast_mode
            else self.trees
        )

        iterations = (
            40
            if fast_mode
            else 90
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        if rf_w > 0:

            model = RandomForestClassifier(
                n_estimators=trees,
                max_depth=6,
                min_samples_leaf=5,
                class_weight=None,
                n_jobs=-1,
                random_state=42
            )

            model.fit(
                X,
                y
            )

            probs = model.predict_proba(
                X_next
            )[0]

            for c, p in zip(
                model.classes_,
                probs
            ):
                result[int(c)] += (
                    p * rf_w
                )

            total_w += rf_w

        # ----------------------------------------------------
        # Extra Trees
        # ----------------------------------------------------

        if et_w > 0:

            model = ExtraTreesClassifier(
                n_estimators=trees,
                max_depth=6,
                min_samples_leaf=5,
                class_weight=None,
                n_jobs=-1,
                random_state=43
            )

            model.fit(
                X,
                y
            )

            probs = model.predict_proba(
                X_next
            )[0]

            for c, p in zip(
                model.classes_,
                probs
            ):
                result[int(c)] += (
                    p * et_w
                )

            total_w += et_w

        # ----------------------------------------------------
        # Hist Gradient Boosting
        # ----------------------------------------------------

        if hgb_w > 0:

            model = HistGradientBoostingClassifier(
                max_iter=iterations,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                l2_regularization=2.0,
                random_state=44
            )

            model.fit(
                X,
                y
            )

            probs = model.predict_proba(
                X_next
            )[0]

            for c, p in zip(
                model.classes_,
                probs
            ):
                result[int(c)] += (
                    p * hgb_w
                )

            total_w += hgb_w

        if total_w <= 0:
            return np.ones(10) / 10

        result /= total_w

        result += 0.001

        return result / result.sum()


# ============================================================
# 10. COLD SCORE ENGINE
# ============================================================

class ColdScoreEngine:

    @staticmethod
    def normalize_inverse(prob):

        p = np.asarray(
            prob,
            dtype=float
        )

        # เลข probability ต่ำ
        # จะได้ Cold score สูง

        inv = (
            p.max() - p
        )

        if inv.max() <= 0:
            return np.ones(10) / 10

        inv += 0.001

        return inv / inv.sum()

    @staticmethod
    def stability_score(
        cold_history
    ):

        if len(cold_history) == 0:
            return np.zeros(10)

        score = np.zeros(10)

        for cold_set in cold_history:

            for d in cold_set:

                score[int(d)] += 1

        score /= len(
            cold_history
        )

        return score

    def final_score(
        self,
        ai,
        freq,
        trans,
        pattern,
        equation,
        skip_score,
        stability
    ):

        # ----------------------------------------------------
        # Convert probability -> cold orientation
        # ----------------------------------------------------

        ai_c = self.normalize_inverse(ai)
        fq_c = self.normalize_inverse(freq)
        tr_c = self.normalize_inverse(trans)
        pt_c = self.normalize_inverse(pattern)
        eq_c = self.normalize_inverse(equation)

        # ----------------------------------------------------
        # Cold ensemble
        # ----------------------------------------------------

        cold = (
            0.30 * ai_c
            +
            0.18 * fq_c
            +
            0.14 * tr_c
            +
            0.12 * pt_c
            +
            0.16 * eq_c
            +
            0.05 * skip_score
            +
            0.05 * stability
        )

        cold += 1e-9

        return (
            cold /
            cold.sum()
        )


# ============================================================
# 11. MAIN ENSEMBLE
# ============================================================

class ColdEnsemble:

    def __init__(
        self,
        df,
        lottery_name,
        target_dow=None
    ):

        self.df = df.copy()

        self.lottery_name = (
            lottery_name
        )

        self.target_dow = target_dow

        n = len(df)

        self.lags = [
            1, 2, 3, 5
        ]

        self.rolls = [
            3, 5, 10
        ]

        self.trees = 80

        self.bt = (
            12 if n >= 700
            else
            10 if n >= 400
            else
            8
        )

        # ----------------------------------------------------
        # Feature list
        # ----------------------------------------------------

        self.features = [
            "PrevSum",
            "PrevOdd",
            "DistHT",
            "DistTO"
        ]

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            self.features.extend(
                [
                    f"Odd_{pos}",
                    f"High_{pos}",
                    f"Prime_{pos}",
                    f"Mod3_{pos}",
                    f"Mod4_{pos}",
                    f"Skip_{pos}"
                ]
            )

            for lag in self.lags:
                self.features.append(
                    f"L{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.append(
                    f"RM{w}_{pos}"
                )

                self.features.append(
                    f"RSTD{w}_{pos}"
                )

        # ----------------------------------------------------
        # Engines
        # ----------------------------------------------------

        self.freq = (
            FrequencyEngine()
        )

        self.transition = (
            TransitionEngine()
        )

        self.pattern = (
            PatternEngine()
        )

        self.equation = (
            ColdEquationEngine()
        )

        self.ai = ColdAI(
            trees=self.trees,
            weights=(
                0.40,
                0.35,
                0.25
            )
        )

        self.cold_engine = (
            ColdScoreEngine()
        )

        # ----------------------------------------------------
        # Base weights
        # ----------------------------------------------------

        self.base_weights = {
            "AI": 0.30,
            "Freq": 0.18,
            "ST": 0.14,
            "BT": 0.12,
            "Eq": 0.16,
            "Skip": 0.05,
            "Stability": 0.05
        }

    # ========================================================
    # SKIP SCORE
    # ========================================================

    def build_skip_score(
        self,
        hist,
        pos
    ):

        s = (
            hist[pos]
            .astype(int)
            .tolist()
        )

        if len(s) == 0:
            return np.ones(10) / 10

        last_seen = [
            None
        ] * 10

        for i, d in enumerate(s):
            last_seen[d] = i

        current = len(s) - 1

        gaps = np.zeros(10)

        for d in range(10):

            if last_seen[d] is None:
                gaps[d] = len(s)

            else:
                gaps[d] = (
                    current -
                    last_seen[d]
                )

        # Normalize high gap -> high cold tendency
        gaps += 1e-6

        return gaps / gaps.sum()

    # ========================================================
    # BACKTEST
    # ========================================================

    def backtest_cold(
        self,
        pos,
        ext,
        bt_points=None
    ):

        n = len(ext)

        if bt_points is None:
            bt_points = self.bt

        if n < 60:

            return {
                "cold_stability":
                    np.ones(10) * 0.10,

                "top5_rate": 0.0,

                "message":
                    "ข้อมูลไม่เพียงพอ"
            }

        start = max(
            45,
            n - bt_points
        )

        history = []

        hits = []

        for idx in range(
            start,
            n
        ):

            hist = ext.iloc[
                :idx
            ].copy()

            actual = int(
                ext[pos].iloc[idx]
            )

            fq = (
                self.freq.analyze(
                    hist,
                    pos
                )
            )

            tr = (
                self.transition.analyze(
                    hist,
                    pos
                )
            )

            pt = (
                self.pattern.analyze(
                    hist,
                    pos
                )
            )

            skip = (
                self.build_skip_score(
                    hist,
                    pos
                )
            )

            # Simple statistical cold proxy
            cold = (
                0.45 *
                self.cold_engine
                .normalize_inverse(fq)

                +

                0.25 *
                self.cold_engine
                .normalize_inverse(tr)

                +

                0.20 *
                self.cold_engine
                .normalize_inverse(pt)

                +

                0.10 *
                skip
            )

            top5 = np.argsort(
                cold
            )[-5:]

            # top5 = เลขที่โมเดลมองว่า cold สูง
            cold_set = [
                int(x)
                for x in top5
            ]

            history.append(
                cold_set
            )

            # ถ้าผลจริงอยู่ใน COLD
            # ถือว่า "ดับหลุด"
            hits.append(
                0
                if actual in cold_set
                else 1
            )

        stability = (
            self.cold_engine
            .stability_score(
                history
            )
        )

        top5_rate = (
            float(np.mean(hits))
            if hits
            else 0.0
        )

        return {
            "cold_stability":
                stability,

            "top5_rate":
                top5_rate,

            "message":
                f"Cold Stability "
                f"{top5_rate:.0%}"
        }

    # ========================================================
    # PROCESS POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next,
        cold_stability
    ):

        ai = self.ai.predict(
            X,
            hist[pos],
            X_next,
            fast_mode=True
        )

        fq = (
            self.freq.analyze(
                hist,
                pos
            )
        )

        tr = (
            self.transition.analyze(
                hist,
                pos
            )
        )

        pt = (
            self.pattern.analyze(
                hist,
                pos
            )
        )

        eq_res = (
            self.equation.discover(
                X,
                hist[pos],
                X_next,
                pos,
                bt=self.bt
            )
        )

        eq = eq_res["prob"]

        skip = (
            self.build_skip_score(
                hist,
                pos
            )
        )

        # ----------------------------------------------------
        # Final Cold Score
        # ----------------------------------------------------

        final = (
            self.cold_engine.final_score(
                ai=ai,
                freq=fq,
                trans=tr,
                pattern=pt,
                equation=eq,
                skip_score=skip,
                stability=cold_stability
            )
        )

        # ----------------------------------------------------
        # COLD TOP 5
        # ----------------------------------------------------

        cold_idx = np.argsort(
            final
        )[-5:][::-1]

        # เพราะ final คือ Cold Score
        # สูง = ดับมากกว่า

        cold = [
            (
                int(i),
                float(final[i])
            )
            for i in cold_idx
        ]

        # ----------------------------------------------------
        # AI cold
        # ----------------------------------------------------

        ai_cold = (
            self.cold_engine
            .normalize_inverse(ai)
        )

        ai_top = np.argsort(
            ai_cold
        )[-3:][::-1]

        ai_cold = [
            (
                int(i),
                float(ai_cold[i])
            )
            for i in ai_top
        ]

        # ----------------------------------------------------
        # STAT cold
        # ----------------------------------------------------

        stat = (
            fq +
            tr +
            pt
        ) / 3

        stat_cold = (
            self.cold_engine
            .normalize_inverse(stat)
        )

        stat_top = np.argsort(
            stat_cold
        )[-3:][::-1]

        stat_cold = [
            (
                int(i),
                float(stat_cold[i])
            )
            for i in stat_top
        ]

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        stability_top = np.argsort(
            cold_stability
        )[-5:][::-1]

        stability_top = [
            (
                int(i),
                float(cold_stability[i])
            )
            for i in stability_top
        ]

        return {

            "Cold":
                cold,

            "ColdScore":
                final,

            "AI_Cold":
                ai_cold,

            "Stat_Cold":
                stat_cold,

            "Equation_Cold":
                eq_res["cold"],

            "Stability":
                stability_top,

            "EquationStrength":
                eq_res["strength"],

            "StableEquations":
                eq_res["stable"],

            "TotalEquations":
                eq_res["total"],

            "AI_Prob":
                ai,

            "Freq":
                fq,

            "Transition":
                tr,

            "Pattern":
                pt,

            "EquationProb":
                eq,

            "Skip":
                skip
        }

    # ========================================================
    # PREDICT ALL
    # ========================================================

    def predict_all(
        self,
        progress,
        status
    ):

        last_date = (
            self.df["Date"].iloc[-1]
        )

        # ----------------------------------------------------
        # Target date
        # ----------------------------------------------------

        if self.target_dow is not None:

            days = (
                self.target_dow
                -
                last_date.dayofweek
            ) % 7

            if days == 0:
                days = 7

        else:

            if len(self.df) >= 2:

                gap = (
                    self.df["Date"].iloc[-1]
                    -
                    self.df["Date"].iloc[-2]
                ).days

                days = max(
                    1,
                    gap
                )

            else:

                days = 1

        next_date = (
            last_date
            +
            timedelta(days=days)
        )

        # ----------------------------------------------------
        # Add dummy row
        # ----------------------------------------------------

        ext = pd.concat(
            [
                self.df,

                pd.DataFrame(
                    [
                        {
                            "Date":
                                next_date,

                            "Result_3D":
                                "000",

                            "Result_2D":
                                "00"
                        }
                    ]
                )
            ],
            ignore_index=True
        )

        status.markdown(
            "🧠 **Step 2/4:** "
            "สร้าง Historical Features..."
        )

        ext = build_features(
            ext,
            self.lags,
            self.rolls
        )

        progress.progress(30)

        hist = ext.iloc[
            :-1
        ].copy()

        X = hist[
            self.features
        ].astype(
            np.float32
        )

        X_next = ext.iloc[
            [-1]
        ][
            self.features
        ].astype(
            np.float32
        )

        # ----------------------------------------------------
        # Cold stability
        # ----------------------------------------------------

        status.markdown(
            "📊 **Step 3/4:** "
            "คำนวณ Cold Stability..."
        )

        stability_map = {}

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            bt = self.backtest_cold(
                pos,
                ext.iloc[:-1],
                bt_points=20
            )

            stability_map[pos] = (
                bt["cold_stability"]
            )

        progress.progress(45)

        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------

        predictions = {}

        for pos in [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]:

            predictions[pos] = (
                self.process_position(
                    pos,
                    hist,
                    X,
                    X_next,
                    stability_map[pos]
                )
            )

        progress.progress(70)

        return (
            predictions,
            next_date,
            ext
        )

    # ========================================================
    # PAST 10
    # ========================================================

    def evaluate_past_10(
        self,
        ext,
        predictions,
        progress,
        status
    ):

        n = len(self.df)

        count = min(
            10,
            n - 60
        )

        if count < 1:
            return []

        records = []

        start = n - count

        for step, i in enumerate(
            range(start, n)
        ):

            status.markdown(
                f"🕰️ **Step 4/4:** "
                f"ตรวจสอบย้อนหลัง "
                f"{step+1}/{count}"
            )

            progress.progress(
                70
                +
                int(
                    ((step + 1) / count)
                    * 30
                )
            )

            hist = ext.iloc[
                :i
            ].copy()

            X = hist[
                self.features
            ].astype(
                np.float32
            )

            X_next = ext.iloc[
                [i]
            ][
                self.features
            ].astype(
                np.float32
            )

            row = {
                "Date":
                    self.df.iloc[i]["Date"],

                "Result_3D":
                    str(
                        self.df.iloc[i]
                        ["Result_3D"]
                    ).zfill(3),

                "Result_2D":
                    str(
                        self.df.iloc[i]
                        ["Result_2D"]
                    ).zfill(2)
            }

            for pos in [
                "H",
                "T",
                "O",
                "T2",
                "O2"
            ]:

                bt = self.backtest_cold(
                    pos,
                    ext.iloc[:i],
                    bt_points=12
                )

                pred = (
                    self.process_position(
                        pos,
                        hist,
                        X,
                        X_next,
                        bt["cold_stability"]
                    )
                )

                cold_numbers = [
                    int(x[0])
                    for x in pred["Cold"]
                ]

                if pos in [
                    "H",
                    "T",
                    "O"
                ]:

                    actual = int(
                        row["Result_3D"][
                            ["H", "T", "O"]
                            .index(pos)
                        ]
                    )

                else:

                    actual = int(
                        row["Result_2D"][
                            ["T2", "O2"]
                            .index(pos)
                        ]
                    )

                # actual ไม่อยู่ COLD = ดับสำเร็จ
                success = (
                    actual
                    not in
                    cold_numbers
                )

                row[f"{pos}_Actual"] = (
                    actual
                )

                row[f"{pos}_Cold"] = (
                    cold_numbers
                )

                row[f"{pos}_Success"] = (
                    success
                )

            records.append(row)

        return records[::-1]


# ============================================================
# 12. HTML HELPERS
# ============================================================

def html_cold(items):

    return (
        '<span class="dot-sep">'
        '•'
        '</span>'
        .join(
            [
                f'<span class="number-highlight">'
                f'{n}'
                f'</span>'
                for n, p in items
            ]
        )
    )


def html_numbers(items):

    return (
        " • ".join(
            [
                str(n)
                for n, p in items
            ]
        )
    )


def html_score(items):

    return (
        " | ".join(
            [
                f"{n} ({p:.1%})"
                for n, p in items
            ]
        )
    )


def overall_cold(
    predictions,
    positions,
    n=5
):

    score = np.mean(
        [
            predictions[p]["ColdScore"]
            for p in positions
        ],
        axis=0
    )

    idx = np.argsort(
        score
    )[-n:][::-1]

    return [
        (
            int(i),
            float(score[i])
        )
        for i in idx
    ]


# ============================================================
# 13. UI
# ============================================================

render_html(
    """
    <div class="main-title">
    ❄️ LOTTO AI V.MAX COLD/DEAD
    </div>
    """
)

render_html(
    """
    <div class="sub-title">
    AI + Statistics + Equation + 
    <b>Cold Score + Cold Stability</b><br>
    ระบบจัดอันดับเลขที่มีคะแนนดับสูงสุดจากข้อมูลย้อนหลัง
    </div>
    """
)

st.divider()

c1, c2 = st.columns(2)

selected_lotto = c1.selectbox(
    "🎯 เลือกหวย",
    list(
        LOTTERY_SOURCES.keys()
    )
)

day_options = {
    "อัตโนมัติ": None,
    "วันจันทร์": 0,
    "วันอังคาร": 1,
    "วันพุธ": 2,
    "วันพฤหัสบดี": 3,
    "วันศุกร์": 4,
    "วันเสาร์": 5,
    "วันอาทิตย์": 6
}

day_label = c2.selectbox(
    "📅 วันออกรางวัล",
    list(
        day_options.keys()
    )
)

# ============================================================
# RUN
# ============================================================

if st.button(
    "❄️ เริ่มวิเคราะห์ COLD/DEAD V.MAX",
    type="primary",
    use_container_width=True
):

    progress = st.progress(0)

    status = st.empty()

    status.markdown(
        "⏳ **Step 1/4:** "
        "โหลดข้อมูล..."
    )

    df = fetch_and_clean_data(
        LOTTERY_SOURCES[
            selected_lotto
        ]
    )

    if df.empty:

        status.error(
            "❌ ไม่มีข้อมูล"
        )

        st.stop()

    if len(df) < 70:

        status.error(
            f"❌ ต้องมีข้อมูลอย่างน้อย "
            f"70 งวด แต่พบ {len(df)} งวด"
        )

        st.stop()

    progress.progress(15)

    engine = ColdEnsemble(
        df,
        selected_lotto,
        day_options[
            day_label
        ]
    )

    predictions, next_date, ext = (
        engine.predict_all(
            progress,
            status
        )
    )

    past_records = (
        engine.evaluate_past_10(
            ext,
            predictions,
            progress,
            status
        )
    )

    status.empty()
    progress.empty()

    # ========================================================
    # TARGET
    # ========================================================

    days = [
        "จันทร์",
        "อังคาร",
        "พุธ",
        "พฤหัสบดี",
        "ศุกร์",
        "เสาร์",
        "อาทิตย์"
    ]

    st.divider()

    st.info(
        f"📅 งวดเป้าหมาย: "
        f"วัน{days[next_date.dayofweek]} "
        f"{next_date.strftime('%d-%m-%Y')} "
        f"| ข้อมูล {len(df)} งวด"
    )

    # ========================================================
    # LABEL
    # ========================================================

    labels = {

        "H":
            "หลักร้อย 3 ตัวบน",

        "T":
            "หลักสิบ 3 ตัวบน",

        "O":
            "หลักหน่วย 3 ตัวบน",

        "T2":
            "หลักสิบ 2 ตัวล่าง",

        "O2":
            "หลักหน่วย 2 ตัวล่าง"
    }

    # ========================================================
    # EACH POSITION
    # ========================================================

    for pos in [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]:

        res = predictions[pos]

        render_html(
            f"""
            <div class="position-title">
            📍 {labels[pos]}
            </div>
            """
        )

        cold = res["Cold"]

        stability = (
            res["Stability"]
        )

        render_html(
            f"""
            <div class="cold-card-strong">
            <div style="font-weight:900; margin-bottom:8px; color:#0d47a1;">
            ❄️ FINAL COLD-5
            </div>
            <div style="text-align:center; margin:10px 0;">
            {html_cold(cold)}
            </div>
            <div style="text-align:center; color:#546e7a; font-size:13px;">
            Cold Score: {html_score(cold)}
            </div>
            </div>
            <div class="info-row">
            🤖 <b>AI COLD-3:</b>
            <span class="badge badge-ai">
            {html_numbers(res["AI_Cold"])}
            </span>
            </div>
            <div class="info-row">
            📊 <b>STAT COLD-3:</b>
            <span class="badge badge-stat">
            {html_numbers(res["Stat_Cold"])}
            </span>
            </div>
            <div class="info-row">
            🧮 <b>EQUATION COLD-5:</b>
            <span class="badge badge-eq">
            {html_numbers(res["Equation_Cold"])}
            </span>
            </div>
            <div class="info-row">
            🧊 <b>STABILITY COLD-5:</b>
            {html_numbers(stability)}
            </div>
            <div style="font-size:13px; color:#777; margin-top:8px;">
            🧮 Equation: <b>{res["StableEquations"]}</b> / {res["TotalEquations"]} &nbsp; | &nbsp; Strength: <b>{res["EquationStrength"]:.0%}</b>
            </div>
            """
        )

    # ========================================================
    # OVERALL
    # ========================================================

    st.subheader(
        "❄️ COLD-5 ภาพรวม"
    )

    upper = overall_cold(
        predictions,
        ["H", "T", "O"],
        5
    )

    lower = overall_cold(
        predictions,
        ["T2", "O2"],
        5
    )

    render_html(
        f"""
        <div class="cold-card">
        <div style="font-weight:900; color:#1565c0;">
        ❄️ COLD-5 TOP บน
        </div>
        <div style="text-align:center; margin:10px 0;">
        {html_cold(upper)}
        </div>
        <div style="text-align:center; color:#546e7a; font-size:13px;">
        {html_score(upper)}
        </div>
        </div>
        <div class="cold-card">
        <div style="font-weight:900; color:#1565c0;">
        ❄️ COLD-5 TOP ล่าง
        </div>
        <div style="text-align:center; margin:10px 0;">
        {html_cold(lower)}
        </div>
        <div style="text-align:center; color:#546e7a; font-size:13px;">
        {html_score(lower)}
        </div>
        </div>
        """
    )

    # ========================================================
    # BACKTEST HISTORY
    # ========================================================

    if past_records:

        st.subheader(
            "📜 ประวัติย้อนหลัง 10 งวด"
        )

        st.caption(
            "หลักการ: "
            "ถ้าผลจริงไม่อยู่ใน COLD-5 "
            "ถือว่าเลขดับสำเร็จ"
        )

        def mark(
            success,
            actual
        ):

            if success:

                return (
                    "✅ "
                    "<span class='good'>"
                    "ดับสำเร็จ"
                    "</span>"
                )

            return (
                "❌ "
                "<span class='bad'>"
                f"ดับหลุด "
                f"(มา {actual})"
                "</span>"
            )

        for rec in past_records:

            render_html(
                f"""
                <div class="history-card">
                <div style="font-weight:900; color:#1565c0; margin-bottom:10px;">
                📅 {rec["Date"].strftime("%d-%m-%Y")} &nbsp;|&nbsp; ผล: <span style="color:#d32f2f;">{rec["Result_3D"]}-{rec["Result_2D"]}</span>
                </div>
                <div class="info-row">
                <b>หลักร้อยบน:</b> {" • ".join(map(str, rec["H_Cold"]))} → {mark(rec["H_Success"], rec["H_Actual"])}
                </div>
                <div class="info-row">
                <b>หลักสิบบน:</b> {" • ".join(map(str, rec["T_Cold"]))} → {mark(rec["T_Success"], rec["T_Actual"])}
                </div>
                <div class="info-row">
                <b>หลักหน่วยบน:</b> {" • ".join(map(str, rec["O_Cold"]))} → {mark(rec["O_Success"], rec["O_Actual"])}
                </div>
                <hr>
                <div class="info-row">
                <b>หลักสิบล่าง:</b> {" • ".join(map(str, rec["T2_Cold"]))} → {mark(rec["T2_Success"], rec["T2_Actual"])}
                </div>
                <div class="info-row">
                <b>หลักหน่วยล่าง:</b> {" • ".join(map(str, rec["O2_Cold"]))} → {mark(rec["O2_Success"], rec["O2_Actual"])}
                </div>
                </div>
                """
            )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    st.success(
        "✅ วิเคราะห์เสร็จสิ้น — "
        "V.MAX COLD/DEAD Ensemble"
    )

    st.caption(
        "⚠️ Cold Score เป็นคะแนนจากโมเดลและสถิติย้อนหลัง "
        "ไม่ได้หมายความว่าเลขนั้นจะไม่ออกจริง "
        "และผล Backtest ไม่ใช่การรับประกันผลในอนาคต"
    )
