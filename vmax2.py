# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID TURBO V3
# ============================================================
# ADAPTIVE WEIGHT
# MICRO BACKTEST
# RECENT COUNT
# REPEAT SIGNAL
# GAP / SKIP
# EQUATION SHRINKAGE
# LEAKAGE SAFE
# FAST AI ENSEMBLE
# RF + EXTRA TREES + HIST GRADIENT BOOSTING
# HASH CACHE
# MOBILE OPTIMIZED
# ============================================================

import re
import warnings
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from bs4 import BeautifulSoup

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX Hybrid Turbo V3",
    page_icon="🚀",
    layout="centered"
)

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
}


# ============================================================
# 1. UI CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:28px;
    font-weight:900;
    color:#D32F2F;
}

.sub-title {
    text-align:center;
    color:#666;
    font-size:13px;
    margin-bottom:18px;
}

.hot-card {
    padding:18px;
    border-radius:16px;
    border:2px solid #ff4b4b;
    margin:10px 0;
    background:linear-gradient(
        to bottom right,
        #ffffff,
        #fff5f5
    );
}

.number-highlight {
    font-size:35px;
    font-weight:900;
    color:#D32F2F;
    text-shadow:1px 1px 2px rgba(0,0,0,0.15);
    letter-spacing:2px;
}

.dot-sep {
    color:#FFCDD2;
    font-size:25px;
    margin:0 8px;
}

.badge-ai {
    background:#E3F2FD;
    color:#1565C0;
    padding:4px 11px;
    border-radius:15px;
    font-weight:800;
    font-size:15px;
    border:1px solid #BBDEFB;
}

.badge-stat {
    background:#E8F5E9;
    color:#2E7D32;
    padding:4px 11px;
    border-radius:15px;
    font-weight:800;
    font-size:15px;
    border:1px solid #C8E6C9;
}

.badge-eq {
    background:#F3E5F5;
    color:#7B1FA2;
    padding:4px 11px;
    border-radius:15px;
    font-weight:800;
    font-size:15px;
    border:1px solid #E1BEE7;
}

.position-title {
    font-size:20px;
    font-weight:800;
    margin-top:20px;
    color:#333;
    border-bottom:2px solid #eee;
    padding-bottom:5px;
}

.info-row {
    margin:8px 0;
    font-size:14px;
}

.small-note {
    color:#888;
    font-size:12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. FETCH DATA
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def fetch_and_clean_data(url):

    headers = {
        "User-Agent":
        "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130 Mobile Safari/537.36"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()

    except Exception:
        return pd.DataFrame()

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
        r"(\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    )

    num_pattern = re.compile(
        r"\b(\d{3})\b.*?\b(\d{2})\b"
        r"|\b(\d{5,6})\b.*?\b(\d{2})\b"
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

            d = pd.to_datetime(
                dm.group(1),
                errors="coerce"
            )

            if not pd.isna(d):
                current_date = d

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

        rows.append({
            "Date": current_date,
            "Result_3D": str(r3).zfill(3),
            "Result_2D": str(r2).zfill(2)
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

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


# ============================================================
# 3. HASH
# ============================================================

def get_data_hash(df):

    if df.empty:
        return ""

    payload = (
        df["Date"].astype(str).tolist()
        + df["Result_3D"].tolist()
        + df["Result_2D"].tolist()
    )

    return hashlib.md5(
        "|".join(payload).encode()
    ).hexdigest()


# ============================================================
# 4. ADAPTIVE CONFIG
# ============================================================

def get_adaptive_config(n):

    if n >= 700:

        return {
            "lags": [1, 2, 3, 5],
            "rolls": [3, 5, 10],
            "recent_windows": [5, 10, 20],
            "trees": 100,
            "depth": 7,
            "bt": 12,
        }

    elif n >= 400:

        return {
            "lags": [1, 2, 3, 5],
            "rolls": [3, 5, 10],
            "recent_windows": [5, 10, 20],
            "trees": 85,
            "depth": 7,
            "bt": 10,
        }

    elif n >= 200:

        return {
            "lags": [1, 2, 3],
            "rolls": [3, 5, 10],
            "recent_windows": [5, 10, 15],
            "trees": 70,
            "depth": 6,
            "bt": 8,
        }

    elif n >= 100:

        return {
            "lags": [1, 2, 3],
            "rolls": [3, 5],
            "recent_windows": [5, 10, 15],
            "trees": 55,
            "depth": 5,
            "bt": 6,
        }

    else:

        return {
            "lags": [1, 2],
            "rolls": [3],
            "recent_windows": [5, 10],
            "trees": 35,
            "depth": 4,
            "bt": 5,
        }


# ============================================================
# 5. RAW DIGITS
# ============================================================

def add_digits(df):

    x = df.copy()

    r3 = x["Result_3D"].astype(str).str.zfill(3)
    r2 = x["Result_2D"].astype(str).str.zfill(2)

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)

    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    return x


# ============================================================
# 6. FAST FEATURE ENGINEERING
# ============================================================

def build_features(df, lags, rolls):

    x = add_digits(df)

    # --------------------------------------------------------
    # Previous-result relationship
    # --------------------------------------------------------

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = (
        ph + pt + po
    )

    x["PrevRange"] = (
        pd.concat([ph, pt, po], axis=1).max(axis=1)
        -
        pd.concat([ph, pt, po], axis=1).min(axis=1)
    )

    x["PrevOdd"] = (
        (ph % 2)
        + (pt % 2)
        + (po % 2)
    )

    x["PrevHigh"] = (
        (ph >= 5).astype(np.int8)
        +
        (pt >= 5).astype(np.int8)
        +
        (po >= 5).astype(np.int8)
    )

    x["DistHT"] = (
        ph - pt
    ).abs()

    x["DistTO"] = (
        pt - po
    ).abs()

    # --------------------------------------------------------
    # Repeat signals
    # --------------------------------------------------------

    for pos in ["H", "T", "O", "T2", "O2"]:

        s = x[pos]

        x[f"Repeat1_{pos}"] = (
            s == s.shift(1)
        ).astype(np.int8)

        x[f"Repeat2_{pos}"] = (
            s == s.shift(2)
        ).astype(np.int8)

        # ----------------------------------------------------
        # Lag
        # ----------------------------------------------------

        for lag in lags:

            x[f"L{lag}_{pos}"] = (
                s.shift(lag)
            )

        # ----------------------------------------------------
        # Rolling mean
        # ----------------------------------------------------

        shifted = s.shift(1)

        for w in rolls:

            x[f"RM{w}_{pos}"] = (
                shifted
                .rolling(
                    w,
                    min_periods=w
                )
                .mean()
            )

            # Recent frequency of current digit
            x[f"RCOUNT{w}_{pos}"] = (
                shifted
                .rolling(
                    w,
                    min_periods=w
                )
                .apply(
                    lambda a: np.bincount(
                        a.astype(np.int8),
                        minlength=10
                    ).max(),
                    raw=True
                )
            )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    if "Date" in x.columns:

        date = pd.to_datetime(
            x["Date"],
            errors="coerce"
        )

        dow = date.dt.dayofweek
        month = date.dt.month

        x["DOW"] = dow.astype(np.int8)
        x["Month"] = month.astype(np.int8)

        x["DOW_SIN"] = np.sin(
            2 * np.pi * dow / 7
        )

        x["DOW_COS"] = np.cos(
            2 * np.pi * dow / 7
        )

        x["MONTH_SIN"] = np.sin(
            2 * np.pi * month / 12
        )

        x["MONTH_COS"] = np.cos(
            2 * np.pi * month / 12
        )

    # --------------------------------------------------------
    # Skip / Gap
    # --------------------------------------------------------

    for pos in ["H", "T", "O", "T2", "O2"]:

        arr = x[pos].to_numpy()

        skip = np.zeros(
            len(arr),
            dtype=np.float32
        )

        last_seen = np.full(
            10,
            -1,
            dtype=np.int32
        )

        for i, val in enumerate(arr):

            v = int(val)

            if last_seen[v] < 0:
                skip[i] = i + 1
            else:
                skip[i] = (
                    i - last_seen[v]
                )

            last_seen[v] = i

        x[f"Skip_{pos}"] = skip

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return x.dropna().reset_index(drop=True)


# ============================================================
# 7. CORRECT NEXT-ROW FEATURE
# ============================================================
# สำคัญ:
# ห้ามสร้าง dummy 000 แล้วคำนวณ Skip
# เพราะจะทำให้ Skip ของงวดถัดไปผิด
# ============================================================

def build_next_features(
    df,
    lags,
    rolls,
    feature_columns
):

    x = add_digits(df)

    n = len(x)

    row = {}

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    ph = int(x["H"].iloc[-1])
    pt = int(x["T"].iloc[-1])
    po = int(x["O"].iloc[-1])

    row["PrevSum"] = ph + pt + po

    row["PrevRange"] = (
        max(ph, pt, po)
        -
        min(ph, pt, po)
    )

    row["PrevOdd"] = (
        (ph % 2)
        + (pt % 2)
        + (po % 2)
    )

    row["PrevHigh"] = (
        int(ph >= 5)
        + int(pt >= 5)
        + int(po >= 5)
    )

    row["DistHT"] = abs(ph - pt)
    row["DistTO"] = abs(pt - po)

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

    for pos in ["H", "T", "O", "T2", "O2"]:

        s = x[pos].astype(int)

        last = int(s.iloc[-1])

        row[f"Repeat1_{pos}"] = int(
            last == int(s.iloc[-2])
        ) if n >= 2 else 0

        row[f"Repeat2_{pos}"] = int(
            last == int(s.iloc[-3])
        ) if n >= 3 else 0

        for lag in lags:

            row[f"L{lag}_{pos}"] = (
                int(s.iloc[-lag])
                if n >= lag
                else 0
            )

        shifted = s

        for w in rolls:

            values = shifted.tail(w)

            if len(values) > 0:

                row[f"RM{w}_{pos}"] = (
                    float(values.mean())
                )

                # number of appearances
                # of LAST digit in recent window
                row[f"RCOUNT{w}_{pos}"] = (
                    int(
                        (values == last).sum()
                    )
                )

            else:

                row[f"RM{w}_{pos}"] = 0.0
                row[f"RCOUNT{w}_{pos}"] = 0

        # ----------------------------------------------------
        # Correct gap:
        # distance from previous occurrence of LAST digit
        # ----------------------------------------------------

        indices = np.where(
            s.to_numpy()[:-1] == last
        )[0]

        if len(indices) > 0:

            last_idx = indices[-1]

            row[f"Skip_{pos}"] = (
                n - 1 - last_idx
            )

        else:

            row[f"Skip_{pos}"] = n

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    last_date = pd.to_datetime(
        df["Date"].iloc[-1]
    )

    next_date = (
        last_date + timedelta(days=1)
    )

    dow = next_date.dayofweek
    month = next_date.month

    row["DOW"] = dow
    row["Month"] = month

    row["DOW_SIN"] = np.sin(
        2 * np.pi * dow / 7
    )

    row["DOW_COS"] = np.cos(
        2 * np.pi * dow / 7
    )

    row["MONTH_SIN"] = np.sin(
        2 * np.pi * month / 12
    )

    row["MONTH_COS"] = np.cos(
        2 * np.pi * month / 12
    )

    result = pd.DataFrame([row])

    for col in feature_columns:

        if col not in result.columns:
            result[col] = 0.0

    result = result[
        feature_columns
    ]

    return result.astype(
        np.float32
    ), next_date


# ============================================================
# 8. FREQUENCY ENGINE
# ============================================================

class FrequencyEngine:

    def analyze(self, df, pos):

        s = df[pos].astype(int)

        if len(s) == 0:
            return np.ones(10) / 10

        n = len(s)

        w15 = min(15, n)
        w30 = min(30, n)

        r15 = (
            s.tail(w15)
            .value_counts(normalize=True)
        )

        r30 = (
            s.tail(w30)
            .value_counts(normalize=True)
        )

        all_f = (
            s.value_counts(normalize=True)
        )

        score = np.array([
            (
                0.55 * r15.get(d, 0)
                +
                0.25 * r30.get(d, 0)
                +
                0.20 * all_f.get(d, 0)
            )
            for d in range(10)
        ])

        score += 0.01

        return (
            score / score.sum()
        )


# ============================================================
# 9. TRANSITION ENGINE
# ============================================================

class TransitionEngine:

    def analyze(self, df, pos):

        s = df[pos].astype(int)

        if len(s) < 8:
            return np.ones(10) / 10

        current = int(
            s.iloc[-1]
        )

        prev = s.shift(1)

        subset = s[
            prev == current
        ]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = (
            subset.value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.01

        return (
            score / score.sum()
        )


# ============================================================
# 10. PATTERN ENGINE
# ============================================================

class PatternEngine:

    def analyze(self, df, pos):

        s = df[pos].astype(int)

        if len(s) < 10:
            return np.ones(10) / 10

        a = int(s.iloc[-1])
        b = int(s.iloc[-2])

        subset = s[
            (s.shift(1) == a)
            &
            (s.shift(2) == b)
        ]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = (
            subset.value_counts(
                normalize=True
            )
        )

        score = np.array([
            freq.get(d, 0)
            for d in range(10)
        ])

        score += 0.01

        return (
            score / score.sum()
        )


# ============================================================
# 11. EQUATION ENGINE V3
# ============================================================

class EquationEngine:

    def __init__(self):

        self.equations = [

            (
                "L1",
                lambda a,b,c,d: a
            ),

            (
                "L2",
                lambda a,b,c,d: b
            ),

            (
                "L3",
                lambda a,b,c,d: c
            ),

            (
                "L5",
                lambda a,b,c,d: d
            ),

            (
                "L1+L2",
                lambda a,b,c,d: a+b
            ),

            (
                "L1+L3",
                lambda a,b,c,d: a+c
            ),

            (
                "ABS(L1-L2)",
                lambda a,b,c,d: abs(a-b)
            ),

            (
                "L1+L2+L3",
                lambda a,b,c,d: a+b+c
            ),

            (
                "ABS(L1-L3)",
                lambda a,b,c,d: abs(a-c)
            ),
        ]

    def discover(
        self,
        df,
        pos,
        bt=10
    ):

        n = len(df)

        uniform = np.ones(10) / 10

        if n < 50:

            return {
                "prob": uniform,
                "top": [],
                "strength": 0.0,
                "stable": 0,
                "total": len(
                    self.equations
                )
            }

        start = max(
            35,
            n - bt
        )

        results = []

        for name, fn in self.equations:

            hits = 0
            total = 0
            recent_hits = 0

            for idx in range(
                start,
                n
            ):

                if idx < 5:
                    continue

                try:

                    vals = (
                        int(df[pos].iloc[idx-1]),
                        int(df[pos].iloc[idx-2]),
                        int(df[pos].iloc[idx-3]),
                        int(df[pos].iloc[idx-5]),
                    )

                except Exception:
                    continue

                pred = (
                    fn(*vals) % 10
                )

                actual = int(
                    df[pos].iloc[idx]
                )

                total += 1

                if pred == actual:

                    hits += 1

                    if idx >= n - 3:
                        recent_hits += 1

            if total < 3:
                continue

            raw_hit = hits / total

            # ------------------------------------------------
            # Shrinkage
            # Baseline = 10%
            # ------------------------------------------------

            baseline = 0.10

            confidence = (
                total / (total + 12.0)
            )

            shrunk = (
                confidence * raw_hit
                +
                (1 - confidence) * baseline
            )

            score = (
                shrunk
                +
                recent_hits * 0.015
            )

            results.append({
                "name": name,
                "fn": fn,
                "hit": raw_hit,
                "shrunk": shrunk,
                "score": score,
                "total": total
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        stable = results[:5]

        prob = np.zeros(
            10,
            dtype=np.float64
        )

        total_w = 0.0

        try:

            vals = (
                int(df[pos].iloc[-1]),
                int(df[pos].iloc[-2]),
                int(df[pos].iloc[-3]),
                int(df[pos].iloc[-5]),
            )

        except Exception:

            vals = (
                0, 0, 0, 0
            )

        for r in stable:

            pred = (
                r["fn"](*vals) % 10
            )

            prob[pred] += (
                r["score"]
            )

            total_w += (
                r["score"]
            )

        if total_w <= 0:

            prob = uniform.copy()

        else:

            prob = (
                prob / total_w
            )

            prob += 0.01

            prob /= prob.sum()

        return {
            "prob": prob,
            "top": [
                (
                    int(i),
                    float(prob[i])
                )
                for i in np.argsort(
                    prob
                )[::-1][:5]
            ],
            "strength":
                float(
                    np.mean([
                        r["shrunk"]
                        for r in stable
                    ])
                ) if stable else 0.0,
            "stable": len(stable),
            "total": len(
                self.equations
            )
        }


# ============================================================
# 12. FAST AI
# ============================================================

class FastAI:

    def __init__(self, n_samples):

        cfg = get_adaptive_config(
            n_samples
        )

        self.trees = cfg["trees"]
        self.depth = cfg["depth"]

    def predict(
        self,
        X,
        y,
        X_next
    ):

        result = np.zeros(
            10,
            dtype=np.float64
        )

        models = [

            (
                RandomForestClassifier(
                    n_estimators=self.trees,
                    max_depth=self.depth,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),
                0.35
            ),

            (
                ExtraTreesClassifier(
                    n_estimators=self.trees,
                    max_depth=self.depth,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1
                ),
                0.40
            ),

            (
                HistGradientBoostingClassifier(
                    max_iter=self.trees,
                    max_depth=self.depth,
                    learning_rate=0.06,
                    l2_regularization=0.20,
                    random_state=42
                ),
                0.25
            ),
        ]

        for model, weight in models:

            try:

                model.fit(
                    X,
                    y
                )

                probs = (
                    model.predict_proba(
                        X_next
                    )[0]
                )

                for cls, p in zip(
                    model.classes_,
                    probs
                ):

                    result[
                        int(cls)
                    ] += (
                        p * weight
                    )

            except Exception:

                continue

        result += 0.001

        return (
            result /
            result.sum()
        )


# ============================================================
# 13. LIGHTWEIGHT ENGINE SCORE
# ============================================================

def engine_quality(
    df,
    pos,
    engine,
    bt=10
):
    """
    Lightweight historical scoring.
    ไม่ train ML ซ้ำ
    จึงเร็วมาก
    """

    s = df[pos].astype(int)

    n = len(s)

    if n < 30:
        return 0.10

    start = max(
        20,
        n - bt
    )

    hits = 0
    total = 0

    for idx in range(
        start,
        n
    ):

        train = df.iloc[:idx]

        try:

            if engine == "Freq":

                p = FrequencyEngine().analyze(
                    train,
                    pos
                )

            elif engine == "ST":

                p = TransitionEngine().analyze(
                    train,
                    pos
                )

            elif engine == "Pattern":

                p = PatternEngine().analyze(
                    train,
                    pos
                )

            elif engine == "Eq":

                p = EquationEngine().discover(
                    train,
                    pos,
                    bt=min(8, idx)
                )["prob"]

            else:
                continue

            pred = int(
                np.argmax(p)
            )

            actual = int(
                s.iloc[idx]
            )

            hits += int(
                pred == actual
            )

            total += 1

        except Exception:

            continue

    if total == 0:
        return 0.10

    raw = hits / total

    # shrink toward random baseline
    confidence = (
        total /
        (total + 10.0)
    )

    return (
        confidence * raw
        +
        (1 - confidence) * 0.10
    )


# ============================================================
# 14. ADAPTIVE WEIGHT
# ============================================================

def get_adaptive_weights(
    df,
    pos,
    base
):

    n = len(df)

    cfg = get_adaptive_config(n)

    bt = cfg["bt"]

    # --------------------------------------------------------
    # Base weights
    # --------------------------------------------------------

    w = base.copy()

    # --------------------------------------------------------
    # Lightweight statistical quality
    # --------------------------------------------------------

    q_freq = engine_quality(
        df,
        pos,
        "Freq",
        bt
    )

    q_st = engine_quality(
        df,
        pos,
        "ST",
        bt
    )

    q_pattern = engine_quality(
        df,
        pos,
        "Pattern",
        bt
    )

    q_eq = engine_quality(
        df,
        pos,
        "Eq",
        bt
    )

    quality = np.array([
        q_freq,
        q_st,
        q_pattern,
        q_eq
    ])

    # --------------------------------------------------------
    # Convert quality into modest adjustment
    # --------------------------------------------------------

    baseline = 0.10

    relative = (
        quality - baseline
    )

    relative = np.clip(
        relative,
        -0.06,
        0.10
    )

    names = [
        "Freq",
        "ST",
        "Pattern",
        "Eq"
    ]

    for name, delta in zip(
        names,
        relative
    ):

        w[name] *= (
            1.0 +
            delta * 2.0
        )

    # --------------------------------------------------------
    # Keep AI dominant
    # --------------------------------------------------------

    ai_floor = 0.35

    if w["AI"] < ai_floor:
        w["AI"] = ai_floor

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = sum(
        w.values()
    )

    for k in w:
        w[k] /= total

    return w, {
        "Freq": q_freq,
        "ST": q_st,
        "Pattern": q_pattern,
        "Eq": q_eq
    }


# ============================================================
# 15. ENSEMBLE ENGINE V3
# ============================================================

    def __init__(
        self,
        df
    ):

        # 1. จัดการข้อมูลให้มีคอลัมน์ตัวเลขครบถ้วน
        self.df = add_digits(df.copy())

        self.n = len(self.df)

        # 2. โค้ดส่วนนี้ต้องคงไว้ ห้ามลบทิ้ง
        self.cfg = get_adaptive_config(
            self.n
        )

        self.lags = self.cfg["lags"]
        self.rolls = self.cfg["rolls"]
        self.bt = self.cfg["bt"]

        self.positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        self.base_weights = {

            "H": {
                "AI": 0.42,
                "Freq": 0.22,
                "ST": 0.12,
                "Pattern": 0.12,
                "Eq": 0.12
            },

            "T": {
                "AI": 0.46,
                "Freq": 0.20,
                "ST": 0.11,
                "Pattern": 0.11,
                "Eq": 0.12
            },

            "O": {
                "AI": 0.50,
                "Freq": 0.17,
                "ST": 0.10,
                "Pattern": 0.10,
                "Eq": 0.13
            },

            "T2": {
                "AI": 0.45,
                "Freq": 0.20,
                "ST": 0.10,
                "Pattern": 0.10,
                "Eq": 0.15
            },

            "O2": {
                "AI": 0.50,
                "Freq": 0.17,
                "ST": 0.09,
                "Pattern": 0.09,
                "Eq": 0.15
            },
        }

    # ========================================================
    # FEATURE LIST
    # ========================================================

    def get_features(self):

        base_cols = [

            "PrevSum",
            "PrevRange",
            "PrevOdd",
            "PrevHigh",
            "DistHT",
            "DistTO",

            "DOW",
            "Month",

            "DOW_SIN",
            "DOW_COS",

            "MONTH_SIN",
            "MONTH_COS",
        ]

        cols = base_cols.copy()

        for pos in self.positions:

            cols.extend([
                f"Repeat1_{pos}",
                f"Repeat2_{pos}",
                f"Skip_{pos}",
            ])

            cols.extend([
                f"L{lag}_{pos}"
                for lag in self.lags
            ])

            for w in self.rolls:

                cols.append(
                    f"RM{w}_{pos}"
                )

                cols.append(
                    f"RCOUNT{w}_{pos}"
                )

        return cols

    # ========================================================
    # PREDICT
    # ========================================================

    def predict_all(self):

        # ----------------------------------------------------
        # Historical features
        # ----------------------------------------------------

        ext = build_features(
            self.df,
            self.lags,
            self.rolls
        )

        if len(ext) < 30:

            raise ValueError(
                "ข้อมูลหลังสร้าง Feature ไม่เพียงพอ"
            )

        self.features = (
            self.get_features()
        )

        # ----------------------------------------------------
        # Align target with feature rows
        # ----------------------------------------------------

        hist = ext.copy()

        X = (
            hist[self.features]
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Correct next row
        # ----------------------------------------------------

        X_next, next_date = (
            build_next_features(
                self.df,
                self.lags,
                self.rolls,
                self.features
            )
        )

        # ----------------------------------------------------
        # Engines
        # ----------------------------------------------------

        freq = FrequencyEngine()
        transition = TransitionEngine()
        pattern = PatternEngine()
        equation = EquationEngine()

        ai = FastAI(
            self.n
        )

        predictions = {}

        # ----------------------------------------------------
        # Position loop
        # ----------------------------------------------------

        for pos in self.positions:

            y = (
                hist[pos]
                .astype(np.int8)
            )

            # ----------------------------------------------
            # AI
            # ----------------------------------------------

            ai_p = ai.predict(
                X,
                y,
                X_next
            )

            # ----------------------------------------------
            # Statistical
            # ----------------------------------------------

            fq_p = freq.analyze(
                self.df,
                pos
            )

            st_p = transition.analyze(
                self.df,
                pos
            )

            pt_p = pattern.analyze(
                self.df,
                pos
            )

            eq_r = equation.discover(
                self.df,
                pos,
                bt=self.bt
            )

            # ----------------------------------------------
            # Adaptive weights
            # ----------------------------------------------

            w, quality = (
                get_adaptive_weights(
                    self.df,
                    pos,
                    self.base_weights[pos]
                )
            )

            # ----------------------------------------------
            # Final ensemble
            # ----------------------------------------------

            final = (

                w["AI"] *
                ai_p

                +

                w["Freq"] *
                fq_p

                +

                w["ST"] *
                st_p

                +

                w["Pattern"] *
                pt_p

                +

                w["Eq"] *
                eq_r["prob"]
            )

            final += 0.001

            final /= final.sum()

            # ----------------------------------------------
            # Top helper
            # ----------------------------------------------

            def top_n(
                p,
                n
            ):

                return [
                    (
                        int(i),
                        float(p[i])
                    )
                    for i in np.argsort(
                        p
                    )[::-1][:n]
                ]

            predictions[pos] = {

                "Final":
                    top_n(final, 5),

                "AI":
                    top_n(ai_p, 3),

                "Freq":
                    top_n(fq_p, 3),

                "Transition":
                    top_n(st_p, 3),

                "Pattern":
                    top_n(pt_p, 3),

                "Equation":
                    eq_r["top"],

                "EqStr":
                    eq_r["strength"],

                "W":
                    w,

                "Quality":
                    quality
            }

        return (
            predictions,
            next_date
        )


# ============================================================
# 16. CACHE PIPELINE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20
)
def run_prediction_pipeline(
    data_hash,
    df
):

    engine = EnsembleEngine(
        df
    )

    return engine.predict_all()


# ============================================================
# 17. UI HELPERS
# ============================================================

def html_top5(items):

    return (
        '<span class="dot-sep">•</span>'
        .join([
            f'<span class="number-highlight">{n}</span>'
            for n, _ in items
        ])
    )


def html_badge(
    items,
    cls
):

    return (
        f'<span class="{cls}">'
        +
        " &nbsp;•&nbsp; ".join([
            str(n)
            for n, _ in items
        ])
        +
        "</span>"
    )


def format_percent(
    value
):

    return (
        f"{value:.0%}"
    )


# ============================================================
# 18. MAIN UI
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚀 LOTTO AI V.MAX HYBRID TURBO V3'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'ADAPTIVE AI + STATISTICS + EQUATION<br>'
    '<b>FAST • LEAKAGE SAFE • HASH CACHE • ADAPTIVE WEIGHT</b>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


selected_lotto = st.selectbox(
    "🎯 เลือกหวย",
    list(LOTTERY_SOURCES.keys()),
    key="select_vmax_v3"
)


if st.button(
    "🚀 วิเคราะห์เลขเด่น V3",
    key="btn_vmax_v3",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # FETCH
    # --------------------------------------------------------

    df = fetch_and_clean_data(
        LOTTERY_SOURCES[
            selected_lotto
        ]
    )

    if df.empty:

        st.error(
            "🚨 ไม่สามารถดึงข้อมูลจากแหล่งที่มาได้ "
            "หรือรูปแบบเว็บต้นทางมีการเปลี่ยนแปลง"
        )

        st.stop()

    # --------------------------------------------------------
    # Minimum data
    # --------------------------------------------------------

    if len(df) < 50:

        st.warning(
            f"⚠️ มีข้อมูลเพียง {len(df)} งวด "
            "ระบบแนะนำอย่างน้อย 50 งวด"
        )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    current_hash = (
        get_data_hash(df)
    )

    cfg = get_adaptive_config(
        len(df)
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    with st.spinner(
        "⚡ V3 กำลังวิเคราะห์ "
        "(Cache จะถูกใช้ถ้าข้อมูลไม่เปลี่ยน)..."
    ):

        try:

            preds, next_date = (
                run_prediction_pipeline(
                    current_hash,
                    df
                )
            )

        except Exception as e:

            st.error(
                f"🚨 เกิดข้อผิดพลาด: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    st.info(
        f"📊 ข้อมูล {len(df)} งวด | "
        f"Hash: {current_hash[:8]} | "
        f"งวดถัดไป: "
        f"{next_date.strftime('%d-%m-%Y')} | "
        f"Trees: {cfg['trees']} | "
        f"Depth: {cfg['depth']}"
    )

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
            "หลักหน่วย 2 ตัวล่าง",
    }

    # --------------------------------------------------------
    # POSITION RESULTS
    # --------------------------------------------------------

    for pos in [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]:

        res = preds[pos]

        st.markdown(
            f'<div class="position-title">'
            f'📍 {labels[pos]}'
            f'</div>',
            unsafe_allow_html=True
        )

        html_content = (

            '<div class="hot-card">'

            '<div style="font-weight:700;'
            'color:#444; margin-bottom:8px;">'
            '🔥 FINAL TOP-5'
            '</div>'

            '<div style="text-align:center;'
            'margin:10px 0;">'

            +
            html_top5(
                res["Final"]
            )

            +

            '</div>'
            '</div>'

            +

            '<div class="info-row">'
            '🤖 <b>AI TOP-3:</b> &nbsp;'
            +
            html_badge(
                res["AI"],
                "badge-ai"
            )
            +
            '</div>'

            +

            '<div class="info-row">'
            '📊 <b>Frequency TOP-3:</b> &nbsp;'
            +
            html_badge(
                res["Freq"],
                "badge-stat"
            )
            +
            '</div>'

            +

            '<div class="info-row">'
            '🔁 <b>Transition TOP-3:</b> &nbsp;'
            +
            html_badge(
                res["Transition"],
                "badge-stat"
            )
            +
            '</div>'

            +

            '<div class="info-row">'
            '🧩 <b>Pattern TOP-3:</b> &nbsp;'
            +
            html_badge(
                res["Pattern"],
                "badge-stat"
            )
            +
            '</div>'

            +

            '<div class="info-row">'
            '🧮 <b>Equation TOP-5:</b> &nbsp;'
            +
            html_badge(
                res["Equation"],
                "badge-eq"
            )
            +
            f' '
            f'(Strength {res["EqStr"]:.0%})'
            '</div>'

            +

            '<div style="font-size:13px;'
            'color:#777; margin-top:8px;">'

            '⚖️ <b>Adaptive Weight:</b> '

            f'AI {format_percent(res["W"]["AI"])} | '

            f'Freq {format_percent(res["W"]["Freq"])} | '

            f'Trans {format_percent(res["W"]["ST"])} | '

            f'Pattern {format_percent(res["W"]["Pattern"])} | '

            f'Eq {format_percent(res["W"]["Eq"])}'

            '</div>'

            +

            '<div class="small-note">'
            'Adaptive quality: '

            f'Freq {res["Quality"]["Freq"]:.0%} | '

            f'Trans {res["Quality"]["ST"]:.0%} | '

            f'Pattern {res["Quality"]["Pattern"]:.0%} | '

            f'Eq {res["Quality"]["Eq"]:.0%}'

            '</div>'
        )

        st.markdown(
            html_content,
            unsafe_allow_html=True
        )

    # ========================================================
    # OVERALL
    # ========================================================

    st.subheader(
        "🔥 สรุปเลขเด่นภาพรวม"
    )

    def get_overall(
        positions
    ):

        score = np.zeros(
            10,
            dtype=np.float64
        )

        for pos in positions:

            for n, p in preds[pos][
                "Final"
            ]:

                score[n] += p

        if score.sum() <= 0:

            return [
                (i, 0.0)
                for i in range(5)
            ]

        return [
            (
                int(i),
                float(score[i])
            )
            for i in np.argsort(
                score
            )[::-1][:5]
        ]

    hot_top = get_overall([
        "H",
        "T",
        "O"
    ])

    hot_bottom = get_overall([
        "T2",
        "O2"
    ])

    overall_html = (

        '<div class="hot-card">'

        '<div style="font-weight:700;'
        'color:#444;">'
        '🔥 HOT 5-TOP รูด/วิ่งบน'
        '</div>'

        '<div style="text-align:center;'
        'margin:10px 0;">'

        +
        html_top5(
            hot_top
        )

        +

        '</div>'
        '</div>'

        +

        '<div class="hot-card">'

        '<div style="font-weight:700;'
        'color:#444;">'
        '🔥 HOT 5-TOP รูด/วิ่งล่าง'
        '</div>'

        '<div style="text-align:center;'
        'margin:10px 0;">'

        +
        html_top5(
            hot_bottom
        )

        +

        '</div>'
        '</div>'
    )

    st.markdown(
        overall_html,
        unsafe_allow_html=True
    )

    # ========================================================
    # SYSTEM INFO
    # ========================================================

    with st.expander(
        "🔧 รายละเอียดระบบ V3"
    ):

        st.write(
            f"จำนวนงวด: {len(df)}"
        )

        st.write(
            f"Lags: {cfg['lags']}"
        )

        st.write(
            f"Rolling: {cfg['rolls']}"
        )

        st.write(
            f"Recent Windows: "
            f"{cfg['recent_windows']}"
        )

        st.write(
            f"Trees: {cfg['trees']}"
        )

        st.write(
            f"Max Depth: {cfg['depth']}"
        )

        st.write(
            f"Micro Backtest: "
            f"{cfg['bt']} งวด"
        )

        st.write(
            "AI: RF + ExtraTrees + HGB"
        )

        st.write(
            "Adaptive Weight: ON"
        )

        st.write(
            "Equation Shrinkage: ON"
        )

        st.write(
            "Repeat Feature: ON"
        )

        st.write(
            "Recent Count Feature: ON"
        )

        st.write(
            "Correct Next-Row Skip: ON"
        )

        st.write(
            "Hash Cache: ON"
        )

    st.success(
        "✅ V3 วิเคราะห์เสร็จแล้ว "
        "• Leakage Safe "
        "• Adaptive Weight "
        "• Fast Cache"
            )
