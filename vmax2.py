# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID
# TOP-3 EVERY POSITION + TOP-5 OVERALL + 10-DRAW HISTORY
# ============================================================
#
# AI:
#   RandomForest
#   ExtraTrees
#   HistGradientBoosting
#
# STATISTICS:
#   Frequency
#   Transition
#   Pattern
#
# EQUATION:
#   L1, L2, L3, L5
#   + / -
#   ABS
#   weighted voting
#
# VALIDATION:
#   Strict Causal
#   Walk-Forward
#   Stability
#
# OUTPUT:
#   TOP-3 ทุกหลัก
#   TOP-5 บน
#   TOP-5 ล่าง
#   ประวัติย้อนหลัง 10 งวด
#
# DESIGN:
#   NO Calendar
#   NO Day
#   NO Month
#   NO Persistent Model
#   NO Joblib
#   Leakage Safe
#   Mobile Optimized
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
# 0. CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX",
    page_icon="🚀",
    layout="centered"
)


# ============================================================
# 1. SOURCES
# ============================================================

LOTTERY_SOURCES = {
    "1. หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "2. หวยธกส.": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "3. หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "4. หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "5. หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
    "6. หวยมาเลย์": "https://suksan18190.blogspot.com/2026/07/blog-post_10.html",
    "7. หวยหุ้นไทยเย็น": "https://suksan18190.blogspot.com/2026/07/blog-post_11.html",
    "8. หวยหุ้นนิเคอิบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_412.html",
    "9. หวยหุ้นฮั่งเส็งบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_229.html",
    "10. หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html"
}


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 30px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 3px;
    }

    .sub-title {
        font-size: 13px;
        text-align: center;
        color: #777;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .position-card {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 12px;
        margin: 7px 0;
        text-align: center;
    }

    .position-name {
        font-size: 14px;
        font-weight: 700;
        color: #666;
        margin-bottom: 5px;
    }

    .top-number {
        font-size: 27px;
        font-weight: 800;
        margin: 0 5px;
    }

    .prob {
        font-size: 11px;
        color: #888;
    }

    .overall-card {
        border: 1px solid #ddd;
        border-radius: 14px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
    }

    .overall-number {
        font-size: 29px;
        font-weight: 800;
        margin: 0 5px;
    }

    .history-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 11px;
        margin: 8px 0;
    }

    .history-date {
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .history-line {
        font-size: 13px;
        margin: 3px 0;
    }

    .real {
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. FETCH DATA
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def fetch_and_clean_data(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Linux; Android 10) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Mobile Safari/537.36"
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

    for pos in [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]:

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

        for lag in lags:

            x[f"L{lag}_{pos}"] = (
                s.shift(lag)
            )

        for w in rolls:

            x[f"RM{w}_{pos}"] = (
                s.shift(1)
                .rolling(
                    w,
                    min_periods=1
                )
                .mean()
            )

        # ----------------------------------------------------
        # Skip
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
# 5. FREQUENCY
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

        r15 = (
            s.tail(15)
            .value_counts(
                normalize=True
            )
        )

        r30 = (
            s.tail(30)
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
                r15.get(d, 0) * 0.55
                +
                r30.get(d, 0) * 0.30
                +
                all_f.get(d, 0) * 0.15
                for d in range(10)
            ]
        )

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 6. TRANSITION
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
            df[pos].shift(1)
            == current
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

        return (
            score /
            score.sum()
        )


# ============================================================
# 7. PATTERN
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
                df[pos].shift(1)
                == a
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

        return (
            score /
            score.sum()
        )


# ============================================================
# 8. EQUATION ENGINE
# ============================================================

class EquationEngine:

    def __init__(self):

        self.equations = (
            self._build_equations()
        )

    def _build_equations(self):

        eq = []

        eq += [
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
            )
        ]

        eq += [
            (
                "L1+L2",
                lambda a,b,c,d: a+b
            ),
            (
                "L1+L3",
                lambda a,b,c,d: a+c
            ),
            (
                "L1+L5",
                lambda a,b,c,d: a+d
            ),
            (
                "L2+L3",
                lambda a,b,c,d: b+c
            ),
            (
                "L2+L5",
                lambda a,b,c,d: b+d
            ),
            (
                "L3+L5",
                lambda a,b,c,d: c+d
            )
        ]

        eq += [
            (
                "L1-L2",
                lambda a,b,c,d: a-b
            ),
            (
                "L1-L3",
                lambda a,b,c,d: a-c
            ),
            (
                "L1-L5",
                lambda a,b,c,d: a-d
            ),
            (
                "L2-L3",
                lambda a,b,c,d: b-c
            ),
            (
                "L2-L5",
                lambda a,b,c,d: b-d
            ),
            (
                "L3-L5",
                lambda a,b,c,d: c-d
            )
        ]

        eq += [
            (
                "ABS(L1-L2)",
                lambda a,b,c,d:
                    abs(a-b)
            ),
            (
                "ABS(L1-L3)",
                lambda a,b,c,d:
                    abs(a-c)
            ),
            (
                "ABS(L1-L5)",
                lambda a,b,c,d:
                    abs(a-d)
            ),
            (
                "ABS(L2-L3)",
                lambda a,b,c,d:
                    abs(b-c)
            ),
            (
                "ABS(L2-L5)",
                lambda a,b,c,d:
                    abs(b-d)
            ),
            (
                "ABS(L3-L5)",
                lambda a,b,c,d:
                    abs(c-d)
            )
        ]

        eq += [
            (
                "L1+L2+L3",
                lambda a,b,c,d:
                    a+b+c
            ),
            (
                "L1+L3+L5",
                lambda a,b,c,d:
                    a+c+d
            ),
            (
                "L1+L2+L5",
                lambda a,b,c,d:
                    a+b+d
            )
        ]

        eq += [
            (
                "2L1+L2",
                lambda a,b,c,d:
                    2*a+b
            ),
            (
                "L1+2L2",
                lambda a,b,c,d:
                    a+2*b
            ),
            (
                "2L1+L3",
                lambda a,b,c,d:
                    2*a+c
            ),
            (
                "L1+2L3",
                lambda a,b,c,d:
                    a+2*c
            ),
            (
                "2L1+L5",
                lambda a,b,c,d:
                    2*a+d
            ),
            (
                "L1+2L5",
                lambda a,b,c,d:
                    a+2*d
            )
        ]

        return eq

    def _lags(
        self,
        df,
        pos,
        idx
    ):

        if idx < 5:
            return None

        return (
            int(df[pos].iloc[idx-1]),
            int(df[pos].iloc[idx-2]),
            int(df[pos].iloc[idx-3]),
            int(df[pos].iloc[idx-5])
        )

    def _predict(
        self,
        fn,
        vals
    ):

        try:

            return int(
                fn(*vals)
            ) % 10

        except:

            return -1

    def discover(
        self,
        df,
        pos,
        bt=10
    ):

        n = len(df)

        if n < 50:

            return {
                "prob":
                    np.ones(10) / 10,
                "strength": 0.0
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
            recent_total = 0

            for idx in range(
                start,
                n
            ):

                vals = self._lags(
                    df,
                    pos,
                    idx
                )

                if vals is None:
                    continue

                pred = self._predict(
                    fn,
                    vals
                )

                if pred < 0:
                    continue

                actual = int(
                    df[pos].iloc[idx]
                )

                total += 1

                if pred == actual:

                    hits += 1

                    if idx >= n - 5:
                        recent_hits += 1

                if idx >= n - 5:
                    recent_total += 1

            if total == 0:
                continue

            hit = hits / total

            recent = (
                recent_hits /
                recent_total
                if recent_total > 0
                else 0
            )

            if (
                hit >= 0.10
                and
                recent >= 0.10
            ):

                score = (
                    0.70 * hit
                    +
                    0.30 * recent
                )

                results.append(
                    (
                        name,
                        fn,
                        hit,
                        recent,
                        score
                    )
                )

        if not results:

            return {
                "prob":
                    np.ones(10) / 10,
                "strength": 0.0
            }

        results.sort(
            key=lambda x: x[4],
            reverse=True
        )

        selected = results[:8]

        vals = self._lags(
            df,
            pos,
            n
        )

        if vals is None:

            return {
                "prob":
                    np.ones(10) / 10,
                "strength": 0.0
            }

        prob = np.zeros(10)

        total_weight = 0

        strengths = []

        for name, fn, hit, recent, score in selected:

            pred = self._predict(
                fn,
                vals
            )

            if pred < 0:
                continue

            weight = (
                0.50 + score
            )

            prob[pred] += weight

            total_weight += weight

            strengths.append(hit)

        if total_weight <= 0:

            prob = np.ones(10) / 10

        else:

            prob /= total_weight

            prob += 0.01

            prob /= prob.sum()

        return {
            "prob": prob,
            "strength":
                float(
                    np.mean(strengths)
                    if strengths
                    else 0
                )
        }


# ============================================================
# 9. AI
# ============================================================

class FastAI:

    def __init__(
        self,
        trees=45
    ):

        self.trees = trees

        self.weights = (
            0.35,
            0.35,
            0.30
        )

    def predict(
        self,
        X,
        y,
        X_next
    ):

        rf_w, et_w, hgb_w = (
            self.weights
        )

        result = np.zeros(10)

        total_w = 0

        # ----------------------------------------------------
        # RF
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=self.trees,
            max_depth=6,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X, y)

        proba = model.predict_proba(
            X_next
        )[0]

        for c, p in zip(
            model.classes_,
            proba
        ):

            result[int(c)] += (
                p * rf_w
            )

        total_w += rf_w

        # ----------------------------------------------------
        # ExtraTrees
        # ----------------------------------------------------

        model = ExtraTreesClassifier(
            n_estimators=self.trees,
            max_depth=6,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=43
        )

        model.fit(X, y)

        proba = model.predict_proba(
            X_next
        )[0]

        for c, p in zip(
            model.classes_,
            proba
        ):

            result[int(c)] += (
                p * et_w
            )

        total_w += et_w

        # ----------------------------------------------------
        # HGB
        # ----------------------------------------------------

        model = HistGradientBoostingClassifier(
            max_iter=70,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=3,
            l2_regularization=0.5,
            random_state=44
        )

        model.fit(X, y)

        proba = model.predict_proba(
            X_next
        )[0]

        for c, p in zip(
            model.classes_,
            proba
        ):

            result[int(c)] += (
                p * hgb_w
            )

        total_w += hgb_w

        if total_w <= 0:
            return np.ones(10) / 10

        result /= total_w

        result += 0.001

        return (
            result /
            result.sum()
        )


# ============================================================
# 10. ENSEMBLE
# ============================================================

class EnsembleEngine:

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

        self.target_dow = (
            target_dow
        )

        n = len(df)

        # ----------------------------------------------------
        # SPEED CONFIG
        # ----------------------------------------------------

        if n >= 700:

            self.trees = 45
            self.bt = 10

        elif n >= 400:

            self.trees = 40
            self.bt = 9

        else:

            self.trees = 35
            self.bt = 8

        self.lags = [
            1, 2, 3, 5
        ]

        self.rolls = [
            3, 5, 10
        ]

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
            EquationEngine()
        )

        self.ai = FastAI(
            self.trees
        )

        # ----------------------------------------------------
        # BASE WEIGHT
        # ----------------------------------------------------

        self.base_weights = {

            "AI": 0.50,

            "Freq": 0.18,

            "ST": 0.12,

            "Pattern": 0.08,

            "Eq": 0.12
        }


    # ========================================================
    # FAST WALK-FORWARD
    # ========================================================

    def backtest(
        self,
        pos,
        X,
        df_hist
    ):

        n = len(X)

        if n < 45:

            return (
                self.base_weights.copy()
            )

        start = max(
            35,
            n - self.bt
        )

        scores = {
            "AI": 0,
            "Freq": 0,
            "ST": 0,
            "Pattern": 0,
            "Eq": 0
        }

        total_decay = 0

        # ----------------------------------------------------
        # ใช้ TOP-5 validation
        # ----------------------------------------------------

        for step, idx in enumerate(
            range(start, n)
        ):

            decay = (
                1.08 ** step
            )

            total_decay += decay

            Xtr = X.iloc[:idx]

            ytr = (
                df_hist[pos]
                .iloc[:idx]
            )

            xt = X.iloc[
                [idx]
            ]

            actual = int(
                df_hist[pos]
                .iloc[idx]
            )

            # ------------------------------------------------
            # AI PROXY
            # ------------------------------------------------

            try:

                proxy = ExtraTreesClassifier(
                    n_estimators=8,
                    max_depth=5,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=200 + step
                )

                proxy.fit(
                    Xtr,
                    ytr
                )

                proba = proxy.predict_proba(
                    xt
                )[0]

                tmp = np.zeros(10)

                for c, p in zip(
                    proxy.classes_,
                    proba
                ):

                    tmp[int(c)] = p

                if actual in np.argsort(
                    tmp
                )[::-1][:5]:

                    scores["AI"] += decay

            except:
                pass

            # ------------------------------------------------
            # FREQUENCY
            # ------------------------------------------------

            hist = df_hist.iloc[
                :idx
            ]

            fq = self.freq.analyze(
                hist,
                pos
            )

            if actual in np.argsort(
                fq
            )[::-1][:5]:

                scores["Freq"] += decay

            # ------------------------------------------------
            # TRANSITION
            # ------------------------------------------------

            stp = (
                self.transition.analyze(
                    hist,
                    pos
                )
            )

            if actual in np.argsort(
                stp
            )[::-1][:5]:

                scores["ST"] += decay

            # ------------------------------------------------
            # PATTERN
            # ------------------------------------------------

            ptn = (
                self.pattern.analyze(
                    hist,
                    pos
                )
            )

            if actual in np.argsort(
                ptn
            )[::-1][:5]:

                scores["Pattern"] += decay

            # ------------------------------------------------
            # EQUATION
            # ------------------------------------------------

            eq_result = (
                self.equation.discover(
                    hist,
                    pos,
                    bt=min(
                        8,
                        max(
                            5,
                            idx - 35
                        )
                    )
                )
            )

            eq = eq_result["prob"]

            if actual in np.argsort(
                eq
            )[::-1][:5]:

                scores["Eq"] += decay

        if total_decay <= 0:

            return (
                self.base_weights.copy()
            )

        accuracy = {
            k:
            scores[k] /
            total_decay

            for k in scores
        }

        # ----------------------------------------------------
        # ADAPTIVE WEIGHT
        # ----------------------------------------------------

        weighted = {}

        for k in self.base_weights:

            adaptive = max(
                0.10,
                accuracy[k]
            )

            weighted[k] = (
                self.base_weights[k]
                *
                (
                    0.35
                    +
                    0.65 *
                    adaptive
                )
            )

        total = sum(
            weighted.values()
        )

        if total <= 0:

            return (
                self.base_weights.copy()
            )

        weights = {
            k:
            v / total

            for k, v
            in weighted.items()
        }

        # ----------------------------------------------------
        # จำกัด AI สูงสุด 58%
        # ----------------------------------------------------

        if weights["AI"] > 0.58:

            diff = (
                weights["AI"]
                - 0.58
            )

            weights["AI"] = 0.58

            other_sum = sum(
                v
                for k, v
                in weights.items()
                if k != "AI"
            )

            if other_sum > 0:

                for k in weights:

                    if k != "AI":

                        weights[k] += (
                            diff *
                            (
                                weights[k]
                                /
                                other_sum
                            )
                        )

        return weights


    # ========================================================
    # PROCESS POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next
    ):

        weights = self.backtest(
            pos,
            X,
            hist
        )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        ai = self.ai.predict(
            X,
            hist[pos],
            X_next
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        fq = self.freq.analyze(
            hist,
            pos
        )

        stp = (
            self.transition.analyze(
                hist,
                pos
            )
        )

        ptn = (
            self.pattern.analyze(
                hist,
                pos
            )
        )

        # ----------------------------------------------------
        # Equation
        # ----------------------------------------------------

        eq_result = (
            self.equation.discover(
                hist,
                pos,
                bt=self.bt
            )
        )

        eq = eq_result["prob"]

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        final = (

            weights["AI"] * ai

            +

            weights["Freq"] * fq

            +

            weights["ST"] * stp

            +

            weights["Pattern"] * ptn

            +

            weights["Eq"] * eq
        )

        final += 0.001

        final /= final.sum()

        top3 = [
            (
                int(i),
                float(final[i])
            )

            for i in np.argsort(
                final
            )[::-1][:3]
        ]

        return {
            "Prob": final,
            "Top3": top3,
            "Weights": weights
        }


    # ========================================================
    # PREDICT CURRENT
    # ========================================================

    def predict_all(
        self,
        progress_bar=None,
        status_text=None
    ):

        last_date = (
            self.df["Date"].iloc[-1]
        )

        # ----------------------------------------------------
        # Next Date
        # ----------------------------------------------------

        if self.target_dow is not None:

            days = (
                self.target_dow
                -
                last_date.dayofweek
            ) % 7

            if days <= 0:
                days = 7

        else:

            if len(self.df) >= 2:

                days = max(
                    1,
                    (
                        last_date
                        -
                        self.df[
                            "Date"
                        ].iloc[-2]
                    ).days
                )

            else:

                days = 7

        next_date = (
            last_date
            +
            timedelta(days=days)
        )

        # ----------------------------------------------------
        # Dummy target row
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

        if status_text:

            status_text.markdown(
                "🧠 **กำลังสร้าง Features...**"
            )

        ext = build_features(
            ext,
            self.lags,
            self.rolls
        )

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

        predictions = {}

        positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        for i, pos in enumerate(
            positions
        ):

            if status_text:

                status_text.markdown(
                    f"⚙️ **วิเคราะห์ {i+1}/5:** {pos}"
                )

            predictions[pos] = (
                self.process_position(
                    pos,
                    hist,
                    X,
                    X_next
                )
            )

            if progress_bar:

                progress_bar.progress(
                    20 +
                    int(
                        (
                            (i + 1)
                            / 5
                        ) * 30
                    )
                )

        return (
            predictions,
            next_date
        )


    # ========================================================
    # HISTORY 10 DRAW
    # ========================================================

    def evaluate_history(
        self,
        progress_bar=None,
        status_text=None
    ):

        n_total = len(
            self.df
        )

        num_records = min(
            10,
            n_total - 45
        )

        if num_records <= 0:
            return []

        start_idx = (
            n_total
            -
            num_records
        )

        records = []

        # ----------------------------------------------------
        # Feature data
        # ----------------------------------------------------

        ext = build_features(
            self.df,
            self.lags,
            self.rolls
        )

        positions = [
            "H",
            "T",
            "O",
            "T2",
            "O2"
        ]

        labels = {
            "H":
                "ร้อยบน",

            "T":
                "สิบบน",

            "O":
                "หน่วยบน",

            "T2":
                "สิบล่าง",

            "O2":
                "หน่วยล่าง"
        }

        for step, i in enumerate(
            range(
                start_idx,
                n_total
            )
        ):

            if status_text:

                status_text.markdown(
                    f"🕰️ **ประวัติย้อนหลัง:** "
                    f"{step+1}/{num_records}"
                )

            if progress_bar:

                progress_bar.progress(
                    55 +
                    int(
                        (
                            (step + 1)
                            /
                            num_records
                        )
                        * 40
                    )
                )

            # ------------------------------------------------
            # IMPORTANT:
            # ใช้ข้อมูลก่อนงวดนั้นเท่านั้น
            # ------------------------------------------------

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

            preds = {}

            for pos in positions:

                preds[pos] = (
                    self.process_position(
                        pos,
                        hist,
                        X,
                        X_next
                    )
                )

            # ------------------------------------------------
            # Actual
            # ------------------------------------------------

            actual_row = (
                self.df.iloc[i]
            )

            actual_3d = str(
                actual_row[
                    "Result_3D"
                ]
            ).zfill(3)

            actual_2d = str(
                actual_row[
                    "Result_2D"
                ]
            ).zfill(2)

            actual_digits = {
                "H":
                    int(actual_3d[0]),

                "T":
                    int(actual_3d[1]),

                "O":
                    int(actual_3d[2]),

                "T2":
                    int(actual_2d[0]),

                "O2":
                    int(actual_2d[1])
            }

            # ------------------------------------------------
            # Position TOP-3
            # ------------------------------------------------

            position_data = {}

            for pos in positions:

                top3 = preds[
                    pos
                ]["Top3"]

                nums = [
                    n
                    for n, p
                    in top3
                ]

                actual = (
                    actual_digits[pos]
                )

                position_data[pos] = {
                    "nums": nums,
                    "actual": actual,
                    "hit":
                        actual in nums
                }

            # ------------------------------------------------
            # Overall TOP-5
            # ------------------------------------------------

            upper_score = (
                preds["H"]["Prob"]
                +
                preds["T"]["Prob"]
                +
                preds["O"]["Prob"]
            ) / 3

            lower_score = (
                preds["T2"]["Prob"]
                +
                preds["O2"]["Prob"]
            ) / 2

            upper_top5 = [
                int(i)
                for i in np.argsort(
                    upper_score
                )[::-1][:5]
            ]

            lower_top5 = [
                int(i)
                for i in np.argsort(
                    lower_score
                )[::-1][:5]
            ]

            upper_actuals = [
                int(x)
                for x in actual_3d
            ]

            lower_actuals = [
                int(x)
                for x in actual_2d
            ]

            upper_hit = any(
                x in upper_top5
                for x in upper_actuals
            )

            lower_hit = any(
                x in lower_top5
                for x in lower_actuals
            )

            records.append(
                {
                    "Date":
                        actual_row[
                            "Date"
                        ],

                    "Result_3D":
                        actual_3d,

                    "Result_2D":
                        actual_2d,

                    "Positions":
                        position_data,

                    "UpperTop5":
                        upper_top5,

                    "LowerTop5":
                        lower_top5,

                    "UpperHit":
                        upper_hit,

                    "LowerHit":
                        lower_hit
                }
            )

        return records[::-1]


# ============================================================
# 11. DISPLAY HELPERS
# ============================================================

def format_top3(items):

    return " • ".join(
        str(n)
        for n, p in items
    )


def format_top5(nums):

    return " • ".join(
        str(n)
        for n in nums
    )


def top3_html(items):

    return " ".join(
        f"""
        <span class="top-number">
            {n}
        </span>
        """
        for n, p in items
    )


def top5_html(nums):

    return " ".join(
        f"""
        <span class="overall-number">
            {n}
        </span>
        """
        for n in nums
    )


def hit_mark(hit):

    if hit:

        return (
            "✅ <b>เข้า</b>"
        )

    return (
        "❌ หลุด"
    )


# ============================================================
# 12. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🚀 LOTTO AI V.MAX
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        HYBRID AI + STATISTICS + EQUATION<br>
        TOP-3 EVERY POSITION • TOP-5 OVERALL • 10-DRAW HISTORY
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 13. SELECT
# ============================================================

c1, c2 = st.columns(2)

selected_lotto = c1.selectbox(
    "🎯 เลือกหวย",
    list(
        LOTTERY_SOURCES.keys()
    )
)

day_options = {

    "อัตโนมัติ":
        None,

    "วันจันทร์":
        0,

    "วันอังคาร":
        1,

    "วันพุธ":
        2,

    "วันพฤหัสบดี":
        3,

    "วันศุกร์":
        4,

    "วันเสาร์":
        5,

    "วันอาทิตย์":
        6
}

day_label = c2.selectbox(
    "📅 วันออกรางวัล",
    list(
        day_options.keys()
    )
)


# ============================================================
# 14. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์เลขเด่น",
    type="primary",
    use_container_width=True
):

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    status_text.markdown(
        "⏳ **กำลังโหลดข้อมูลล่าสุด...**"
    )

    df = fetch_and_clean_data(
        LOTTERY_SOURCES[
            selected_lotto
        ]
    )

    if df.empty:

        status_text.error(
            "❌ ไม่สามารถดึงข้อมูลได้"
        )

        st.stop()

    if len(df) < 50:

        status_text.error(
            f"❌ ต้องมีอย่างน้อย 50 งวด "
            f"(พบ {len(df)} งวด)"
        )

        st.stop()

    progress_bar.progress(
        10
    )

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = EnsembleEngine(
        df,
        selected_lotto,
        day_options[
            day_label
        ]
    )

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    predictions, next_date = (
        engine.predict_all(
            progress_bar,
            status_text
        )
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = (
        engine.evaluate_history(
            progress_bar,
            status_text
        )
    )

    progress_bar.progress(
        100
    )

    status_text.empty()
    progress_bar.empty()

    days = [
        "จันทร์",
        "อังคาร",
        "พุธ",
        "พฤหัสบดี",
        "ศุกร์",
        "เสาร์",
        "อาทิตย์"
    ]

    # ========================================================
    # TARGET DATE
    # ========================================================

    st.divider()

    st.info(
        f"📅 งวดเป้าหมาย: "
        f"วัน{days[next_date.dayofweek]} "
        f"{next_date.strftime('%d-%m-%Y')} "
        f"| ข้อมูล {len(df)} งวด"
    )


    # ========================================================
    # TOP-3 EVERY POSITION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 เลขเด่น TOP-3 ทุกหลัก'
        '</div>',
        unsafe_allow_html=True
    )

    position_labels = {

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

    for pos in [
        "H",
        "T",
        "O",
        "T2",
        "O2"
    ]:

        res = predictions[
            pos
        ]

        st.markdown(
            f"""
            <div class="position-card">

                <div class="position-name">
                    {position_labels[pos]}
                </div>

                <div>
                    {top3_html(res["Top3"])}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # OVERALL TOP-5
    # ========================================================

    upper_score = (
        predictions["H"]["Prob"]
        +
        predictions["T"]["Prob"]
        +
        predictions["O"]["Prob"]
    ) / 3

    lower_score = (
        predictions["T2"]["Prob"]
        +
        predictions["O2"]["Prob"]
    ) / 2

    upper_top5 = [
        int(i)
        for i in np.argsort(
            upper_score
        )[::-1][:5]
    ]

    lower_top5 = [
        int(i)
        for i in np.argsort(
            lower_score
        )[::-1][:5]
    ]


    st.markdown(
        '<div class="section-title">'
        '🔥 สรุปเลขเด่นภาพรวม'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="overall-card">

            <div style="
                font-weight:800;
                margin-bottom:8px;
            ">
                🔴 เลขเด่นบน TOP-5
            </div>

            <div>
                {top5_html(upper_top5)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="overall-card">

            <div style="
                font-weight:800;
                margin-bottom:8px;
            ">
                🔵 เลขเด่นล่าง TOP-5
            </div>

            <div>
                {top5_html(lower_top5)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # HISTORY 10 DRAW
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📜 ประวัติย้อนหลัง 10 งวด'
        '</div>',
        unsafe_allow_html=True
    )

    if not history:

        st.warning(
            "ไม่มีข้อมูลย้อนหลังเพียงพอ"
        )

    else:

        for rec in history:

            date_str = (
                rec["Date"]
                .strftime(
                    "%d-%m-%Y"
                )
            )

            # ------------------------------------------------
            # Position rows
            # ------------------------------------------------

            pos_html = ""

            for pos in [
                "H",
                "T",
                "O",
                "T2",
                "O2"
            ]:

                pdata = (
                    rec[
                        "Positions"
                    ][pos]
                )

                nums = " • ".join(
                    str(x)
                    for x in pdata[
                        "nums"
                    ]
                )

                actual = (
                    pdata["actual"]
                )

                mark = hit_mark(
                    pdata["hit"]
                )

                pos_html += f"""
                <div class="history-line">
                    <b>
                        {position_labels[pos]}
                    </b>
                    :
                    {nums}
                    |
                    จริง <span class="real">
                        {actual}
                    </span>
                    {mark}
                </div>
                """

            # ------------------------------------------------
            # Overall
            # ------------------------------------------------

            upper_nums = (
                format_top5(
                    rec["UpperTop5"]
                )
            )

            lower_nums = (
                format_top5(
                    rec["LowerTop5"]
                )
            )

            upper_mark = hit_mark(
                rec["UpperHit"]
            )

            lower_mark = hit_mark(
                rec["LowerHit"]
            )

            st.markdown(
                f"""
                <div class="history-card">

                    <div class="history-date">
                        📅 {date_str}
                    </div>

                    <div class="history-line">
                        <b>ผลจริงบน:</b>
                        {rec["Result_3D"]}
                    </div>

                    <div class="history-line">
                        <b>ผลจริงล่าง:</b>
                        {rec["Result_2D"]}
                    </div>

                    <hr>

                    {pos_html}

                    <hr>

                    <div class="history-line">
                        🔴 <b>บน TOP-5:</b>
                        {upper_nums}
                        {upper_mark}
                    </div>

                    <div class="history-line">
                        🔵 <b>ล่าง TOP-5:</b>
                        {lower_nums}
                        {lower_mark}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # HISTORY SUMMARY
    # ========================================================

    if history:

        st.markdown(
            '<div class="section-title">'
            '📊 สรุปผลงานย้อนหลัง 10 งวด'
            '</div>',
            unsafe_allow_html=True
        )

        pos_hits = {
            p: sum(
                1
                for r in history
                if r["Positions"][p]["hit"]
            )
            for p in [
                "H",
                "T",
                "O",
                "T2",
                "O2"
            ]
        }

        upper_hits = sum(
            r["UpperHit"]
            for r in history
        )

        lower_hits = sum(
            r["LowerHit"]
            for r in history
        )

        total = len(history)

        st.markdown(
            f"""
            <div class="history-card">

                <div class="history-line">
                    🔢 <b>ร้อยบน TOP-3:</b>
                    {pos_hits["H"]}/{total}
                </div>

                <div class="history-line">
                    🔢 <b>สิบบน TOP-3:</b>
                    {pos_hits["T"]}/{total}
                </div>

                <div class="history-line">
                    🔢 <b>หน่วยบน TOP-3:</b>
                    {pos_hits["O"]}/{total}
                </div>

                <div class="history-line">
                    🔢 <b>สิบล่าง TOP-3:</b>
                    {pos_hits["T2"]}/{total}
                </div>

                <div class="history-line">
                    🔢 <b>หน่วยล่าง TOP-3:</b>
                    {pos_hits["O2"]}/{total}
                </div>

                <hr>

                <div class="history-line">
                    🔴 <b>บน TOP-5:</b>
                    {upper_hits}/{total}
                </div>

                <div class="history-line">
                    🔵 <b>ล่าง TOP-5:</b>
                    {lower_hits}/{total}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FOOTER
    # ========================================================

    st.success(
        "✅ วิเคราะห์เสร็จสิ้น"
    )

    st.caption(
        "Strict Causal • Leakage-safe • "
        "Walk-Forward • Dynamic Ensemble"
    )

    st.caption(
        "ไม่มี Persistent Model และไม่มี Joblib "
        "โมเดลคำนวณใหม่ทุกครั้งที่กดวิเคราะห์"
    )

    st.caption(
        "⚠️ TOP-3/TOP-5 เป็นคะแนนจากโมเดลและสถิติ "
        "ไม่ใช่การรับประกันผลรางวัลจริง"
    )
