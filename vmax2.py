# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID SPEED-STABLE
# ============================================================
#
# OUTPUT:
#   ✅ เลขเด่น TOP-3 แยกแต่ละหลัก
#   ✅ สรุปเลขเด่นบน TOP-5
#   ✅ สรุปเลขเด่นล่าง TOP-5
#   ❌ ไม่แสดง AI / Frequency / Transition / Pattern / Equation แยก
#
# ENGINE:
#   AI:
#       RandomForest
#       ExtraTrees
#       HistGradientBoosting
#
#   STATISTICS:
#       Frequency
#       Transition
#       Pattern
#
#   EQUATION:
#       24 Core Equations
#       Strict causal
#       Stability filtering
#
#   ENSEMBLE:
#       Walk-Forward
#       Recent Skill
#       Stability
#       Consensus Bonus
#
# DESIGN:
#   NO Calendar
#   NO Day / Month
#   NO fixed equation
#   NO persistent model
#   NO joblib
#   Leakage-safe
#   Mobile optimized
#
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
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX Speed-Stable",
    page_icon="🚀",
    layout="centered"
)


# ============================================================
# 1. LOTTERY SOURCES
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
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 13px;
        text-align: center;
        color: #777;
        margin-bottom: 15px;
    }

    .position-title {
        font-size: 18px;
        font-weight: 800;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    .hot-card {
        padding: 14px;
        border-radius: 14px;
        border: 1px solid #ddd;
        margin: 8px 0;
    }

    .top3-number {
        font-size: 28px;
        font-weight: 800;
        padding: 5px 12px;
    }

    .top5-number {
        font-size: 28px;
        font-weight: 800;
        padding: 5px 10px;
    }

    .dot-sep {
        color: #999;
        margin: 0 4px;
    }

    .confidence {
        font-size: 13px;
        color: #888;
        text-align: center;
        margin-top: 5px;
    }

    .small-info {
        font-size: 12px;
        color: #888;
        text-align: center;
        margin-top: 6px;
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

        current_date = pd.Timestamp(datetime.now())

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

                except Exception:
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

            rows.append({
                "Date": current_date,
                "Result_3D": str(r3).zfill(3),
                "Result_2D": str(r2).zfill(2)
            })

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
    # Previous 3D structure
    # --------------------------------------------------------

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = (
        ph + pt + po
    )

    x["PrevOdd"] = (
        (ph % 2) +
        (pt % 2) +
        (po % 2)
    )

    x["DistHT"] = (
        ph - pt
    ).abs()

    x["DistTO"] = (
        pt - po
    ).abs()

    # --------------------------------------------------------
    # Position features
    # --------------------------------------------------------

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
        ).astype(np.float32)

        x[f"High_{pos}"] = (
            prev >= 5
        ).astype(np.float32)

        x[f"Prime_{pos}"] = (
            prev.isin([2, 3, 5, 7])
        ).astype(np.float32)

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

    def analyze(self, df, pos):

        s = df[pos].astype(int)

        if len(s) == 0:
            return np.ones(10) / 10

        r15 = (
            s.tail(15)
            .value_counts(normalize=True)
        )

        r30 = (
            s.tail(30)
            .value_counts(normalize=True)
        )

        all_f = (
            s.value_counts(normalize=True)
        )

        score = np.array([

            r15.get(d, 0) * 0.55 +
            r30.get(d, 0) * 0.30 +
            all_f.get(d, 0) * 0.15

            for d in range(10)

        ])

        score += 0.01

        return (
            score /
            score.sum()
        )


# ============================================================
# 6. TRANSITION
# ============================================================

class TransitionEngine:

    def analyze(self, df, pos):

        if len(df) < 6:
            return np.ones(10) / 10

        s = df[pos].astype(int)

        current = int(
            s.iloc[-1]
        )

        previous = s.shift(1)

        mask = (
            previous == current
        )

        subset = s[mask]

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
            score /
            score.sum()
        )


# ============================================================
# 7. PATTERN
# ============================================================

class PatternEngine:

    def analyze(self, df, pos):

        if len(df) < 7:
            return np.ones(10) / 10

        s = df[pos].astype(int)

        a = int(
            s.iloc[-1]
        )

        b = int(
            s.iloc[-2]
        )

        previous_1 = s.shift(1)
        previous_2 = s.shift(2)

        mask = (
            (previous_1 == a) &
            (previous_2 == b)
        )

        subset = s[mask]

        if len(subset) < 2:

            subset = s[
                previous_1 == a
            ]

        if len(subset) < 1:
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

        return [

            # Copy
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

            # Sum
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
            ),

            # Difference
            (
                "ABS(L1-L2)",
                lambda a,b,c,d: abs(a-b)
            ),
            (
                "ABS(L1-L3)",
                lambda a,b,c,d: abs(a-c)
            ),
            (
                "ABS(L1-L5)",
                lambda a,b,c,d: abs(a-d)
            ),
            (
                "ABS(L2-L3)",
                lambda a,b,c,d: abs(b-c)
            ),
            (
                "ABS(L2-L5)",
                lambda a,b,c,d: abs(b-d)
            ),
            (
                "ABS(L3-L5)",
                lambda a,b,c,d: abs(c-d)
            ),

            # Weighted
            (
                "2L1+L2",
                lambda a,b,c,d: 2*a+b
            ),
            (
                "L1+2L2",
                lambda a,b,c,d: a+2*b
            ),
            (
                "2L1+L3",
                lambda a,b,c,d: 2*a+c
            ),
            (
                "L1+2L3",
                lambda a,b,c,d: a+2*c
            ),
            (
                "2L1+L5",
                lambda a,b,c,d: 2*a+d
            ),
            (
                "L1+2L5",
                lambda a,b,c,d: a+2*d
            )
        ]

    def discover(
        self,
        df,
        pos,
        bt=8
    ):

        n = len(df)

        if n < 50:

            return {
                "prob":
                    np.ones(10) / 10,
                "strength": 0.0,
                "stable": 0,
                "total":
                    len(self.equations)
            }

        s = (
            df[pos]
            .astype(int)
            .to_numpy()
        )

        start = max(
            35,
            n - bt
        )

        # ----------------------------------------------------
        # Pre-calculate lags
        # ----------------------------------------------------

        L1 = s[start-1:n-1]

        L2 = s[start-2:n-2]

        L3 = s[start-3:n-3]

        L5 = s[start-5:n-5]

        actual = s[start:n]

        results = []

        total_count = len(actual)

        for name, fn in self.equations:

            try:

                pred = np.array([
                    fn(
                        int(a),
                        int(b),
                        int(c),
                        int(d)
                    ) % 10
                    for a,b,c,d in zip(
                        L1,
                        L2,
                        L3,
                        L5
                    )
                ])

            except Exception:
                continue

            if len(pred) == 0:
                continue

            hits = (
                pred == actual
            )

            hit_rate = (
                float(hits.mean())
            )

            recent_n = min(
                5,
                len(hits)
            )

            recent_rate = float(
                hits[-recent_n:].mean()
            )

            # ------------------------------------------------
            # Minimum sample / stability
            # ------------------------------------------------

            stable = (
                total_count >= 6 and
                hit_rate >= 0.10 and
                recent_rate >= 0.10
            )

            if not stable:
                continue

            score = (
                0.55 * hit_rate +
                0.45 * recent_rate
            )

            results.append({
                "name": name,
                "fn": fn,
                "hit": hit_rate,
                "recent": recent_rate,
                "score": score
            })

        if not results:

            return {
                "prob":
                    np.ones(10) / 10,
                "strength": 0.0,
                "stable": 0,
                "total":
                    len(self.equations)
            }

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ----------------------------------------------------
        # Top 6 stable equations
        # ----------------------------------------------------

        selected = results[:6]

        # ----------------------------------------------------
        # Current lags
        # ----------------------------------------------------

        vals = (
            int(s[-1]),
            int(s[-2]),
            int(s[-3]),
            int(s[-5])
        )

        prob = np.zeros(10)

        total_weight = 0.0

        predictions = []

        for r in selected:

            try:

                pred = int(
                    r["fn"](*vals)
                ) % 10

            except Exception:
                continue

            weight = (
                0.50 +
                r["score"]
            )

            prob[pred] += weight

            total_weight += weight

            predictions.append(
                pred
            )

        if total_weight <= 0:

            prob = (
                np.ones(10) / 10
            )

        else:

            prob /= total_weight

            prob += 0.01

            prob /= prob.sum()

        # ----------------------------------------------------
        # Consensus bonus
        # ----------------------------------------------------

        if predictions:

            counts = np.bincount(
                predictions,
                minlength=10
            )

            consensus = (
                counts /
                max(
                    1,
                    len(predictions)
                )
            )

            prob *= (
                1.0 +
                0.20 * consensus
            )

            prob /= prob.sum()

        strength = float(
            np.mean([
                r["score"]
                for r in selected
            ])
        )

        return {
            "prob": prob,
            "strength": strength,
            "stable": len(selected),
            "total":
                len(self.equations)
        }


# ============================================================
# 9. AI ENGINE
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

        result = np.zeros(10)

        rf_w, et_w, hgb_w = (
            self.weights
        )

        total_w = 0.0

        # ----------------------------------------------------
        # RF
        # ----------------------------------------------------

        model = RandomForestClassifier(

            n_estimators=self.trees,

            max_depth=6,

            min_samples_leaf=3,

            max_features="sqrt",

            class_weight=None,

            n_jobs=-1,

            random_state=42

        )

        model.fit(
            X,
            y
        )

        proba = (
            model
            .predict_proba(X_next)[0]
        )

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

            class_weight=None,

            n_jobs=-1,

            random_state=43

        )

        model.fit(
            X,
            y
        )

        proba = (
            model
            .predict_proba(X_next)[0]
        )

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

            max_iter=65,

            learning_rate=0.05,

            max_leaf_nodes=15,

            min_samples_leaf=3,

            l2_regularization=0.5,

            random_state=44

        )

        model.fit(
            X,
            y
        )

        proba = (
            model
            .predict_proba(X_next)[0]
        )

        for c, p in zip(
            model.classes_,
            proba
        ):

            result[int(c)] += (
                p * hgb_w
            )

        total_w += hgb_w

        if total_w <= 0:

            return (
                np.ones(10) / 10
            )

        result /= total_w

        result += 0.001

        return (
            result /
            result.sum()
        )


# ============================================================
# 10. ENSEMBLE ENGINE
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
        # Adaptive configuration
        # ----------------------------------------------------

        if n >= 700:

            self.trees = 55
            self.bt = 10

        elif n >= 400:

            self.trees = 50
            self.bt = 9

        else:

            self.trees = 45
            self.bt = 8

        self.lags = [
            1, 2, 3, 5
        ]

        self.rolls = [
            3, 5, 10
        ]

        # ----------------------------------------------------
        # Features
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

            self.features.extend([

                f"Odd_{pos}",
                f"High_{pos}",
                f"Prime_{pos}",
                f"Skip_{pos}"

            ])

            for lag in self.lags:

                self.features.append(
                    f"L{lag}_{pos}"
                )

            for w in self.rolls:

                self.features.append(
                    f"RM{w}_{pos}"
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
            EquationEngine()
        )

        self.ai = FastAI(
            self.trees
        )

        # ----------------------------------------------------
        # Base weights
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

    def fast_backtest(
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
            "AI": [],
            "Freq": [],
            "ST": [],
            "Pattern": [],
            "Eq": []
        }

        # ----------------------------------------------------
        # Lightweight WF
        # ----------------------------------------------------

        for idx in range(
            start,
            n
        ):

            actual = int(
                df_hist[pos].iloc[idx]
            )

            hist = (
                df_hist.iloc[:idx]
            )

            # ------------------------------------------------
            # Frequency
            # ------------------------------------------------

            f = self.freq.analyze(
                hist,
                pos
            )

            scores["Freq"].append(
                int(
                    actual in
                    np.argsort(f)[::-1][:5]
                )
            )

            # ------------------------------------------------
            # Transition
            # ------------------------------------------------

            t = self.transition.analyze(
                hist,
                pos
            )

            scores["ST"].append(
                int(
                    actual in
                    np.argsort(t)[::-1][:5]
                )
            )

            # ------------------------------------------------
            # Pattern
            # ------------------------------------------------

            p = self.pattern.analyze(
                hist,
                pos
            )

            scores["Pattern"].append(
                int(
                    actual in
                    np.argsort(p)[::-1][:5]
                )
            )

            # ------------------------------------------------
            # Equation
            # ------------------------------------------------

            eq = self.equation.discover(
                hist,
                pos,
                bt=min(
                    6,
                    max(
                        5,
                        idx - 35
                    )
                )
            )

            scores["Eq"].append(
                int(
                    actual in
                    np.argsort(
                        eq["prob"]
                    )[::-1][:5]
                )
            )

        # ----------------------------------------------------
        # AI only on latest WF point
        # ----------------------------------------------------

        try:

            idx = n - 1

            Xtr = X.iloc[
                :idx
            ]

            ytr = df_hist[pos].iloc[
                :idx
            ]

            xt = X.iloc[
                [idx]
            ]

            model = ExtraTreesClassifier(

                n_estimators=12,

                max_depth=5,

                min_samples_leaf=3,

                max_features="sqrt",

                class_weight=None,

                n_jobs=-1,

                random_state=777

            )

            model.fit(
                Xtr,
                ytr
            )

            proba = (
                model
                .predict_proba(xt)[0]
            )

            ai_top = [
                int(c)
                for c in
                model.classes_[
                    np.argsort(proba)[::-1]
                ][:5]
            ]

            actual = int(
                df_hist[pos].iloc[-1]
            )

            scores["AI"].append(
                int(
                    actual in ai_top
                )
            )

        except Exception:

            scores["AI"].append(
                0
            )

        # ----------------------------------------------------
        # Calculate adaptive weights
        # ----------------------------------------------------

        skill = {}

        for k, vals in scores.items():

            if vals:

                arr = np.array(
                    vals,
                    dtype=float
                )

                recent_n = min(
                    5,
                    len(arr)
                )

                recent = float(
                    arr[-recent_n:].mean()
                )

                overall = float(
                    arr.mean()
                )

                # Stability
                stability = float(
                    1.0 -
                    np.std(arr)
                )

                stability = float(
                    np.clip(
                        stability,
                        0.40,
                        1.00
                    )
                )

                skill[k] = (

                    0.50 * overall +

                    0.30 * recent +

                    0.20 * stability

                )

            else:

                skill[k] = 0.50

        # ----------------------------------------------------
        # Adaptive weight
        # ----------------------------------------------------

        weighted = {}

        for k in self.base_weights:

            weighted[k] = (

                self.base_weights[k] *

                (
                    0.45 +
                    0.55 *
                    np.clip(
                        skill[k],
                        0.10,
                        1.00
                    )
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

            k: v / total

            for k, v in
            weighted.items()

        }

        # ----------------------------------------------------
        # AI cap
        # ----------------------------------------------------

        if weights["AI"] > 0.58:

            diff = (
                weights["AI"] -
                0.58
            )

            weights["AI"] = 0.58

            others = sum(
                v
                for k, v in
                weights.items()
                if k != "AI"
            )

            if others > 0:

                for k in weights:

                    if k != "AI":

                        weights[k] += (
                            diff *
                            weights[k] /
                            others
                        )

        return weights

    # ========================================================
    # POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next
    ):

        # ----------------------------------------------------
        # Dynamic weight
        # ----------------------------------------------------

        weights = self.fast_backtest(
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

        stp = self.transition.analyze(
            hist,
            pos
        )

        ptn = self.pattern.analyze(
            hist,
            pos
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
        # Consensus
        # ----------------------------------------------------

        engines = np.vstack([
            ai,
            fq,
            stp,
            ptn,
            eq
        ])

        mean_prob = (
            engines.mean(axis=0)
        )

        std_prob = (
            engines.std(axis=0)
        )

        # ----------------------------------------------------
        # Main ensemble
        # ----------------------------------------------------

        final = (

            weights["AI"] *
            ai

            +

            weights["Freq"] *
            fq

            +

            weights["ST"] *
            stp

            +

            weights["Pattern"] *
            ptn

            +

            weights["Eq"] *
            eq

        )

        # ----------------------------------------------------
        # Consensus bonus
        # ----------------------------------------------------

        consensus_bonus = (

            1.0 +

            0.18 *
            mean_prob

        )

        # ----------------------------------------------------
        # Variance penalty
        # ----------------------------------------------------

        variance_penalty = (

            1.0 /

            (
                1.0 +
                1.50 *
                std_prob
            )

        )

        final *= (
            consensus_bonus *
            variance_penalty
        )

        final += 0.001

        final /= final.sum()

        return {

            "Prob": final,

            "Top3": [
                (
                    int(i),
                    float(final[i])
                )
                for i in
                np.argsort(
                    final
                )[::-1][:3]
            ],

            "Weights": weights,

            "EquationStrength":
                eq_result["strength"],

            "StableEquations":
                eq_result["stable"]

        }

    # ========================================================
    # PREDICT ALL
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
        # Next date
        # ----------------------------------------------------

        if self.target_dow is not None:

            days = (
                self.target_dow -
                last_date.dayofweek
            ) % 7

            if days <= 0:
                days = 7

        else:

            if len(self.df) >= 2:

                days = max(
                    1,
                    (
                        last_date -
                        self.df[
                            "Date"
                        ].iloc[-2]
                    ).days
                )

            else:

                days = 7

        next_date = (
            last_date +
            timedelta(days=days)
        )

        # ----------------------------------------------------
        # Append dummy row
        # ----------------------------------------------------

        ext = pd.concat([

            self.df,

            pd.DataFrame([{

                "Date":
                    next_date,

                "Result_3D":
                    "000",

                "Result_2D":
                    "00"

            }])

        ], ignore_index=True)

        if status_text:

            status_text.markdown(
                "🧠 **Step 2/4:** "
                "สกัดฟีเจอร์..."
            )

        ext = build_features(
            ext,
            self.lags,
            self.rolls
        )

        if progress_bar:
            progress_bar.progress(25)

        hist = (
            ext.iloc[:-1]
            .copy()
        )

        X = (
            hist[
                self.features
            ]
            .astype(np.float32)
        )

        X_next = (
            ext.iloc[[-1]][
                self.features
            ]
            .astype(np.float32)
        )

        if status_text:

            status_text.markdown(
                "⚙️ **Step 3/4:** "
                "วิเคราะห์ Ensemble..."
            )

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
                    X_next
                )
            )

        if progress_bar:
            progress_bar.progress(70)

        return (
            predictions,
            next_date
        )


# ============================================================
# 11. COMBINE TOP
# ============================================================

def combine_positions(
    preds,
    positions,
    n=5
):

    score = sum(
        preds[pos]["Prob"]
        for pos in positions
    ) / len(positions)

    return [

        (
            int(i),
            float(score[i])
        )

        for i in
        np.argsort(
            score
        )[::-1][:n]

    ]


# ============================================================
# 12. HTML HELPERS
# ============================================================

def render_top3(items):

    return (
        '<span class="dot-sep">•</span>'
        .join(
            f'<span class="top3-number">{n}</span>'
            for n, p in items
        )
    )


def render_top5(items):

    return (
        '<span class="dot-sep">•</span>'
        .join(
            f'<span class="top5-number">{n}</span>'
            for n, p in items
        )
    )


def render_probability(items):

    return (
        " &nbsp;|&nbsp; ".join(
            f"{n} ({p:.1%})"
            for n, p in items
        )
    )


# ============================================================
# 13. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚀 LOTTO AI V.MAX'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'HYBRID SPEED-STABLE<br>'
    'AI + Statistics + Equation + Walk-Forward'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 14. SELECT
# ============================================================

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
# 15. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์เลขเด่น",
    type="primary",
    use_container_width=True
):

    progress_bar = st.progress(0)

    status_text = st.empty()

    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    status_text.markdown(
        "⏳ **Step 1/4:** "
        "โหลดข้อมูลล่าสุด..."
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

    progress_bar.progress(10)

    # --------------------------------------------------------
    # Engine
    # --------------------------------------------------------

    engine = EnsembleEngine(

        df,

        selected_lotto,

        day_options[
            day_label
        ]

    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    preds, next_date = (
        engine.predict_all(
            progress_bar,
            status_text
        )
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    status_text.markdown(
        "✨ **Step 4/4:** "
        "จัดอันดับเลขเด่น..."
    )

    progress_bar.progress(95)

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
    # INFO
    # ========================================================

    st.divider()

    st.info(
        f"📅 งวดเป้าหมาย: "
        f"วัน{days[next_date.dayofweek]} "
        f"{next_date.strftime('%d-%m-%Y')} "
        f"• ข้อมูล {len(df)} งวด"
    )

    # ========================================================
    # TOP 3 EACH POSITION
    # ========================================================

    st.subheader(
        "🎯 เลขเด่น TOP-3 แต่ละหลัก"
    )

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

        st.markdown(
            f"""
            <div class="hot-card">

                <div style="
                    text-align:center;
                    margin:8px 0;
                ">
                    {render_top3(res["Top3"])}
                </div>

                <div class="confidence">
                    {render_probability(res["Top3"])}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # TOP 5 ON
    # ========================================================

    hot_top = combine_positions(
        preds,
        ["H", "T", "O"],
        5
    )

    # ========================================================
    # TOP 5 BOTTOM
    # ========================================================

    hot_bottom = combine_positions(
        preds,
        ["T2", "O2"],
        5
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 สรุปเลขเด่นภาพรวม"
    )

    # --------------------------------------------------------
    # TOP
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="hot-card">

            <div style="
                font-size:17px;
                font-weight:800;
                margin-bottom:8px;
            ">
                🔥 เลขเด่นบน TOP-5
            </div>

            <div style="
                text-align:center;
                margin:12px 0;
            ">
                {render_top5(hot_top)}
            </div>

            <div class="confidence">
                {render_probability(hot_top)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # BOTTOM
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="hot-card">

            <div style="
                font-size:17px;
                font-weight:800;
                margin-bottom:8px;
            ">
                🔥 เลขเด่นล่าง TOP-5
            </div>

            <div style="
                text-align:center;
                margin:12px 0;
            ">
                {render_top5(hot_bottom)}
            </div>

            <div class="confidence">
                {render_probability(hot_bottom)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # SYSTEM INFO
    # ========================================================

    st.divider()

    st.success(
        "✅ วิเคราะห์เสร็จสิ้น"
    )

    st.caption(
        "🚀 Speed-Stable Engine: "
        "RF + ExtraTrees + HGB + "
        "Frequency + Transition + Pattern + "
        "24 Core Equations + Walk-Forward + Consensus"
    )

    st.caption(
        "🔒 Leakage-safe • ไม่มี Calendar/Day/Month • "
        "ไม่มี Persistent Model"
    )

    st.caption(
        "⚠️ คะแนนเป็นคะแนนเชิงสถิติของระบบ "
        "ไม่ใช่ความน่าจะเป็นที่รับประกันผลรางวัลจริง"
    )

    progress_bar.progress(100)

    status_text.empty()

    progress_bar.empty()
