# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID TURBO
# ============================================================
# PURE HISTORICAL MODE
#
# AI:
#   RF + ExtraTrees + HistGradientBoosting
#
# STATISTICS:
#   Frequency + Transition + Pattern
#
# EQUATION:
#   Lag 1,2,3,5
#   Strict causal evaluation
#   Walk-Forward validation
#   Stability filtering
#
# PERFORMANCE:
#   Proxy Backtest = DecisionTreeClassifier
#   Final prediction = RF + ET + HGB
#
# DESIGN:
#   NO Day / Month / Calendar features
#   NO target weekday
#   NO persistent trained model
#   NO joblib
#   Leakage-safe
#   Mobile optimized
# ============================================================

import re
import warnings
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
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 0. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Lotto AI V.MAX Hybrid Turbo",
    page_icon="🚀",
    layout="centered",
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
    "10. หวยหุ้นจีนบ่าย": "https://suksan18190.blogspot.com/2026/07/blog-post_162.html",
}

# ============================================================
# 2. UI
# ============================================================

st.markdown(
    """
<style>
.main-title {
    text-align:center;
    font-size:28px;
    font-weight:900;
    color:#D32F2F;
}
.sub-title {
    text-align:center;
    color:#555;
    font-size:14px;
    margin-bottom:20px;
}
.hot-card {
    padding:18px;
    border-radius:16px;
    border:2px solid #ff4b4b;
    margin:10px 0;
    background:linear-gradient(to bottom right,#ffffff,#fff5f5);
}
.number-highlight {
    font-size:36px;
    font-weight:900;
    color:#D32F2F;
    text-shadow:1px 1px 2px rgba(0,0,0,0.15);
    letter-spacing:2px;
}
.dot-sep {
    color:#FFCDD2;
    font-size:26px;
    margin:0 10px;
}
.badge-ai {
    background:#E3F2FD;
    color:#1565C0;
    padding:4px 12px;
    border-radius:15px;
    font-weight:800;
    font-size:16px;
    border:1px solid #BBDEFB;
}
.badge-stat {
    background:#E8F5E9;
    color:#2E7D32;
    padding:4px 12px;
    border-radius:15px;
    font-weight:800;
    font-size:16px;
    border:1px solid #C8E6C9;
}
.badge-eq {
    background:#F3E5F5;
    color:#7B1FA2;
    padding:4px 12px;
    border-radius:15px;
    font-weight:800;
    font-size:16px;
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
    font-size:15px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 3. FETCH DATA
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def fetch_and_clean_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ ดึงข้อมูลไม่ได้: {exc}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.find(
        "div",
        class_=re.compile(
            r"post-body|entry-content|post-content|content"
        ),
    )

    if main is None:
        main = soup

    lines = main.get_text(separator="\n").split("\n")

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
            d = pd.to_datetime(
                dm.group(1),
                errors="coerce",
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

        rows.append(
            {
                "Date": current_date,
                "Result_3D": str(r3).zfill(3),
                "Result_2D": str(r2).zfill(2),
            }
        )

    if len(rows) < 10:
        st.error("❌ ข้อมูลย้อนหลังน้อยเกินไป")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df = (
        df.dropna()
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# IMPORTANT:
# Date is used ONLY to sort/history display.
# Date / Day / Month / Calendar are NEVER features.
# ============================================================

def build_features(df, lags, rolls):
    x = df.copy()

    r3 = x["Result_3D"].astype(str)
    r2 = x["Result_2D"].astype(str)

    x["H"] = r3.str[0].astype(np.int8)
    x["T"] = r3.str[1].astype(np.int8)
    x["O"] = r3.str[2].astype(np.int8)

    x["T2"] = r2.str[0].astype(np.int8)
    x["O2"] = r2.str[1].astype(np.int8)

    # Previous-result relationships.
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

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        prev = s.shift(1)

        x[f"Odd_{pos}"] = prev % 2
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (
            prev.isin([2, 3, 5, 7])
        ).astype(np.int8)

        for lag in lags:
            x[f"L{lag}_{pos}"] = s.shift(lag)

        for w in rolls:
            x[f"RM{w}_{pos}"] = (
                s.shift(1)
                .rolling(w, min_periods=1)
                .mean()
            )

        # Historical skip only.
        arr = s.to_numpy()
        skip = np.zeros(len(arr), dtype=np.float32)
        last = np.full(10, -1, dtype=np.int32)

        for i, val in enumerate(arr):
            v = int(val)

            if last[v] < 0:
                skip[i] = i
            else:
                skip[i] = i - last[v]

            last[v] = i

        x[f"Skip_{pos}"] = skip

    return (
        x.replace([np.inf, -np.inf], np.nan)
        .fillna(-1)
    )


# ============================================================
# 5. FREQUENCY ENGINE
# ============================================================

class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)

        if len(s) == 0:
            return np.ones(10) / 10

        r15 = s.tail(15).value_counts(normalize=True)
        r30 = s.tail(30).value_counts(normalize=True)
        all_f = s.value_counts(normalize=True)

        score = np.array(
            [
                r15.get(d, 0) * 0.55
                + r30.get(d, 0) * 0.30
                + all_f.get(d, 0) * 0.15
                for d in range(10)
            ],
            dtype=np.float64,
        )

        score += 0.01
        return score / score.sum()


# ============================================================
# 6. TRANSITION ENGINE
# ============================================================

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 6:
            return np.ones(10) / 10

        current = int(df[pos].iloc[-1])

        subset = df[df[pos].shift(1) == current]

        if len(subset) < 2:
            return np.ones(10) / 10

        freq = subset[pos].value_counts(normalize=True)

        score = np.array(
            [freq.get(d, 0) for d in range(10)],
            dtype=np.float64,
        )

        score += 0.01
        return score / score.sum()


# ============================================================
# 7. PATTERN ENGINE
# ============================================================

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 7:
            return np.ones(10) / 10

        a = int(df[pos].iloc[-1])
        b = int(df[pos].iloc[-2])

        subset = df[
            (df[pos].shift(1) == a)
            & (df[pos].shift(2) == b)
        ]

        if len(subset) < 2:
            subset = df[df[pos].shift(1) == a]

        if len(subset) < 1:
            return np.ones(10) / 10

        freq = subset[pos].value_counts(normalize=True)

        score = np.array(
            [freq.get(d, 0) for d in range(10)],
            dtype=np.float64,
        )

        score += 0.01
        return score / score.sum()


# ============================================================
# 8. EQUATION DISCOVERY
# ============================================================

class EquationEngine:
    def __init__(self):
        self.equations = self._build_equations()

    def _build_equations(self):
        eq = [
            ("L1", lambda a, b, c, d: a),
            ("L2", lambda a, b, c, d: b),
            ("L3", lambda a, b, c, d: c),
            ("L5", lambda a, b, c, d: d),

            ("L1+L2", lambda a, b, c, d: a + b),
            ("L1+L3", lambda a, b, c, d: a + c),
            ("L1+L5", lambda a, b, c, d: a + d),
            ("L2+L3", lambda a, b, c, d: b + c),
            ("L2+L5", lambda a, b, c, d: b + d),
            ("L3+L5", lambda a, b, c, d: c + d),

            ("L1-L2", lambda a, b, c, d: a - b),
            ("L1-L3", lambda a, b, c, d: a - c),
            ("L1-L5", lambda a, b, c, d: a - d),
            ("L2-L3", lambda a, b, c, d: b - c),
            ("L2-L5", lambda a, b, c, d: b - d),
            ("L3-L5", lambda a, b, c, d: c - d),

            ("ABS(L1-L2)", lambda a, b, c, d: abs(a - b)),
            ("ABS(L1-L3)", lambda a, b, c, d: abs(a - c)),
            ("ABS(L1-L5)", lambda a, b, c, d: abs(a - d)),
            ("ABS(L2-L3)", lambda a, b, c, d: abs(b - c)),
            ("ABS(L2-L5)", lambda a, b, c, d: abs(b - d)),
            ("ABS(L3-L5)", lambda a, b, c, d: abs(c - d)),

            ("L1+L2+L3", lambda a, b, c, d: a + b + c),
            ("L1+L3+L5", lambda a, b, c, d: a + c + d),
            ("L1+L2+L5", lambda a, b, c, d: a + b + d),

            ("2L1+L2", lambda a, b, c, d: 2 * a + b),
            ("L1+2L2", lambda a, b, c, d: a + 2 * b),
            ("2L1+L3", lambda a, b, c, d: 2 * a + c),
            ("L1+2L3", lambda a, b, c, d: a + 2 * c),
            ("2L1+L5", lambda a, b, c, d: 2 * a + d),
            ("L1+2L5", lambda a, b, c, d: a + 2 * d),
        ]
        return eq

    def _get_lags(self, df, pos, idx):
        if idx < 5:
            return None

        return (
            int(df[pos].iloc[idx - 1]),
            int(df[pos].iloc[idx - 2]),
            int(df[pos].iloc[idx - 3]),
            int(df[pos].iloc[idx - 5]),
        )

    def _predict_eq(self, fn, vals):
        try:
            value = fn(*vals)
            return int(value) % 10
        except (ValueError, TypeError, OverflowError):
            return -1

    def discover(self, df, pos, bt=10):
        n = len(df)

        if n < 50:
            return {
                "prob": np.ones(10) / 10,
                "top": [],
                "strength": 0.0,
                "stable": 0,
                "total": len(self.equations),
                "equations": [],
            }

        start = max(35, n - bt)
        results = []

        for name, fn in self.equations:
            hits = 0
            total = 0
            recent_hits = 0

            for idx in range(start, n):
                vals = self._get_lags(df, pos, idx)

                if vals is None:
                    continue

                pred = self._predict_eq(fn, vals)
                actual = int(df[pos].iloc[idx])

                total += 1

                if pred == actual:
                    hits += 1

                    if idx >= n - 5:
                        recent_hits += 1

            if total == 0:
                continue

            hit_rate = hits / total
            recent_rate = recent_hits / min(5, total)

            stable = hit_rate >= 0.10

            if stable:
                score = 0.70 * hit_rate + 0.30 * recent_rate

                results.append(
                    {
                        "name": name,
                        "fn": fn,
                        "hit": hit_rate,
                        "recent": recent_rate,
                        "score": score,
                    }
                )

        if not results:
            return {
                "prob": np.ones(10) / 10,
                "top": [],
                "strength": 0.0,
                "stable": 0,
                "total": len(self.equations),
                "equations": [],
            }

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        selected = results[:8]

        stable_selected = [
            r for r in selected
            if r["hit"] >= 0.10
        ]

        if not stable_selected:
            stable_selected = selected[:3]

        vals = self._get_lags(df, pos, n)

        if vals is None:
            return {
                "prob": np.ones(10) / 10,
                "top": [],
                "strength": 0.0,
                "stable": len(stable_selected),
                "total": len(self.equations),
                "equations": [],
            }

        prob = np.zeros(10, dtype=np.float64)
        total_weight = 0.0
        equation_predictions = []

        for r in stable_selected:
            pred = self._predict_eq(r["fn"], vals)

            if pred < 0:
                continue

            w = 0.50 + r["score"]
            prob[pred] += w
            total_weight += w

            equation_predictions.append(
                {
                    "name": r["name"],
                    "pred": pred,
                    "hit": r["hit"],
                    "score": r["score"],
                }
            )

        if total_weight <= 0:
            prob = np.ones(10) / 10
        else:
            prob /= total_weight
            prob += 0.01
            prob /= prob.sum()

        top = [
            (int(i), float(prob[i]))
            for i in np.argsort(prob)[::-1][:5]
        ]

        strength = (
            float(np.mean([r["hit"] for r in stable_selected]))
            if stable_selected
            else 0.0
        )

        return {
            "prob": prob,
            "top": top,
            "strength": strength,
            "stable": len(stable_selected),
            "total": len(self.equations),
            "equations": equation_predictions,
        }


# ============================================================
# 9. FINAL AI
# ============================================================

class FastAI:
    def __init__(self, trees=55, weights=(0.35, 0.35, 0.30)):
        self.trees = trees
        self.weights = weights

    @staticmethod
    def _add_proba(result, model, proba, weight):
        for c, p in zip(model.classes_, proba):
            result[int(c)] += p * weight

    def predict(self, X, y, X_next):
        rf_w, et_w, hgb_w = self.weights

        result = np.zeros(10, dtype=np.float64)
        total_w = 0.0

        if rf_w > 0:
            model = RandomForestClassifier(
                n_estimators=self.trees,
                max_depth=6,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X, y)
            self._add_proba(
                result,
                model,
                model.predict_proba(X_next)[0],
                rf_w,
            )
            total_w += rf_w

        if et_w > 0:
            model = ExtraTreesClassifier(
                n_estimators=self.trees,
                max_depth=6,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=43,
            )
            model.fit(X, y)
            self._add_proba(
                result,
                model,
                model.predict_proba(X_next)[0],
                et_w,
            )
            total_w += et_w

        if hgb_w > 0:
            model = HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=3,
                l2_regularization=0.5,
                random_state=44,
            )
            model.fit(X, y)
            self._add_proba(
                result,
                model,
                model.predict_proba(X_next)[0],
                hgb_w,
            )
            total_w += hgb_w

        if total_w <= 0:
            return np.ones(10) / 10

        result /= total_w
        result += 0.001
        return result / result.sum()


# ============================================================
# 10. ENSEMBLE ENGINE
# ============================================================

class EnsembleEngine:
    def __init__(self, df, lottery_name):
        self.df = df.copy()
        self.lottery_name = lottery_name

        n = len(df)

        self.trees = 55
        self.lags = [1, 2, 3, 5]
        self.rolls = [3, 5, 10]

        if n >= 700:
            self.bt = 10
        elif n >= 400:
            self.bt = 9
        else:
            self.bt = 8

        self.mode = "V.MAX HYBRID TURBO"

        self.features = [
            "PrevSum",
            "PrevOdd",
            "DistHT",
            "DistTO",
        ]

        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend(
                [
                    f"Odd_{pos}",
                    f"High_{pos}",
                    f"Prime_{pos}",
                    f"Skip_{pos}",
                ]
            )

            for lag in self.lags:
                self.features.append(f"L{lag}_{pos}")

            for w in self.rolls:
                self.features.append(f"RM{w}_{pos}")

        self.freq = FrequencyEngine()
        self.transition = TransitionEngine()
        self.pattern = PatternEngine()
        self.equation = EquationEngine()

        self.ai = FastAI(
            self.trees,
            (0.35, 0.35, 0.30),
        )

        self.base_weights = {
            "AI": 0.50,
            "Freq": 0.18,
            "ST": 0.12,
            "BT": 0.08,
            "Eq": 0.12,
        }

    # ========================================================
    # FAST STRICT WALK-FORWARD
    # ========================================================
    # Proxy is deliberately lightweight.
    # Final AI is still RF + ET + HGB.
    # ========================================================

    def backtest(self, pos, X, df_hist):
        n = len(X)

        if n < 45:
            return (
                self.base_weights.copy(),
                "Backtest ข้อมูลน้อย",
            )

        start = max(35, n - self.bt)

        scores = {
            "AI": 0.0,
            "Freq": 0.0,
            "ST": 0.0,
            "BT": 0.0,
            "Eq": 0.0,
        }

        total_decay = 0.0

        for step, idx in enumerate(range(start, n)):
            decay = 1.08 ** step
            total_decay += decay

            Xtr = X.iloc[:idx]
            ytr = df_hist[pos].iloc[:idx]
            xt = X.iloc[[idx]]
            actual = int(df_hist[pos].iloc[idx])

            # ------------------------------------------------
            # FAST AI PROXY
            # DecisionTree is used only for WF weighting.
            # ------------------------------------------------
            proxy = DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=3,
                random_state=200 + step,
            )

            proxy.fit(Xtr, ytr)

            tmp = np.zeros(10, dtype=np.float64)

            proba = proxy.predict_proba(xt)[0]

            for c, p in zip(proxy.classes_, proba):
                tmp[int(c)] = p

            if actual in np.argsort(tmp)[::-1][:5]:
                scores["AI"] += decay

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------
            hist = df_hist.iloc[:idx]

            f = self.freq.analyze(hist, pos)
            s = self.transition.analyze(hist, pos)
            b = self.pattern.analyze(hist, pos)

            if actual in np.argsort(f)[::-1][:5]:
                scores["Freq"] += decay

            if actual in np.argsort(s)[::-1][:5]:
                scores["ST"] += decay

            if actual in np.argsort(b)[::-1][:5]:
                scores["BT"] += decay

            # ------------------------------------------------
            # Equation
            # ------------------------------------------------
            eq_result = self.equation.discover(
                hist,
                pos,
                bt=min(8, max(5, idx - 35)),
            )

            eq_prob = eq_result["prob"]

            if actual in np.argsort(eq_prob)[::-1][:5]:
                scores["Eq"] += decay

        if total_decay <= 0:
            return (
                self.base_weights.copy(),
                "Backtest error",
            )

        accuracy = {
            k: scores[k] / total_decay
            for k in scores
        }

        weighted = {}

        for k in accuracy:
            weighted[k] = (
                self.base_weights[k]
                * (
                    0.35
                    + 0.65 * max(0.10, accuracy[k])
                )
            )

        total = sum(weighted.values())

        if total <= 0:
            weights = self.base_weights.copy()
        else:
            weights = {
                k: v / total
                for k, v in weighted.items()
            }

        # AI cap.
        if weights["AI"] > 0.58:
            diff = weights["AI"] - 0.58
            weights["AI"] = 0.58

            other_sum = sum(
                v
                for k, v in weights.items()
                if k != "AI"
            )

            if other_sum > 0:
                for k in weights:
                    if k != "AI":
                        weights[k] += (
                            diff
                            * weights[k]
                            / other_sum
                        )

        msg = (
            f"WF {self.bt} งวด | "
            f"AI {accuracy['AI']:.0%} | "
            f"Freq {accuracy['Freq']:.0%} | "
            f"Transition {accuracy['ST']:.0%} | "
            f"Pattern {accuracy['BT']:.0%} | "
            f"Equation {accuracy['Eq']:.0%}"
        )

        return weights, msg

    # ========================================================
    # PROCESS POSITION
    # ========================================================

    def process_position(
        self,
        pos,
        hist,
        X,
        X_next,
    ):
        weights, bt_msg = self.backtest(
            pos,
            X,
            hist,
        )

        ai = self.ai.predict(
            X,
            hist[pos],
            X_next,
        )

        fq = self.freq.analyze(hist, pos)
        stp = self.transition.analyze(hist, pos)
        ptn = self.pattern.analyze(hist, pos)

        eq_result = self.equation.discover(
            hist,
            pos,
            bt=self.bt,
        )

        eq = eq_result["prob"]

        final = (
            weights["AI"] * ai
            + weights["Freq"] * fq
            + weights["ST"] * stp
            + weights["BT"] * ptn
            + weights["Eq"] * eq
        )

        final += 0.001
        final /= final.sum()

        def top_n(p, n):
            return [
                (int(i), float(p[i]))
                for i in np.argsort(p)[::-1][:n]
            ]

        return {
            "Final": top_n(final, 5),
            "AI": top_n(ai, 3),
            "Freq": top_n(fq, 3),
            "Transition": top_n(stp, 3),
            "Pattern": top_n(ptn, 3),
            "Equation": eq_result["top"],
            "EquationStrength": eq_result["strength"],
            "StableEquations": eq_result["stable"],
            "TotalEquations": eq_result["total"],
            "Prob": final,
            "Weights": weights,
            "BT": bt_msg,
        }

    # ========================================================
    # PREDICT ALL
    # ========================================================
    # IMPORTANT:
    # No weekday is selected or used.
    # next_date is display-only and has ZERO influence on model.
    # ========================================================

    def predict_all(self):
        last_date = self.df["Date"].iloc[-1]

        # Display-only date.
        # It is NOT a feature and NOT used by the model.
        next_date = last_date + timedelta(days=1)

        ext = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    [
                        {
                            "Date": next_date,
                            "Result_3D": "000",
                            "Result_2D": "00",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        ext = build_features(
            ext,
            self.lags,
            self.rolls,
        )

        hist = ext.iloc[:-1].copy()

        X = hist[self.features].astype(np.float32)

        X_next = ext.iloc[[-1]][self.features].astype(
            np.float32
        )

        predictions = {}

        for pos in ["H", "T", "O", "T2", "O2"]:
            predictions[pos] = self.process_position(
                pos,
                hist,
                X,
                X_next,
            )

        return predictions, next_date


# ============================================================
# 11. UI HELPERS
# ============================================================

def html_top5(items):
    parts = [
        f'<span class="number-highlight">{n}</span>'
        for n, _ in items
    ]
    return '<span class="dot-sep">•</span>'.join(parts)


def html_badge(items, badge_class):
    parts = [str(n) for n, _ in items]

    return (
        f'<span class="{badge_class}">'
        + " &nbsp;•&nbsp; ".join(parts)
        + "</span>"
    )


def nums_prob(items):
    return " | ".join(
        f"{n} ({p:.1%})"
        for n, p in items
    )


def combine_top_n(preds, positions, n=5):
    score = (
        sum(preds[pos]["Prob"] for pos in positions)
        / len(positions)
    )

    return [
        (int(i), float(score[i]))
        for i in np.argsort(score)[::-1][:n]
    ]


# ============================================================
# 12. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 LOTTO AI V.MAX TURBO</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'PURE HISTORICAL HYBRID<br>'
    '<b>AI + Statistics + Equation Discovery + Strict Walk-Forward</b>'
    '<br>NO DAY • NO MONTH • NO CALENDAR'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()

# ============================================================
# 13. SELECT
# ============================================================

selected_lotto = st.selectbox(
    "🎯 เลือกหวย",
    list(LOTTERY_SOURCES.keys()),
)

# ============================================================
# 14. RUN
# ============================================================

if st.button(
    "🚀 วิเคราะห์เลขเด่น",
    type="primary",
    use_container_width=True,
):
    with st.spinner(
        "⚡ AI + Statistical + Equation Discovery กำลังประมวลผล..."
    ):
        df = fetch_and_clean_data(
            LOTTERY_SOURCES[selected_lotto]
        )

        if df.empty:
            st.stop()

        if len(df) < 50:
            st.warning(
                f"⚠️ มีข้อมูลเพียง {len(df)} งวด "
                "ระบบยังทำงานได้ แต่ Equation/WF จะมีความเสถียรต่ำ"
            )

        engine = EnsembleEngine(
            df,
            selected_lotto,
        )

        preds, next_date = engine.predict_all()

        labels = {
            "H": "หลักร้อย 3 ตัวบน",
            "T": "หลักสิบ 3 ตัวบน",
            "O": "หลักหน่วย 3 ตัวบน",
            "T2": "หลักสิบ 2 ตัวล่าง",
            "O2": "หลักหน่วย 2 ตัวล่าง",
        }

        st.divider()

        st.info(
            f"📊 ข้อมูลย้อนหลัง {len(df)} งวด | "
            f"งวดถัดไปสำหรับแสดงผล: "
            f"{next_date.strftime('%d-%m-%Y')} "
            f"| วันที่ไม่มีผลต่อการคำนวณเลข"
        )

        # ====================================================
        # POSITION RESULTS
        # ====================================================

        for pos in ["H", "T", "O", "T2", "O2"]:
            res = preds[pos]

            st.markdown(
                f'<div class="position-title">'
                f'📍 {labels[pos]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="hot-card">
                    <div style="
                        font-weight:700;
                        color:#444;
                        margin-bottom:8px;
                    ">
                        🔥 FINAL TOP-5
                    </div>

                    <div style="
                        text-align:center;
                        margin:10px 0;
                    ">
                        {html_top5(res["Final"])}
                    </div>

                    <div style="
                        font-size:13px;
                        color:#888;
                        text-align:center;
                    ">
                        {nums_prob(res["Final"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="info-row">
                    🤖 <b>AI TOP-3:</b>
                    &nbsp;
                    {html_badge(res["AI"], "badge-ai")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="info-row">
                    📊 <b>Frequency TOP-3:</b>
                    &nbsp;
                    {html_badge(res["Freq"], "badge-stat")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="info-row">
                    🧮 <b>Equation TOP-5:</b>
                    &nbsp;
                    {html_badge(res["Equation"], "badge-eq")}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    font-size:13px;
                    color:#777;
                    margin-top:8px;
                ">
                    🧮 สมการผ่าน Stability:
                    <b>{res["StableEquations"]}</b>
                    / {res["TotalEquations"]}
                    &nbsp; | &nbsp;
                    Strength:
                    <b>{res["EquationStrength"]:.0%}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    font-size:13px;
                    color:#999;
                    margin-top:8px;
                ">
                    📈 {res["BT"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            w = res["Weights"]

            st.markdown(
                f"""
                <div style="
                    font-size:13px;
                    color:#999;
                ">
                    ⚖️ น้ำหนัก:
                    AI {w["AI"]:.0%}
                    |
                    Frequency {w["Freq"]:.0%}
                    |
                    Transition {w["ST"]:.0%}
                    |
                    Pattern {w["BT"]:.0%}
                    |
                    Equation {w["Eq"]:.0%}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

        # ====================================================
        # OVERALL
        # ====================================================

        hot_top = combine_top_n(
            preds,
            ["H", "T", "O"],
        )

        hot_bottom = combine_top_n(
            preds,
            ["T2", "O2"],
        )

        st.subheader("🔥 สรุปเลขเด่นภาพรวม")

        st.markdown(
            f"""
            <div class="hot-card">
                <div style="
                    font-weight:700;
                    color:#444;
                ">
                    🔥 HOT 5-TOP รูด/วิ่งบน
                </div>

                <div style="
                    text-align:center;
                    margin:10px 0;
                ">
                    {html_top5(hot_top)}
                </div>

                <div style="
                    font-size:13px;
                    color:#888;
                    text-align:center;
                ">
                    {nums_prob(hot_top)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="hot-card">
                <div style="
                    font-weight:700;
                    color:#444;
                ">
                    🔥 HOT 5-TOP รูด/วิ่งล่าง
                </div>

                <div style="
                    text-align:center;
                    margin:10px 0;
                ">
                    {html_top5(hot_bottom)}
                </div>

                <div style="
                    font-size:13px;
                    color:#888;
                    text-align:center;
                ">
                    {nums_prob(hot_bottom)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(
            "✅ วิเคราะห์เสร็จสิ้น • "
            "Pure Historical AI + Statistics + Equation"
        )

        st.caption(
            "ระบบไม่มี Persistent Model — "
            "โมเดลและสมการคำนวณใหม่จากข้อมูลปัจจุบันทุกครั้งที่กดวิเคราะห์"
        )

        st.caption(
            "⚡ Performance: Walk-Forward ใช้ DecisionTree เป็น Proxy "
            "เพื่อคำนวณน้ำหนักอย่างรวดเร็ว ส่วน Predict รอบสุดท้ายใช้ "
            "RF + ExtraTrees + HGB"
        )
