# ============================================================
# ❄️ LOTTO AI V.MAX HYBRID (ระบบเลขดับ)
# ============================================================
#
# AI:
#   RF + ExtraTrees + HistGradientBoosting
#
# STATISTICS:
#   Frequency + Transition + Pattern
#
# EQUATION DISCOVERY:
#   Lag 1,2,3,5
#   Strict causal evaluation
#   Walk-Forward validation
#   Stability filtering
#   Equation COLD-5 voting
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
    page_title="Lotto AI V.MAX (เลขดับ)",
    page_icon="❄️",
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
# 2. BASIC CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 800;
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
        font-weight: 800;
        margin-top: 18px;
        margin-bottom: 8px;
        color: #1565c0;
    }
    .cold-card {
        padding: 14px;
        border-radius: 14px;
        border: 1px solid #bbdefb;
        background-color: #e3f2fd;
        margin: 8px 0;
    }
    .number-highlight {
        font-size: 25px;
        font-weight: 800;
        padding: 4px 8px;
        color: #0d47a1;
    }
    .dot-sep {
        color: #90caf9;
        margin: 0 3px;
    }
    .info-row {
        padding: 5px 0;
        font-size: 14px;
    }
    .badge-ai,
    .badge-stat,
    .badge-eq {
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #b0bec5;
        background-color: #ffffff;
        color: #37474f;
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
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if main is None:
            main = soup

        lines = main.get_text(separator="\n").split("\n")
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
        num_pattern = re.compile(r"\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b")
        current_date = pd.Timestamp(datetime.now())
        rows = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            dm = date_pattern.search(line)
            if dm:
                try:
                    d = pd.to_datetime(dm.group(1), errors="coerce")
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
            raise ValueError("ข้อมูลน้อยเกินไป")

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"❌ ดึงข้อมูลไม่ได้: {e}")
        return pd.DataFrame()


# ============================================================
# 4. FEATURE ENGINEERING
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

    ph = x["H"].shift(1)
    pt = x["T"].shift(1)
    po = x["O"].shift(1)

    x["PrevSum"] = ph + pt + po
    x["PrevOdd"] = (ph % 2) + (pt % 2) + (po % 2)
    x["DistHT"] = (ph - pt).abs()
    x["DistTO"] = (pt - po).abs()

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        prev = s.shift(1)

        x[f"Odd_{pos}"] = (prev % 2)
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (prev.isin([2, 3, 5, 7])).astype(np.int8)

        for lag in lags:
            x[f"L{lag}_{pos}"] = s.shift(lag)

        for w in rolls:
            x[f"RM{w}_{pos}"] = s.shift(1).rolling(w, min_periods=1).mean()

        arr = s.to_numpy()
        raw_skip = np.zeros(len(arr), dtype=np.float32)
        last = np.full(10, -1, dtype=np.int32)

        for i, val in enumerate(arr):
            v = int(val)
            if last[v] < 0:
                raw_skip[i] = i
            else:
                raw_skip[i] = i - last[v]
            last[v] = i

        x[f"Skip_{pos}"] = pd.Series(raw_skip, index=x.index).shift(1)

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)


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

        score = np.array([
            r15.get(d, 0) * 0.55 + r30.get(d, 0) * 0.30 + all_f.get(d, 0) * 0.15
            for d in range(10)
        ])
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
        score = np.array([freq.get(d, 0) for d in range(10)])
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

        subset = df[(df[pos].shift(1) == a) & (df[pos].shift(2) == b)]
        if len(subset) < 2:
            subset = df[df[pos].shift(1) == a]

        if len(subset) < 1:
            return np.ones(10) / 10

        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)])
        score += 0.01
        return score / score.sum()


# ============================================================
# 8. EQUATION DISCOVERY
# ============================================================

class EquationEngine:
    def __init__(self):
        self.equations = self._build_equations()

    def _build_equations(self):
        eq = []
        eq += [
            ("L1", lambda a, b, c, d: a),
            ("L2", lambda a, b, c, d: b),
            ("L3", lambda a, b, c, d: c),
            ("L5", lambda a, b, c, d: d)
        ]
        eq += [
            ("L1+L2", lambda a, b, c, d: a + b),
            ("L1+L3", lambda a, b, c, d: a + c),
            ("L1+L5", lambda a, b, c, d: a + d),
            ("L2+L3", lambda a, b, c, d: b + c),
            ("L2+L5", lambda a, b, c, d: b + d),
            ("L3+L5", lambda a, b, c, d: c + d)
        ]
        eq += [
            ("L1-L2", lambda a, b, c, d: a - b),
            ("L1-L3", lambda a, b, c, d: a - c),
            ("L1-L5", lambda a, b, c, d: a - d),
            ("L2-L3", lambda a, b, c, d: b - c),
            ("L2-L5", lambda a, b, c, d: b - d),
            ("L3-L5", lambda a, b, c, d: c - d)
        ]
        eq += [
            ("ABS(L1-L2)", lambda a, b, c, d: abs(a - b)),
            ("ABS(L1-L3)", lambda a, b, c, d: abs(a - c)),
            ("ABS(L1-L5)", lambda a, b, c, d: abs(a - d)),
            ("ABS(L2-L3)", lambda a, b, c, d: abs(b - c)),
            ("ABS(L2-L5)", lambda a, b, c, d: abs(b - d)),
            ("ABS(L3-L5)", lambda a, b, c, d: abs(c - d))
        ]
        eq += [
            ("L1+L2+L3", lambda a, b, c, d: a + b + c),
            ("L1+L3+L5", lambda a, b, c, d: a + c + d),
            ("L1+L2+L5", lambda a, b, c, d: a + b + d)
        ]
        eq += [
            ("2L1+L2", lambda a, b, c, d: 2 * a + b),
            ("L1+2L2", lambda a, b, c, d: a + 2 * b),
            ("2L1+L3", lambda a, b, c, d: 2 * a + c),
            ("L1+2L3", lambda a, b, c, d: a + 2 * c),
            ("2L1+L5", lambda a, b, c, d: 2 * a + d),
            ("L1+2L5", lambda a, b, c, d: a + 2 * d)
        ]
        return eq

    def _get_lags(self, df, pos, idx):
        if idx < 5:
            return None
        a = int(df[pos].iloc[idx - 1])
        b = int(df[pos].iloc[idx - 2])
        c = int(df[pos].iloc[idx - 3])
        d = int(df[pos].iloc[idx - 5])
        return (a, b, c, d)

    def _predict_eq(self, fn, vals):
        try:
            value = fn(*vals)
            return int(value) % 10
        except Exception:
            return -1

    def discover(self, df, pos, bt=10):
        n = len(df)
        if n < 50:
            return {
                "prob": np.ones(10) / 10,
                "cold": [],
                "strength": 0.0,
                "stable": 0,
                "total": len(self.equations)
            }

        start = max(35, n - bt)
        results = []

        for name, fn in self.equations:
            hits = 0
            total = 0
            recent_hits = 0
            recent_total = 0

            for idx in range(start, n):
                vals = self._get_lags(df, pos, idx)
                if vals is None:
                    continue
                pred = self._predict_eq(fn, vals)
                if pred < 0:
                    continue
                actual = int(df[pos].iloc[idx])
                total += 1
                if pred == actual:
                    hits += 1
                    if idx >= n - 5:
                        recent_hits += 1
                if idx >= n - 5:
                    recent_total += 1

            if total == 0:
                continue

            hit_rate = hits / total
            recent_rate = recent_hits / recent_total if recent_total > 0 else 0.0
            stable = hit_rate >= 0.10 and recent_rate >= 0.10
            score = 0.70 * hit_rate + 0.30 * recent_rate

            if stable:
                results.append({
                    "name": name,
                    "fn": fn,
                    "hit": hit_rate,
                    "recent": recent_rate,
                    "score": score
                })

        if not results:
            return {
                "prob": np.ones(10) / 10,
                "cold": [],
                "strength": 0.0,
                "stable": 0,
                "total": len(self.equations)
            }

        results.sort(key=lambda x: x["score"], reverse=True)
        selected = results[:8]
        stable_selected = [r for r in selected if r["hit"] >= 0.10 and r["recent"] >= 0.10]

        if not stable_selected:
            stable_selected = selected[:3]

        vals = self._get_lags(df, pos, n)
        if vals is None:
            return {
                "prob": np.ones(10) / 10,
                "cold": [],
                "strength": 0.0,
                "stable": len(stable_selected),
                "total": len(self.equations)
            }

        prob = np.zeros(10)
        total_weight = 0.0
        equation_predictions = []

        for r in stable_selected:
            pred = self._predict_eq(r["fn"], vals)
            if pred < 0:
                continue
            w = 0.50 + r["score"]
            prob[pred] += w
            total_weight += w
            equation_predictions.append({
                "name": r["name"],
                "pred": pred,
                "hit": r["hit"],
                "recent": r["recent"],
                "score": r["score"]
            })

        if total_weight <= 0:
            prob = np.ones(10) / 10
        else:
            prob /= total_weight
            prob += 0.01
            prob /= prob.sum()

        # ดึงเลขที่คะแนนต่ำสุด (เลขดับ)
        cold = [(int(i), float(prob[i])) for i in np.argsort(prob)[:5]]
        strength = np.mean([r["hit"] for r in stable_selected]) if stable_selected else 0.0

        return {
            "prob": prob,
            "cold": cold,
            "strength": strength,
            "stable": len(stable_selected),
            "total": len(self.equations),
            "equations": equation_predictions
        }


# ============================================================
# 9. AI MODEL
# ============================================================

class FastAI:
    def __init__(self, trees=55, weights=(0.35, 0.35, 0.30)):
        self.trees = trees
        self.weights = weights

    def predict(self, X, y, X_next):
        rf_w, et_w, hgb_w = self.weights
        result = np.zeros(10)
        total_w = 0.0

        if rf_w > 0:
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
            proba = model.predict_proba(X_next)[0]
            for c, p in zip(model.classes_, proba):
                result[int(c)] += p * rf_w
            total_w += rf_w

        if et_w > 0:
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
            proba = model.predict_proba(X_next)[0]
            for c, p in zip(model.classes_, proba):
                result[int(c)] += p * et_w
            total_w += et_w

        if hgb_w > 0:
            model = HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=3,
                l2_regularization=0.5,
                random_state=44
            )
            model.fit(X, y)
            proba = model.predict_proba(X_next)[0]
            for c, p in zip(model.classes_, proba):
                result[int(c)] += p * hgb_w
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
    def __init__(self, df, lottery_name, target_dow=None):
        self.df = df.copy()
        self.lottery_name = lottery_name
        self.target_dow = target_dow
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

        self.features = ["PrevSum", "PrevOdd", "DistHT", "DistTO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"Odd_{pos}", f"High_{pos}", f"Prime_{pos}", f"Skip_{pos}"])
            for lag in self.lags:
                self.features.append(f"L{lag}_{pos}")
            for w in self.rolls:
                self.features.append(f"RM{w}_{pos}")

        self.freq = FrequencyEngine()
        self.transition = TransitionEngine()
        self.pattern = PatternEngine()
        self.equation = EquationEngine()
        self.ai = FastAI(self.trees, (0.35, 0.35, 0.30))

        self.base_weights = {
            "AI": 0.50,
            "Freq": 0.18,
            "ST": 0.12,
            "BT": 0.08,
            "Eq": 0.12
        }

    # ฟังก์ชัน Backtest วัดความแม่นยำของ Model ยังคงใช้ตรรกะตรวจเลขที่ออก (เพราะโมเดลที่แม่นยำ จะคัดเลขดับได้แม่นยำด้วย)
    def backtest(self, pos, X, df_hist):
        n = len(X)
        if n < 45:
            return (self.base_weights.copy(), "Backtest ข้อมูลน้อย")

        start = max(35, n - self.bt)
        scores = {"AI": 0.0, "Freq": 0.0, "ST": 0.0, "BT": 0.0, "Eq": 0.0}
        outcomes = {"AI": [], "Freq": [], "ST": [], "BT": [], "Eq": []}
        total_decay = 0.0

        for step, idx in enumerate(range(start, n)):
            decay = 1.08 ** step
            total_decay += decay

            Xtr = X.iloc[:idx]
            ytr = df_hist[pos].iloc[:idx]
            xt = X.iloc[[idx]]
            actual = int(df_hist[pos].iloc[idx])

            try:
                proxy = ExtraTreesClassifier(
                    n_estimators=10, max_depth=5, min_samples_leaf=3,
                    max_features="sqrt", random_state=200 + step, n_jobs=-1
                )
                proxy.fit(Xtr, ytr)
                tmp = np.zeros(10)
                proba = proxy.predict_proba(xt)[0]
                for c, p in zip(proxy.classes_, proba):
                    tmp[int(c)] = p
                if actual in np.argsort(tmp)[::-1][:5]:
                    scores["AI"] += decay
                    outcomes["AI"].append(1)
                else:
                    outcomes["AI"].append(0)
            except Exception:
                outcomes["AI"].append(0)

            hist = df_hist.iloc[:idx].copy()
            f = self.freq.analyze(hist, pos)
            freq_hit = int(actual in np.argsort(f)[::-1][:5])
            if freq_hit: scores["Freq"] += decay
            outcomes["Freq"].append(freq_hit)

            s = self.transition.analyze(hist, pos)
            st_hit = int(actual in np.argsort(s)[::-1][:5])
            if st_hit: scores["ST"] += decay
            outcomes["ST"].append(st_hit)

            b = self.pattern.analyze(hist, pos)
            pattern_hit = int(actual in np.argsort(b)[::-1][:5])
            if pattern_hit: scores["BT"] += decay
            outcomes["BT"].append(pattern_hit)

            eq_result = self.equation.discover(hist, pos, bt=min(8, max(5, idx - 35)))
            eq_prob = eq_result["prob"]
            eq_hit = int(actual in np.argsort(eq_prob)[::-1][:5])
            if eq_hit: scores["Eq"] += decay
            outcomes["Eq"].append(eq_hit)

        if total_decay <= 0:
            return (self.base_weights.copy(), "Backtest error")

        accuracy = {k: scores[k] / total_decay for k in scores}
        stability = {}
        for k, vals in outcomes.items():
            if len(vals) < 2:
                stability[k] = 0.70
                continue
            std = float(np.std(vals))
            stability_score = 1.0 - 0.60 * (std / 0.50)
            stability[k] = float(np.clip(stability_score, 0.40, 1.00))

        adaptive_score = {k: accuracy[k] * stability[k] for k in accuracy}
        weighted = {k: self.base_weights[k] * (0.35 + 0.65 * max(0.10, adaptive_score[k])) for k in adaptive_score}

        total = sum(weighted.values())
        if total <= 0:
            weights = self.base_weights.copy()
        else:
            weights = {k: v / total for k, v in weighted.items()}

        if weights["AI"] > 0.58:
            diff = weights["AI"] - 0.58
            weights["AI"] = 0.58
            other_sum = sum(v for k, v in weights.items() if k != "AI")
            if other_sum > 0:
                for k in weights:
                    if k != "AI":
                        weights[k] += diff * (weights[k] / other_sum)

        msg = (
            f"WF {self.bt} งวด | AI {accuracy['AI']:.0%} (S {stability['AI']:.0%}) | "
            f"Freq {accuracy['Freq']:.0%} (S {stability['Freq']:.0%}) | "
            f"ST {accuracy['ST']:.0%} (S {stability['ST']:.0%}) | "
            f"Pattern {accuracy['BT']:.0%} (S {stability['BT']:.0%}) | "
            f"Equation {accuracy['Eq']:.0%} (S {stability['Eq']:.0%})"
        )
        return (weights, msg)

    def process_position(self, pos, hist, X, X_next, next_date):
        weights, bt_msg = self.backtest(pos, X, hist)
        ai = self.ai.predict(X, hist[pos], X_next)
        fq = self.freq.analyze(hist, pos)
        stp = self.transition.analyze(hist, pos)
        ptn = self.pattern.analyze(hist, pos)
        eq_result = self.equation.discover(hist, pos, bt=self.bt)
        eq = eq_result["prob"]

        final = (
            weights["AI"] * ai +
            weights["Freq"] * fq +
            weights["ST"] * stp +
            weights["BT"] * ptn +
            weights["Eq"] * eq
        )
        final += 0.001
        final /= final.sum()

        # ดึงตัวที่ Probability *น้อยที่สุด* (เลขดับ) โดยเรียงจากน้อยไปมาก
        def cold_n(p, n):
            return [(int(i), float(p[i])) for i in np.argsort(p)[:n]]

        return {
            "Final_Cold": cold_n(final, 5),
            "AI_Cold": cold_n(ai, 3),
            "Freq_Cold": cold_n(fq, 3),
            "Transition_Cold": cold_n(stp, 3),
            "Pattern_Cold": cold_n(ptn, 3),
            "Equation_Cold": eq_result["cold"],
            "EquationStrength": eq_result["strength"],
            "StableEquations": eq_result["stable"],
            "TotalEquations": eq_result["total"],
            "Prob": final,
            "Weights": weights,
            "BT": bt_msg
        }

    # ========================================================
    # PREDICT ALL (อัปเดตสเตป Progress)
    # ========================================================
    def predict_all(self, progress_bar=None, status_text=None):
        last_date = self.df["Date"].iloc[-1]

        if self.target_dow is not None:
            days = (self.target_dow - last_date.dayofweek) % 7
            if days <= 0:
                days = 7
        else:
            if len(self.df) >= 2:
                days = max(1, (last_date - self.df["Date"].iloc[-2]).days)
            else:
                days = 7

        next_date = last_date + timedelta(days=days)

        ext = pd.concat([
            self.df,
            pd.DataFrame([{"Date": next_date, "Result_3D": "000", "Result_2D": "00"}])
        ], ignore_index=True)

        if status_text:
            status_text.markdown("🧠 **Step 2/5:** สกัดฟีเจอร์ (Feature Engineering)...")
            
        ext = build_features(ext, self.lags, self.rolls)

        if progress_bar:
            progress_bar.progress(20)

        hist = ext.iloc[:-1].copy()
        X = hist[self.features].astype(np.float32)
        X_next = ext.iloc[[-1]][self.features].astype(np.float32)

        if status_text:
            status_text.markdown("⚙️ **Step 3/5:** รันระบบ Walk-Forward ยืนยันข้อมูลล่าสุด...")
        
        predictions = {}
        for pos in ["H", "T", "O", "T2", "O2"]:
            predictions[pos] = self.process_position(pos, hist, X, X_next, next_date)

        if progress_bar:
            progress_bar.progress(40)

        return predictions, next_date

    # ========================================================
    # EVALUATE PAST 10 DRAWS (Backtest เลขดับ)
    # ========================================================
    def evaluate_past_10(self, progress_bar=None, status_text=None):
        n_total = len(self.df)
        num_records = min(10, n_total - 35)
        
        if num_records < 1:
            return []

        start_idx = n_total - num_records
        records = []
        ext = build_features(self.df, self.lags, self.rolls)

        for step, i in enumerate(range(start_idx, n_total)):
            if status_text:
                status_text.markdown(f"🕰️ **Step 4/5:** วิเคราะห์ย้อนหลัง งวดที่ {step+1}/{num_records}...")
            if progress_bar:
                prog = 40 + int(((step + 1) / num_records) * 50)
                progress_bar.progress(prog)

            hist = ext.iloc[:i].copy()
            X = hist[self.features].astype(np.float32)
            X_next = ext.iloc[[i]][self.features].astype(np.float32)

            preds = {}
            for pos in ["H", "T", "O", "T2", "O2"]:
                preds[pos] = self.process_position(pos, hist, X, X_next, None)

            # คำนวณความน่าจะเป็นรวม
            top_score = sum(preds[pos]["Prob"] for pos in ["H", "T", "O"]) / 3
            bot_score = sum(preds[pos]["Prob"] for pos in ["T2", "O2"]) / 2

            # ดึงเลขดับ 7 ตัว (คะแนนน้อยสุด)
            cold_top = [int(idx) for idx in np.argsort(top_score)[:7]]
            cold_bot = [int(idx) for idx in np.argsort(bot_score)[:7]]

            actual_row = self.df.iloc[i]
            date_str = actual_row["Date"].strftime("%d-%m-%Y")
            actual_3d = str(actual_row["Result_3D"]).zfill(3)
            actual_2d = str(actual_row["Result_2D"]).zfill(2)

            # เงื่อนไขเลขดับ: ถ้า "ไม่มี" เลขที่ทายโผล่มาในผลลัพธ์ = ทายถูก (ดับอยู่/ดับผ่าน)
            top_hit = not any(int(d) in cold_top for d in actual_3d)
            bot_hit = not any(int(d) in cold_bot for d in actual_2d)

            records.append({
                "Date": date_str,
                "Result_3D": actual_3d,
                "Result_2D": actual_2d,
                "Cold_Top": cold_top,
                "Cold_Bot": cold_bot,
                "Top_Hit": top_hit,
                "Bot_Hit": bot_hit
            })

        return records[::-1]


# ============================================================
# 11. UI HELPERS
# ============================================================

def html_top5(items):
    parts = [f'<span class="number-highlight">{n}</span>' for n, p in items]
    return '<span class="dot-sep">•</span>'.join(parts)

def html_badge(items, badge_class):
    parts = [str(n) for n, p in items]
    return f'<span class="{badge_class}">' + " &nbsp;•&nbsp; ".join(parts) + '</span>'

def nums_prob(items):
    return " | ".join(f"{n} ({p:.1%})" for n, p in items)

# เปลี่ยนฟังก์ชันเรียงลำดับดึงตัวที่ Probability ต่ำสุด
def combine_cold_n(preds, positions, n=7):
    score = sum(preds[pos]["Prob"] for pos in positions) / len(positions)
    return [(int(i), float(score[i])) for i in np.argsort(score)[:n]]


# ============================================================
# 12. HEADER
# ============================================================

st.markdown('<div class="main-title">❄️ LOTTO AI V.MAX (ระบบเลขดับ)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI + Statistics + Equation Discovery + Strict Walk-Forward<br> โฟกัสคำนวณหา <b>เลขที่มีความน่าจะเป็นต่ำสุด (ดับ)</b> ประจำงวด</div>', unsafe_allow_html=True)
st.divider()


# ============================================================
# 13. SELECT
# ============================================================

c1, c2 = st.columns(2)
selected_lotto = c1.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()))
day_options = {
    "อัตโนมัติ": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2,
    "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6
}
day_label = c2.selectbox("📅 วันออกรางวัล", list(day_options.keys()))


# ============================================================
# 14. RUN
# ============================================================

if st.button("❄️ วิเคราะห์เลขดับด้วย AI + สมการ", type="primary", use_container_width=True):

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.markdown("⏳ **Step 1/5:** โหลดข้อมูลล่าสุดจากแหล่งที่มา...")
    df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])

    if df.empty:
        status_text.error("❌ ไม่สามารถดึงข้อมูลได้")
        st.stop()
    if len(df) < 50:
        status_text.error(f"❌ ต้องมีข้อมูลอย่างน้อย 50 งวด (พบ {len(df)} งวด)")
        st.stop()

    progress_bar.progress(10)

    # 1. รันเอนจิ้นสำหรับข้อมูลล่าสุด
    engine = EnsembleEngine(df, selected_lotto, day_options[day_label])
    preds, next_date = engine.predict_all(progress_bar, status_text)

    # 2. รันจำลองผลย้อนหลัง 10 งวด
    past_records = engine.evaluate_past_10(progress_bar, status_text)

    status_text.markdown("✨ **Step 5/5:** ประมวลผลเสร็จสิ้น จัดเตรียมการแสดงผล...")
    progress_bar.progress(100)

    # ล้างข้อความโหลด
    status_text.empty()
    progress_bar.empty()

    labels = {
        "H": "หลักร้อย 3 ตัวบน", "T": "หลักสิบ 3 ตัวบน", "O": "หลักหน่วย 3 ตัวบน",
        "T2": "หลักสิบ 2 ตัวล่าง", "O2": "หลักหน่วย 2 ตัวล่าง"
    }
    days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

    # ====================================================
    # INFO
    # ====================================================
    st.divider()
    st.info(f"📅 วิเคราะห์เลขดับงวดเป้าหมาย: วัน{days[next_date.dayofweek]} {next_date.strftime('%d-%m-%Y')} (อิงจากข้อมูล {len(df)} งวด)")

    # ====================================================
    # POSITION RESULTS
    # ====================================================
    for pos in ["H", "T", "O", "T2", "O2"]:
        res = preds[pos]
        st.markdown(f'<div class="position-title">📍 {labels[pos]}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="cold-card">
                <div style="font-weight:700; margin-bottom:8px; color: #1976d2;">❄️ FINAL COLD-5 (ดับ 5 ตัว)</div>
                <div style="text-align:center; margin:10px 0;">{html_top5(res["Final_Cold"])}</div>
                <div style="font-size:13px; color:#546e7a; text-align:center;">{nums_prob(res["Final_Cold"])}</div>
            </div>
            <div class="info-row">🤖 <b>AI COLD-3:</b> &nbsp; {html_badge(res["AI_Cold"], "badge-ai")}</div>
            <div class="info-row">📊 <b>Frequency COLD-3:</b> &nbsp; {html_badge(res["Freq_Cold"], "badge-stat")}</div>
            <div class="info-row">🔄 <b>Transition COLD-3:</b> &nbsp; {html_badge(res["Transition_Cold"], "badge-stat")}</div>
            <div class="info-row">🧩 <b>Pattern COLD-3:</b> &nbsp; {html_badge(res["Pattern_Cold"], "badge-stat")}</div>
            <div class="info-row">🧮 <b>Equation COLD-5:</b> &nbsp; {html_badge(res["Equation_Cold"], "badge-eq")}</div>
            <div style="font-size:13px; color:#777; margin-top:8px;">
                🧮 สมการผ่าน Stability: <b>{res["StableEquations"]}</b> / {res["TotalEquations"]} &nbsp; | &nbsp; Model Strength: <b>{res["EquationStrength"]:.0%}</b>
            </div>
            <div style="font-size:13px; color:#888; margin-top:8px;">📈 {res["BT"]}</div>
            <div style="font-size:13px; color:#777; margin-top:5px;">
                ⚖️ Dynamic Weight: AI {res["Weights"]["AI"]:.0%} | Frequency {res["Weights"]["Freq"]:.0%} | 
                Transition {res["Weights"]["ST"]:.0%} | Pattern {res["Weights"]["BT"]:.0%} | Equation {res["Weights"]["Eq"]:.0%}
            </div>
            """, unsafe_allow_html=True
        )
        st.write("")

    # ====================================================
    # OVERALL COLD (เลขดับภาพรวม)
    # ====================================================
    cold_top_overall = combine_cold_n(preds, ["H", "T", "O"], 7)
    cold_bottom_overall = combine_cold_n(preds, ["T2", "O2"], 7)

    st.subheader("❄️ สรุปเลขดับภาพรวม (ตัดทิ้ง)")

    st.markdown(
        f"""
        <div class="cold-card">
            <div style="font-weight:700; color:#1565c0;">❄️ COLD 7-TOP ดับบน (ความน่าจะเป็นต่ำ)</div>
            <div style="text-align:center; margin:10px 0;">{html_top5(cold_top_overall)}</div>
            <div style="font-size:13px; color:#546e7a; text-align:center;">{nums_prob(cold_top_overall)}</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="cold-card">
            <div style="font-weight:700; color:#1565c0;">❄️ COLD 7-TOP ดับล่าง (ความน่าจะเป็นต่ำ)</div>
            <div style="text-align:center; margin:10px 0;">{html_top5(cold_bottom_overall)}</div>
            <div style="font-size:13px; color:#546e7a; text-align:center;">{nums_prob(cold_bottom_overall)}</div>
        </div>
        """, unsafe_allow_html=True
    )

    # ====================================================
    # HISTORY 10 DRAWS (Backtest ผลเลขดับ)
    # ====================================================
    if past_records:
        st.write("")
        st.subheader("📜 ประวัติย้อนหลัง 10 งวด (Backtest เลขดับ 7-TOP)")
        
        for rec in past_records:
            top_mark = "✅ <span style='color:green;font-weight:bold;'>ดับอยู่ (ไม่มา)</span>" if rec["Top_Hit"] else "❌ <span style='color:red;'>ดับหลุด (ออกเต็มๆ)</span>"
            bot_mark = "✅ <span style='color:green;font-weight:bold;'>ดับอยู่ (ไม่มา)</span>" if rec["Bot_Hit"] else "❌ <span style='color:red;'>ดับหลุด (ออกเต็มๆ)</span>"
            
            top_nums = " • ".join(str(x) for x in rec["Cold_Top"])
            bot_nums = " • ".join(str(x) for x in rec["Cold_Bot"])
            
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #fafafa;">
                    <div style="font-weight: 700; color: #444; margin-bottom: 8px;">
                        📅 งวดวันที่ {rec['Date']}
                    </div>
                    <div style="font-size: 14px; margin-bottom: 4px;">
                        <b>บน ({rec['Result_3D']}):</b> ตัดเลข {top_nums} &nbsp;👉&nbsp; {top_mark}
                    </div>
                    <div style="font-size: 14px;">
                        <b>ล่าง ({rec['Result_2D']}):</b> ตัดเลข {bot_nums} &nbsp;👉&nbsp; {bot_mark}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    st.success("✅ วิเคราะห์เสร็จสิ้น • AI + Statistics + Equation Discovery + Strict Walk-Forward + Stability")
    st.caption("ระบบไม่มี Persistent Model — โมเดลและสมการคำนวณใหม่จากข้อมูลปัจจุบันทุกครั้งที่กดวิเคราะห์")
    st.caption("⚠️ เปอร์เซ็นต์ (Probability) เป็นคะแนนความน่าจะเป็นเชิงสถิติของโมเดล ไม่ใช่การรับประกันผลรางวัลจริง")
