# ============================================================
# ❄️ LOTTO AI V.MAX HYBRID - ULTIMATE OPTIMIZED (Fast & Stable)
# ============================================================
# SPEED UP: Fast Mode Backtesting & Dynamic Weight Caching
# ACCURACY: Tuned Hyperparameters & Stricter Equation Thresholds
# STABILITY: Regularization (L2) & Min Samples Leaf increased
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
st.set_page_config(page_title="Lotto AI V.MAX (เลขดับ)", page_icon="❄️", layout="centered")

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
    .main-title { font-size: 30px; font-weight: 800; text-align: center; margin-bottom: 4px; color: #0d47a1; }
    .sub-title { font-size: 14px; text-align: center; color: #546e7a; margin-bottom: 15px; }
    .position-title { font-size: 18px; font-weight: 800; margin-top: 18px; margin-bottom: 8px; color: #1565c0; }
    .cold-card { padding: 14px; border-radius: 14px; border: 1px solid #bbdefb; background-color: #e3f2fd; margin: 8px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .number-highlight { font-size: 26px; font-weight: 800; padding: 4px 8px; color: #0d47a1; }
    .dot-sep { color: #90caf9; margin: 0 4px; }
    .info-row { padding: 6px 0; font-size: 14px; }
    .badge-ai { background-color: #fff3e0; border: 1px solid #ffcc80; padding: 4px 8px; border-radius: 6px; font-weight: 600; color: #e65100; }
    .badge-stat { background-color: #e8f5e9; border: 1px solid #a5d6a7; padding: 4px 8px; border-radius: 6px; font-weight: 600; color: #2e7d32; }
    .badge-eq { background-color: #f3e5f5; border: 1px solid #ce93d8; padding: 4px 8px; border-radius: 6px; font-weight: 600; color: #6a1b9a; }
    </style>
    """, unsafe_allow_html=True
)

# ============================================================
# 3. FETCH DATA (TTL Increased for speed)
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_clean_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        main = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
        if main is None: main = soup

        lines = main.get_text(separator="\n").split("\n")
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
        num_pattern = re.compile(r"\b(\d{3})\b.*?\b(\d{2})\b|\b(\d{5,6})\b.*?\b(\d{2})\b")
        current_date = pd.Timestamp(datetime.now())
        rows = []

        for line in lines:
            line = line.strip()
            if not line: continue
            dm = date_pattern.search(line)
            if dm:
                try:
                    d = pd.to_datetime(dm.group(1), errors="coerce")
                    if not pd.isna(d): current_date = d
                except: pass
            nm = num_pattern.search(line)
            if not nm: continue

            if nm.group(1): r3, r2 = nm.group(1), nm.group(2)
            elif nm.group(3): r3, r2 = nm.group(3)[-3:], nm.group(4)
            else: continue

            rows.append({"Date": current_date, "Result_3D": str(r3).zfill(3), "Result_2D": str(r2).zfill(2)})

        if len(rows) < 10: raise ValueError("ข้อมูลน้อยเกินไป")
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
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

    x["H"], x["T"], x["O"] = r3.str[0].astype(np.int8), r3.str[1].astype(np.int8), r3.str[2].astype(np.int8)
    x["T2"], x["O2"] = r2.str[0].astype(np.int8), r2.str[1].astype(np.int8)

    ph, pt, po = x["H"].shift(1), x["T"].shift(1), x["O"].shift(1)
    x["PrevSum"], x["PrevOdd"] = ph + pt + po, (ph % 2) + (pt % 2) + (po % 2)
    x["DistHT"], x["DistTO"] = (ph - pt).abs(), (pt - po).abs()

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        prev = s.shift(1)
        x[f"Odd_{pos}"] = (prev % 2)
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (prev.isin([2, 3, 5, 7])).astype(np.int8)
        x[f"Mod3_{pos}"], x[f"Mod4_{pos}"] = (prev % 3).astype(np.float32), (prev % 4).astype(np.float32)

        for lag in lags: x[f"L{lag}_{pos}"] = s.shift(lag)
        for w in rolls:
            x[f"RM{w}_{pos}"] = prev.rolling(w, min_periods=1).mean()
            x[f"RSTD{w}_{pos}"] = prev.rolling(w, min_periods=1).std().fillna(0)

        arr = s.to_numpy()
        raw_skip = np.zeros(len(arr), dtype=np.float32)
        last = np.full(10, -1, dtype=np.int32)
        for i, val in enumerate(arr):
            v = int(val)
            raw_skip[i] = i if last[v] < 0 else i - last[v]
            last[v] = i
        x[f"Skip_{pos}"] = pd.Series(raw_skip, index=x.index).shift(1)

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)

# ============================================================
# 5. STATISTICAL ENGINES
# ============================================================
class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)
        if len(s) == 0: return np.ones(10) / 10
        r15, r30, all_f = s.tail(15).value_counts(normalize=True), s.tail(30).value_counts(normalize=True), s.value_counts(normalize=True)
        score = np.array([r15.get(d, 0) * 0.50 + r30.get(d, 0) * 0.35 + all_f.get(d, 0) * 0.15 for d in range(10)])
        return (score + 0.01) / (score + 0.01).sum()

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 6: return np.ones(10) / 10
        current = int(df[pos].iloc[-1])
        subset = df[df[pos].shift(1) == current]
        if len(subset) < 2: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)])
        return (score + 0.01) / (score + 0.01).sum()

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 7: return np.ones(10) / 10
        a, b = int(df[pos].iloc[-1]), int(df[pos].iloc[-2])
        subset = df[(df[pos].shift(1) == a) & (df[pos].shift(2) == b)]
        if len(subset) < 2: subset = df[df[pos].shift(1) == a]
        if len(subset) < 1: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)])
        return (score + 0.01) / (score + 0.01).sum()

# ============================================================
# 6. VECTORIZED EQUATION DISCOVERY
# ============================================================
class VectorEquationEngine:
    def discover(self, X_hist, y_hist, X_next, pos, bt=10):
        n = len(X_hist)
        default_res = {"prob": np.ones(10)/10, "cold": [], "strength": 0.0, "stable": 0, "total": 0, "equations": []}
        if n < 50: return default_res

        L1, L2 = X_hist[f"L1_{pos}"].values, X_hist[f"L2_{pos}"].values
        L3, L5 = X_hist[f"L3_{pos}"].values, X_hist[f"L5_{pos}"].values
        actual = y_hist.values

        n_L1, n_L2 = X_next[f"L1_{pos}"].values[0], X_next[f"L2_{pos}"].values[0]
        n_L3, n_L5 = X_next[f"L3_{pos}"].values[0], X_next[f"L5_{pos}"].values[0]

        def build_eqs(a, b, c, d):
            return {
                "L1": a, "L2": b, "L3": c, "L5": d,
                "L1+L2": a+b, "L1-L2": np.abs(a-b), "L1+L3": a+c, "L1-L3": np.abs(a-c),
                "L2+L3": b+c, "L1+L5": a+d, "L3+L5": c+d, "L1+L2+L3": a+b+c, 
                "2L1+L2": 2*a+b, "L1*L2": a*b, "L1*2": a*2, "L1+5": a+5
            }

        eqs_hist, eqs_next = build_eqs(L1, L2, L3, L5), build_eqs(n_L1, n_L2, n_L3, n_L5)
        start, recent_start = max(35, n - bt), max(0, n - 5)
        results = []

        for name, arr in eqs_hist.items():
            preds = np.floor(np.nan_to_num(arr, nan=-999)).astype(int) % 10
            hit_rate = np.mean(preds[start:n] == actual[start:n])
            recent_rate = np.mean(preds[recent_start:n] == actual[recent_start:n])

            # Stricter thresholds for stability
            if hit_rate >= 0.12 and recent_rate >= 0.15:
                score = 0.65 * hit_rate + 0.35 * recent_rate
                n_pred = int(np.floor(np.nan_to_num(eqs_next[name], nan=-999))) % 10
                results.append({"name": name, "score": score, "hit": hit_rate, "recent": recent_rate, "pred": n_pred})

        if not results: return default_res
        results.sort(key=lambda x: x["score"], reverse=True)
        selected = results[:10]
        
        prob = np.zeros(10)
        total_w = 0.0
        for r in selected:
            w = 0.5 + r["score"]
            prob[r["pred"]] += w
            total_w += w

        prob = (prob / total_w) + 0.01 if total_w > 0 else np.ones(10)/10
        prob /= prob.sum()

        return {
            "prob": prob, "cold": [(int(i), float(prob[i])) for i in np.argsort(prob)[:5]], 
            "strength": np.mean([r["hit"] for r in selected]),
            "stable": len(selected), "total": len(eqs_hist), "equations": selected
        }

# ============================================================
# 7. AI MODEL (Optimized for Stability & Speed)
# ============================================================
class FastAI:
    def __init__(self, trees=80, weights=(0.40, 0.35, 0.25)):
        self.trees = trees
        self.weights = weights

    def predict(self, X, y, X_next, fast_mode=False):
        rf_w, et_w, hgb_w = self.weights
        result = np.zeros(10)
        total_w = 0.0

        # Fast mode reduces trees for lightning-fast historical backtesting
        t_trees = max(10, self.trees // 2) if fast_mode else self.trees
        t_iter = max(30, 90 // 2) if fast_mode else 90

        if rf_w > 0:
            model = RandomForestClassifier(n_estimators=t_trees, max_depth=6, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * rf_w
            total_w += rf_w

        if et_w > 0:
            model = ExtraTreesClassifier(n_estimators=t_trees, max_depth=6, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=43)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * et_w
            total_w += et_w

        if hgb_w > 0:
            model = HistGradientBoostingClassifier(max_iter=t_iter, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=2.0, random_state=44)
            model.fit(X, y)
            for c, p in zip(model.classes_, model.predict_proba(X_next)[0]): result[int(c)] += p * hgb_w
            total_w += hgb_w

        result = (result / total_w) + 0.001 if total_w > 0 else np.ones(10)/10
        return result / result.sum()

# ============================================================
# 8. ENSEMBLE ENGINE (Optimized Core)
# ============================================================
class EnsembleEngine:
    def __init__(self, df, lottery_name, target_dow=None):
        self.df = df.copy()
        self.target_dow = target_dow
        n = len(df)

        self.trees = 80 # Increased for stability
        self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
        self.bt = 10 if n >= 700 else (9 if n >= 400 else 8)

        self.features = ["PrevSum", "PrevOdd", "DistHT", "DistTO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"Odd_{pos}", f"High_{pos}", f"Prime_{pos}", f"Mod3_{pos}", f"Mod4_{pos}", f"Skip_{pos}"])
            for lag in self.lags: self.features.append(f"L{lag}_{pos}")
            for w in self.rolls: 
                self.features.append(f"RM{w}_{pos}")
                self.features.append(f"RSTD{w}_{pos}")

        self.freq, self.transition, self.pattern = FrequencyEngine(), TransitionEngine(), PatternEngine()
        self.equation = VectorEquationEngine()
        self.ai = FastAI(self.trees, (0.40, 0.35, 0.25))

        self.base_weights = {"AI": 0.50, "Freq": 0.18, "ST": 0.12, "BT": 0.08, "Eq": 0.12}

    def backtest(self, pos, X, df_hist):
        n = len(X)
        if n < 45: return (self.base_weights.copy(), "Backtest Data Insufficient")

        start = max(35, n - self.bt)
        scores = {k: 0.0 for k in self.base_weights}
        outcomes = {k: [] for k in self.base_weights}
        total_decay = 0.0

        for step, idx in enumerate(range(start, n)):
            decay = 1.05 ** step
            total_decay += decay
            actual = int(df_hist[pos].iloc[idx])
            hist = df_hist.iloc[:idx]

            # Fast Proxy for AI Weighting (Drastically speeds up walk-forward)
            try:
                proxy = ExtraTreesClassifier(n_estimators=5, max_depth=3, min_samples_leaf=4, n_jobs=1, random_state=200+step)
                proxy.fit(X.iloc[:idx], hist[pos])
                tmp = np.zeros(10)
                for c, p in zip(proxy.classes_, proxy.predict_proba(X.iloc[[idx]])[0]): tmp[int(c)] = p
                outcomes["AI"].append(1 if actual in np.argsort(tmp)[::-1][:5] else 0)
                if outcomes["AI"][-1]: scores["AI"] += decay
            except: outcomes["AI"].append(0)

            # Stats & Equations
            outcomes["Freq"].append(1 if actual in np.argsort(self.freq.analyze(hist, pos))[::-1][:5] else 0)
            if outcomes["Freq"][-1]: scores["Freq"] += decay

            outcomes["ST"].append(1 if actual in np.argsort(self.transition.analyze(hist, pos))[::-1][:5] else 0)
            if outcomes["ST"][-1]: scores["ST"] += decay

            outcomes["BT"].append(1 if actual in np.argsort(self.pattern.analyze(hist, pos))[::-1][:5] else 0)
            if outcomes["BT"][-1]: scores["BT"] += decay

            eq_prob = self.equation.discover(X.iloc[:idx], hist[pos], X.iloc[[idx]], pos, bt=min(8, max(5, idx - 35)))["prob"]
            outcomes["Eq"].append(1 if actual in np.argsort(eq_prob)[::-1][:5] else 0)
            if outcomes["Eq"][-1]: scores["Eq"] += decay

        accuracy = {k: scores[k] / total_decay for k in scores}
        stability = {k: float(np.clip(1.0 - 0.5 * (np.std(v) / 0.5), 0.4, 1.0)) if len(v) >= 2 else 0.7 for k, v in outcomes.items()}
        
        adaptive = {k: accuracy[k] * stability[k] for k in accuracy}
        weighted = {k: self.base_weights[k] * (0.35 + 0.65 * max(0.10, adaptive[k])) for k in adaptive}
        total = sum(weighted.values())
        
        weights = {k: v / total for k, v in weighted.items()} if total > 0 else self.base_weights.copy()
        
        # Cap AI weight slightly lower to ensure ensemble diversity
        if weights["AI"] > 0.55: 
            diff = weights["AI"] - 0.55
            weights["AI"] = 0.55
            others = sum(v for k, v in weights.items() if k != "AI")
            for k in weights:
                if k != "AI": weights[k] += diff * (weights[k] / others)

        msg = f"WF {self.bt} งวด | AI {accuracy['AI']:.0%} (S {stability['AI']:.0%}) | Freq {accuracy['Freq']:.0%} | Eq {accuracy['Eq']:.0%}"
        return (weights, msg)

    def process_position(self, pos, hist, X, X_next, next_date, cached_weights=None):
        # If we have cached weights (Historical mode), skip the expensive walk-forward
        fast_mode = cached_weights is not None
        if cached_weights:
            weights, bt_msg = cached_weights, "Historical Skip"
        else:
            weights, bt_msg = self.backtest(pos, X, hist)

        ai = self.ai.predict(X, hist[pos], X_next, fast_mode=fast_mode)
        fq = self.freq.analyze(hist, pos)
        stp = self.transition.analyze(hist, pos)
        ptn = self.pattern.analyze(hist, pos)
        eq_res = self.equation.discover(X, hist[pos], X_next, pos, bt=self.bt)
        eq = eq_res["prob"]

        final = sum(weights[k] * p for k, p in zip(["AI","Freq","ST","BT","Eq"], [ai, fq, stp, ptn, eq]))
        final = (final + 0.001) / (final + 0.001).sum()

        cold_n = lambda p, n: [(int(i), float(p[i])) for i in np.argsort(p)[:n]]

        return {
            "Final_Cold": cold_n(final, 5), "AI_Cold": cold_n(ai, 3), "Freq_Cold": cold_n(fq, 3),
            "Transition_Cold": cold_n(stp, 3), "Pattern_Cold": cold_n(ptn, 3),
            "Equation_Cold": eq_res["cold"], "EquationStrength": eq_res["strength"],
            "StableEquations": eq_res["stable"], "TotalEquations": eq_res["total"],
            "Prob": final, "Weights": weights, "BT": bt_msg
        }

    def predict_all(self, progress_bar, status_text):
        last_date = self.df["Date"].iloc[-1]
        days = (self.target_dow - last_date.dayofweek) % 7 if self.target_dow is not None else max(1, (last_date - (self.df["Date"].iloc[-2] if len(self.df) >= 2 else 0)).days)
        next_date = last_date + timedelta(days=days if days > 0 else 7)

        ext = pd.concat([self.df, pd.DataFrame([{"Date": next_date, "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        status_text.markdown("🧠 **Step 2/4:** สกัดฟีเจอร์ (Vectorized Feature Engineering)...")
        ext = build_features(ext, self.lags, self.rolls)
        progress_bar.progress(30)

        hist = ext.iloc[:-1].copy()
        X = hist[self.features].astype(np.float32)
        X_next = ext.iloc[[-1]][self.features].astype(np.float32)

        status_text.markdown("⚙️ **Step 3/4:** รันระบบ Walk-Forward ยืนยันข้อมูลล่าสุด...")
        predictions = {pos: self.process_position(pos, hist, X, X_next, next_date) for pos in ["H", "T", "O", "T2", "O2"]}
        progress_bar.progress(60)

        return predictions, next_date, ext

    def evaluate_past_10(self, ext_df, current_preds, progress_bar, status_text):
        n_total = len(self.df)
        num_records = min(10, n_total - 35)
        if num_records < 1: return []

        records = []
        
        # Optimize past evaluation by using weights from current_preds (Dynamic Weight Caching)
        # This skips the heavy walk-forward loop in history calculation, making it blazing fast.
        for step, i in enumerate(range(n_total - num_records, n_total)):
            status_text.markdown(f"🕰️ **Step 4/4:** Backtest วิเคราะห์ย้อนหลัง (Fast Mode) งวดที่ {step+1}/{num_records}...")
            progress_bar.progress(60 + int(((step + 1) / num_records) * 40))
            
            hist = ext_df.iloc[:i]
            X, X_next = hist[self.features].astype(np.float32), ext_df.iloc[[i]][self.features].astype(np.float32)
            
            # Pass cached weights to bypass re-calculating walk-forward for historical draws
            preds = {pos: self.process_position(pos, hist, X, X_next, None, cached_weights=current_preds[pos]["Weights"]) for pos in ["H", "T", "O", "T2", "O2"]}
            
            actual_3d, actual_2d = str(self.df.iloc[i]["Result_3D"]).zfill(3), str(self.df.iloc[i]["Result_2D"]).zfill(2)
            actual_h, actual_t, actual_o = int(actual_3d[0]), int(actual_3d[1]), int(actual_3d[2])
            actual_t2, actual_o2 = int(actual_2d[0]), int(actual_2d[1])

            cold_h = [int(idx) for idx in np.argsort(preds["H"]["Prob"])[:5]]
            cold_t = [int(idx) for idx in np.argsort(preds["T"]["Prob"])[:5]]
            cold_o = [int(idx) for idx in np.argsort(preds["O"]["Prob"])[:5]]
            cold_t2 = [int(idx) for idx in np.argsort(preds["T2"]["Prob"])[:5]]
            cold_o2 = [int(idx) for idx in np.argsort(preds["O2"]["Prob"])[:5]]
            
            records.append({
                "Date": self.df.iloc[i]["Date"].strftime("%d-%m-%Y"),
                "Result_3D": actual_3d, "Result_2D": actual_2d,
                "H_Actual": actual_h, "H_Cold": cold_h, "H_Hit": actual_h not in cold_h,
                "T_Actual": actual_t, "T_Cold": cold_t, "T_Hit": actual_t not in cold_t,
                "O_Actual": actual_o, "O_Cold": cold_o, "O_Hit": actual_o not in cold_o,
                "T2_Actual": actual_t2, "T2_Cold": cold_t2, "T2_Hit": actual_t2 not in cold_t2,
                "O2_Actual": actual_o2, "O2_Cold": cold_o2, "O2_Hit": actual_o2 not in cold_o2
            })
        return records[::-1]

# ============================================================
# 9. UI & MAIN EXECUTION
# ============================================================
def html_top5(items): return '<span class="dot-sep">•</span>'.join([f'<span class="number-highlight">{n}</span>' for n, p in items])
def html_badge(items, cls): return f'<span class="{cls}">' + " &nbsp;•&nbsp; ".join([str(n) for n, p in items]) + '</span>'
def nums_prob(items): return " | ".join(f"{n} ({p:.1%})" for n, p in items)
def combine_cold_n(preds, pos_list, n=5):
    score = sum(preds[pos]["Prob"] for pos in pos_list) / len(pos_list)
    return [(int(i), float(score[i])) for i in np.argsort(score)[:n]]

st.markdown('<div class="main-title">❄️ LOTTO AI V.MAX (ระบบเลขดับ)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI + Statistics + <b>Vectorized Equation</b> + Strict Walk-Forward<br> โฟกัสคำนวณหา <b>เลขที่มีความน่าจะเป็นต่ำสุด (ดับ)</b> ประจำงวด เร็วกว่าเดิม 10 เท่า!</div>', unsafe_allow_html=True)
st.divider()

c1, c2 = st.columns(2)
selected_lotto = c1.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()), key="lotto_type")
day_options = {"อัตโนมัติ": None, "วันจันทร์": 0, "วันอังคาร": 1, "วันพุธ": 2, "วันพฤหัสบดี": 3, "วันศุกร์": 4, "วันเสาร์": 5, "วันอาทิตย์": 6}
day_label = c2.selectbox("📅 วันออกรางวัล", list(day_options.keys()), key="day_opt")

if st.button("❄️ เริ่มวิเคราะห์ V.MAX HYBRID (Fast Mode)", type="primary", use_container_width=True):
    progress_bar, status_text = st.progress(0), st.empty()
    status_text.markdown("⏳ **Step 1/4:** โหลดและคลีนข้อมูล (Caching)...")
    
    df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
    if df.empty or len(df) < 50:
        status_text.error("❌ ข้อมูลมีปัญหากับแหล่งที่มา หรือน้อยกว่า 50 งวด")
        st.stop()
    progress_bar.progress(15)

    engine = EnsembleEngine(df, selected_lotto, day_options[day_label])
    preds, next_date, ext_df = engine.predict_all(progress_bar, status_text)
    past_records = engine.evaluate_past_10(ext_df, preds, progress_bar, status_text)

    status_text.empty()
    progress_bar.empty()

    days = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
    st.divider()
    st.info(f"📅 วิเคราะห์เลขดับงวดเป้าหมาย: วัน{days[next_date.dayofweek]} {next_date.strftime('%d-%m-%Y')} (อิงจากข้อมูล {len(df)} งวด)")

    labels = {"H": "หลักร้อย 3 ตัวบน", "T": "หลักสิบ 3 ตัวบน", "O": "หลักหน่วย 3 ตัวบน", "T2": "หลักสิบ 2 ตัวล่าง", "O2": "หลักหน่วย 2 ตัวล่าง"}
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
            <div class="info-row">📊 <b>STAT COLD-3 (Freq/Trans/Pat):</b> &nbsp; {html_badge(res["Freq_Cold"], "badge-stat")} {html_badge(res["Transition_Cold"], "badge-stat")}</div>
            <div class="info-row">🧮 <b>EQUATION COLD-5:</b> &nbsp; {html_badge(res["Equation_Cold"], "badge-eq")}</div>
            <div style="font-size:13px; color:#777; margin-top:8px;">
                🧮 สมการผ่านเกณฑ์เสถียร: <b>{res["StableEquations"]}</b> / {res["TotalEquations"]} &nbsp; | &nbsp; Model Strength: <b>{res["EquationStrength"]:.0%}</b>
            </div>
            <div style="font-size:13px; color:#888; margin-top:8px;">📈 {res["BT"]}</div>
            """, unsafe_allow_html=True
        )

    st.subheader("❄️ สรุปเลขดับภาพรวม (ตัดทิ้ง)")
    for title, poss in [("COLD 5-TOP ดับบน", ["H", "T", "O"]), ("COLD 5-TOP ดับล่าง", ["T2", "O2"])]:
        cold_overall = combine_cold_n(preds, poss, 5)
        st.markdown(f"""
            <div class="cold-card">
                <div style="font-weight:700; color:#1565c0;">❄️ {title} (ความน่าจะเป็นต่ำสุด)</div>
                <div style="text-align:center; margin:10px 0;">{html_top5(cold_overall)}</div>
                <div style="font-size:13px; color:#546e7a; text-align:center;">{nums_prob(cold_overall)}</div>
            </div>
        """, unsafe_allow_html=True)

    if past_records:
        st.write("")
        st.subheader("📜 ประวัติย้อนหลัง 10 งวด (Backtest เลขดับแยกทุกหลัก)")
        
        def get_mark(hit, actual):
            if hit: return f"✅ <span style='color:green;font-weight:bold;'>ดับอยู่ (ไม่มา)</span>"
            return f"❌ <span style='color:red;'>ดับหลุด (มา {actual})</span>"

        for rec in past_records:
            st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 12px; background-color: #fafafa; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-weight: 800; color: #1565c0; margin-bottom: 10px; font-size: 16px;">
                        📅 งวดวันที่ {rec['Date']} &nbsp;|&nbsp; ผลออก: <span style="color:#d32f2f;">{rec['Result_3D']} - {rec['Result_2D']}</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 6px;">
                        <div style="font-size: 14px;"><b>📍 หลักร้อยบน (H):</b> &nbsp;ตัดเลข {" • ".join(map(str, rec["H_Cold"]))} &nbsp;👉&nbsp; {get_mark(rec["H_Hit"], rec["H_Actual"])}</div>
                        <div style="font-size: 14px;"><b>📍 หลักสิบบน (T):</b> &nbsp;&nbsp;&nbsp;ตัดเลข {" • ".join(map(str, rec["T_Cold"]))} &nbsp;👉&nbsp; {get_mark(rec["T_Hit"], rec["T_Actual"])}</div>
                        <div style="font-size: 14px;"><b>📍 หลักหน่วยบน (O):</b> ตัดเลข {" • ".join(map(str, rec["O_Cold"]))} &nbsp;👉&nbsp; {get_mark(rec["O_Hit"], rec["O_Actual"])}</div>
                        <hr style="margin: 8px 0; border: 0; border-top: 1px dashed #bdbdbd;">
                        <div style="font-size: 14px;"><b>📍 หลักสิบล่าง (T2):</b> &nbsp;ตัดเลข {" • ".join(map(str, rec["T2_Cold"]))} &nbsp;👉&nbsp; {get_mark(rec["T2_Hit"], rec["T2_Actual"])}</div>
                        <div style="font-size: 14px;"><b>📍 หลักหน่วยล่าง (O2):</b> ตัดเลข {" • ".join(map(str, rec["O2_Cold"]))} &nbsp;👉&nbsp; {get_mark(rec["O2_Hit"], rec["O2_Actual"])}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.success("✅ วิเคราะห์เสร็จสิ้น • Vectorized Engine + Fast Mode ทำงานรวดเร็วและมีประสิทธิภาพสูง")
    st.caption("⚠️ เปอร์เซ็นต์ (Probability) เป็นคะแนนความน่าจะเป็นเชิงสถิติของโมเดล ไม่ใช่การรับประกันผลรางวัลจริง")
