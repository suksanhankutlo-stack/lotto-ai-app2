# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID TURBO (OPTIMIZED & ADAPTIVE) - V2
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

st.set_page_config(page_title="Lotto AI V.MAX Hybrid Turbo", page_icon="🚀", layout="centered")

LOTTERY_SOURCES = {
    "1. หวยไทย": "https://suksan18190.blogspot.com/2026/07/blog-post_07.html",
    "2. หวยธกส.": "https://suksan18190.blogspot.com/2026/07/blog-post_12.html",
    "3. หวยออมสิน": "https://suksan18190.blogspot.com/2026/07/blog-post_525.html",
    "4. หวยลาว": "https://suksan18190.blogspot.com/2026/07/blog-post.html",
    "5. หวยฮานอย": "https://suksan18190.blogspot.com/2026/07/blog-post_08.html",
}

# ============================================================
# 1. UI CSS
# ============================================================

st.markdown("""
<style>
.main-title { text-align:center; font-size:28px; font-weight:900; color:#D32F2F; }
.sub-title { text-align:center; color:#555; font-size:14px; margin-bottom:20px; }
.hot-card { padding:18px; border-radius:16px; border:2px solid #ff4b4b; margin:10px 0; background:linear-gradient(to bottom right,#ffffff,#fff5f5); }
.number-highlight { font-size:36px; font-weight:900; color:#D32F2F; text-shadow:1px 1px 2px rgba(0,0,0,0.15); letter-spacing:2px; }
.dot-sep { color:#FFCDD2; font-size:26px; margin:0 10px; }
.badge-ai { background:#E3F2FD; color:#1565C0; padding:4px 12px; border-radius:15px; font-weight:800; font-size:16px; border:1px solid #BBDEFB; }
.badge-stat { background:#E8F5E9; color:#2E7D32; padding:4px 12px; border-radius:15px; font-weight:800; font-size:16px; border:1px solid #C8E6C9; }
.badge-eq { background:#F3E5F5; color:#7B1FA2; padding:4px 12px; border-radius:15px; font-weight:800; font-size:16px; border:1px solid #E1BEE7; }
.position-title { font-size:20px; font-weight:800; margin-top:20px; color:#333; border-bottom:2px solid #eee; padding-bottom:5px; }
.info-row { margin:8px 0; font-size:15px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. FETCH DATA & HASHING
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_and_clean_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except:
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    main = soup.find("div", class_=re.compile(r"post-body|entry-content|post-content|content"))
    if not main: main = soup

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
            d = pd.to_datetime(dm.group(1), errors="coerce")
            if not pd.isna(d): current_date = d
        
        nm = num_pattern.search(line)
        if not nm: continue

        if nm.group(1): r3, r2 = nm.group(1), nm.group(2)
        elif nm.group(3): r3, r2 = nm.group(3)[-3:], nm.group(4)
        else: continue

        rows.append({"Date": current_date, "Result_3D": str(r3).zfill(3), "Result_2D": str(r2).zfill(2)})

    df = pd.DataFrame(rows)
    if df.empty: return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)

def get_data_hash(df):
    data_str = "".join(df["Result_3D"].tolist() + df["Result_2D"].tolist())
    return hashlib.md5(data_str.encode()).hexdigest()

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def build_adaptive_features(df, lags, rolls):
    x = df.copy()
    r3, r2 = x["Result_3D"].astype(str), x["Result_2D"].astype(str)

    x["H"], x["T"], x["O"] = r3.str[0].astype(np.int8), r3.str[1].astype(np.int8), r3.str[2].astype(np.int8)
    x["T2"], x["O2"] = r2.str[0].astype(np.int8), r2.str[1].astype(np.int8)

    # Base Features
    ph, pt, po = x["H"].shift(1), x["T"].shift(1), x["O"].shift(1)
    x["PrevSum"] = ph + pt + po
    x["PrevOdd"] = (ph % 2) + (pt % 2) + (po % 2)
    x["DistHT"] = (ph - pt).abs()
    x["DistTO"] = (pt - po).abs()

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        for lag in lags:
            x[f"L{lag}_{pos}"] = s.shift(lag)
        for w in rolls:
            x[f"RM{w}_{pos}"] = s.shift(1).rolling(w, min_periods=1).mean()

        arr = s.to_numpy()
        skip = np.zeros(len(arr), dtype=np.float32)
        last = np.full(10, -1, dtype=np.int32)
        for i, val in enumerate(arr):
            v = int(val)
            skip[i] = i if last[v] < 0 else i - last[v]
            last[v] = i
        x[f"Skip_{pos}"] = skip

    # ปรับแก้: ใช้ dropna() ลบ row ที่เกิด NaN จากการทำ Lag ทิ้งไป (ลบ fillna)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.dropna().reset_index(drop=True)

# ============================================================
# 4. ENGINES
# ============================================================

class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)
        if len(s) == 0: return np.ones(10) / 10
        r15, all_f = s.tail(15).value_counts(normalize=True), s.value_counts(normalize=True)
        score = np.array([r15.get(d, 0)*0.7 + all_f.get(d, 0)*0.3 for d in range(10)], dtype=np.float64)
        return (score + 0.01) / (score + 0.01).sum()

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 6: return np.ones(10) / 10
        current = int(df[pos].iloc[-1])
        subset = df[df[pos].shift(1) == current]
        if len(subset) < 1: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)], dtype=np.float64)
        return (score + 0.01) / (score + 0.01).sum()

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 7: return np.ones(10) / 10
        a, b = int(df[pos].iloc[-1]), int(df[pos].iloc[-2])
        subset = df[(df[pos].shift(1) == a) & (df[pos].shift(2) == b)]
        if len(subset) < 1: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)], dtype=np.float64)
        return (score + 0.01) / (score + 0.01).sum()

class EquationEngine:
    def __init__(self):
        self.equations = [
            ("L1", lambda a,b,c,d: a), ("L2", lambda a,b,c,d: b), 
            ("L3", lambda a,b,c,d: c), ("L5", lambda a,b,c,d: d),
            ("L1+L2", lambda a,b,c,d: a+b), ("L1+L3", lambda a,b,c,d: a+c),
            ("ABS(L1-L2)", lambda a,b,c,d: abs(a-b))
        ]
    def discover(self, df, pos, bt=10):
        n = len(df)
        if n < 50: return {"prob": np.ones(10)/10, "top": [], "strength": 0.0, "stable": 0, "total": len(self.equations)}
        
        start = max(35, n - bt)
        results = []
        for name, fn in self.equations:
            hits, total, recent_hits = 0, 0, 0
            for idx in range(start, n):
                try: vals = (int(df[pos].iloc[idx-1]), int(df[pos].iloc[idx-2]), int(df[pos].iloc[idx-3]), int(df[pos].iloc[idx-5]))
                except: continue
                pred = fn(*vals) % 10
                actual = int(df[pos].iloc[idx])
                total += 1
                if pred == actual:
                    hits += 1
                    if idx >= n - 3: recent_hits += 1
            if total > 0 and hits/total >= 0.10: 
                results.append({"name": name, "fn": fn, "hit": hits/total, "score": (hits/total) + (recent_hits*0.1)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        stable = results[:5]
        prob = np.zeros(10, dtype=np.float64)
        total_w = 0.0
        
        try: vals = (int(df[pos].iloc[-1]), int(df[pos].iloc[-2]), int(df[pos].iloc[-3]), int(df[pos].iloc[-5]))
        except: vals = (0,0,0,0)
        
        for r in stable:
            pred = r["fn"](*vals) % 10
            prob[pred] += r["score"]
            total_w += r["score"]
            
        if total_w <= 0: prob = np.ones(10)/10
        else: prob = (prob / total_w) + 0.01; prob /= prob.sum()
        
        return {"prob": prob, "top": [(int(i), float(prob[i])) for i in np.argsort(prob)[::-1][:5]],
                "strength": float(np.mean([r["hit"] for r in stable])) if stable else 0.0,
                "stable": len(stable), "total": len(self.equations)}

class FastAI:
    def __init__(self, n_samples):
        if n_samples < 100: self.trees, self.depth = 30, 4
        elif n_samples < 200: self.trees, self.depth = 50, 5
        elif n_samples < 400: self.trees, self.depth = 70, 6
        else: self.trees, self.depth = 100, 7

    def predict(self, X, y, X_next):
        result = np.zeros(10, dtype=np.float64)
        
        models = [
            (RandomForestClassifier(n_estimators=self.trees, max_depth=self.depth, random_state=42), 0.4),
            (ExtraTreesClassifier(n_estimators=self.trees, max_depth=self.depth, random_state=42), 0.3),
            (HistGradientBoostingClassifier(max_iter=self.trees, max_depth=self.depth, random_state=42), 0.3)
        ]
        
        for Model, w in models:
            m = Model.fit(X, y)
            for c, p in zip(m.classes_, m.predict_proba(X_next)[0]): 
                result[int(c)] += p * w
                
        result += 0.001
        return result / result.sum()

# ============================================================
# 5. ENSEMBLE ENGINE & DYNAMIC WEIGHTS
# ============================================================

class EnsembleEngine:
    def __init__(self, df):
        self.df = df.copy()
        n = len(df)
        
        if n < 100: self.lags, self.rolls = [1, 2], [3]
        elif n < 200: self.lags, self.rolls = [1, 2, 3], [3, 5]
        else: self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
        
        self.bt = 10 if n >= 700 else (8 if n >= 200 else 6)
        
        self.weights = {
            "H": {"AI": 0.40, "Freq": 0.25, "ST": 0.15, "BT": 0.10, "Eq": 0.10},
            "T": {"AI": 0.45, "Freq": 0.20, "ST": 0.15, "BT": 0.10, "Eq": 0.10},
            "O": {"AI": 0.50, "Freq": 0.15, "ST": 0.15, "BT": 0.10, "Eq": 0.10},
            "T2": {"AI": 0.45, "Freq": 0.20, "ST": 0.10, "BT": 0.10, "Eq": 0.15},
            "O2": {"AI": 0.50, "Freq": 0.15, "ST": 0.10, "BT": 0.10, "Eq": 0.15},
        }

    def predict_all(self):
        ext = pd.concat([self.df, pd.DataFrame([{"Date": self.df["Date"].iloc[-1] + timedelta(days=1), "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        ext = build_adaptive_features(ext, self.lags, self.rolls)
        
        base_cols = ["PrevSum", "PrevOdd", "DistHT", "DistTO"]
        pos_cols = []
        for pos in ["H", "T", "O", "T2", "O2"]:
            pos_cols.extend([f"Skip_{pos}"] + [f"L{l}_{pos}" for l in self.lags] + [f"RM{w}_{pos}" for w in self.rolls])
        self.features = base_cols + pos_cols

        hist, X, X_next = ext.iloc[:-1], ext.iloc[:-1][self.features].astype(np.float32), ext.iloc[[-1]][self.features].astype(np.float32)
        
        freq, transition, pattern = FrequencyEngine(), TransitionEngine(), PatternEngine()
        eq, ai = EquationEngine(), FastAI(len(self.df))
        
        predictions = {}
        for pos in ["H", "T", "O", "T2", "O2"]:
            w = self.weights[pos]
            ai_p = ai.predict(X, hist[pos], X_next)
            fq_p = freq.analyze(hist, pos)
            st_p = transition.analyze(hist, pos)
            pt_p = pattern.analyze(hist, pos)
            eq_r = eq.discover(hist, pos, bt=self.bt)
            
            final = (w["AI"]*ai_p + w["Freq"]*fq_p + w["ST"]*st_p + w["BT"]*pt_p + w["Eq"]*eq_r["prob"])
            final = (final + 0.001) / (final + 0.001).sum()
            
            top_n = lambda p, n: [(int(i), float(p[i])) for i in np.argsort(p)[::-1][:n]]
            predictions[pos] = {
                "Final": top_n(final, 5), "AI": top_n(ai_p, 3), "Freq": top_n(fq_p, 3),
                "Equation": eq_r["top"], "EqStr": eq_r["strength"], "W": w
            }
        return predictions, ext.iloc[-1]["Date"]

# ============================================================
# 6. CACHE PIPELINE
# ============================================================

@st.cache_data(show_spinner=False)
def run_prediction_pipeline(data_hash, df):
    engine = EnsembleEngine(df)
    return engine.predict_all()

# ============================================================
# 7. UI HELPERS & APP MAIN
# ============================================================

def html_top5(items): return '<span class="dot-sep">•</span>'.join([f'<span class="number-highlight">{n}</span>' for n, _ in items])
def html_badge(items, cls): return f'<span class="{cls}">' + " &nbsp;•&nbsp; ".join([str(n) for n, _ in items]) + "</span>"

st.markdown('<div class="main-title">🚀 LOTTO AI V.MAX TURBO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">OPTIMIZED HASH PIPELINE<br><b>Adaptive AI + HistGradientBoosting + Dynamic Weights</b></div>', unsafe_allow_html=True)
st.divider()

selected_lotto = st.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()))

if st.button("🚀 วิเคราะห์เลขเด่น", type="primary", use_container_width=True):
    df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
    
    # ปรับแก้: เพิ่มระบบแจ้งเตือนกรณีดึงข้อมูลล้มเหลว
    if df.empty: 
        st.error("🚨 ไม่สามารถดึงข้อมูลจากแหล่งที่มาได้ หรือรูปแบบเว็บต้นทางมีการเปลี่ยนแปลง กรุณาลองใหม่อีกครั้ง")
        st.stop()
    
    current_hash = get_data_hash(df)
    
    with st.spinner("⚡ กำลังประมวลผล (หรือโหลดจาก Cache ถ้าข้อมูลไม่เปลี่ยน)..."):
        preds, next_date = run_prediction_pipeline(current_hash, df)

    labels = {"H": "หลักร้อย 3 ตัวบน", "T": "หลักสิบ 3 ตัวบน", "O": "หลักหน่วย 3 ตัวบน", "T2": "หลักสิบ 2 ตัวล่าง", "O2": "หลักหน่วย 2 ตัวล่าง"}
    st.info(f"📊 ข้อมูล {len(df)} งวด | Data Hash: {current_hash[:8]} | งวดถัดไป: {next_date.strftime('%d-%m-%Y')}")

    for pos in ["H", "T", "O", "T2", "O2"]:
        res = preds[pos]
        st.markdown(f'<div class="position-title">📍 {labels[pos]}</div>', unsafe_allow_html=True)
        html_content = (
            f'<div class="hot-card">'
            f'<div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 FINAL TOP-5</div>'
            f'<div style="text-align:center; margin:10px 0;">{html_top5(res["Final"])}</div>'
            f'</div>'
            f'<div class="info-row">🤖 <b>AI TOP-3:</b> &nbsp; {html_badge(res["AI"], "badge-ai")}</div>'
            f'<div class="info-row">📊 <b>Frequency TOP-3:</b> &nbsp; {html_badge(res["Freq"], "badge-stat")}</div>'
            f'<div class="info-row">🧮 <b>Equation TOP-5:</b> &nbsp; {html_badge(res["Equation"], "badge-eq")} (ความแม่นยำ {res["EqStr"]:.0%})</div>'
            f'<div style="font-size:13px; color:#999; margin-top:5px;">⚖️ Weight: AI {res["W"]["AI"]:.0%} | Freq {res["W"]["Freq"]:.0%} | Eq {res["W"]["Eq"]:.0%}</div>'
        )
        st.markdown(html_content, unsafe_allow_html=True)

    st.subheader("🔥 สรุปเลขเด่นภาพรวม")
    
    # ปรับแก้: ลบตัวแปร score_top ที่ไม่ได้ใช้งานและติดบั๊กทิ้งไป
    
    def get_overall(positions):
        score = np.zeros(10)
        for pos in positions:
            for n, p in preds[pos]["Final"]: score[n] += p
        return [(int(i), float(score[i])) for i in np.argsort(score)[::-1][:5]]

    hot_top, hot_bottom = get_overall(["H", "T", "O"]), get_overall(["T2", "O2"])

    overall_html = (
        f'<div class="hot-card"><div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่งบน</div>'
        f'<div style="text-align:center; margin:10px 0;">{html_top5(hot_top)}</div></div>'
        f'<div class="hot-card"><div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่งล่าง</div>'
        f'<div style="text-align:center; margin:10px 0;">{html_top5(hot_bottom)}</div></div>'
    )
    st.markdown(overall_html, unsafe_allow_html=True)
    st.success("✅ ประมวลผลเสร็จสิ้น (Hash Check & Clean Data Models Applied)")
