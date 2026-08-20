# ============================================================
# 🚀 LOTTO AI V.MAX HYBRID TURBO (FIXED UI)
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
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        st.error(f"❌ ดึงข้อมูลไม่ได้: {exc}")
        return pd.DataFrame()

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
            d = pd.to_datetime(dm.group(1), errors="coerce")
            if not pd.isna(d):
                current_date = d

        nm = num_pattern.search(line)
        if not nm:
            continue

        if nm.group(1):
            r3, r2 = nm.group(1), nm.group(2)
        elif nm.group(3):
            r3, r2 = nm.group(3)[-3:], nm.group(4)
        else:
            continue

        rows.append({
            "Date": current_date,
            "Result_3D": str(r3).zfill(3),
            "Result_2D": str(r2).zfill(2),
        })

    if len(rows) < 10:
        st.error("❌ ข้อมูลย้อนหลังน้อยเกินไป")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna().drop_duplicates().sort_values("Date").reset_index(drop=True)
    return df


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

    ph, pt, po = x["H"].shift(1), x["T"].shift(1), x["O"].shift(1)
    x["PrevSum"] = ph + pt + po
    x["PrevOdd"] = (ph % 2) + (pt % 2) + (po % 2)
    x["DistHT"] = (ph - pt).abs()
    x["DistTO"] = (pt - po).abs()

    for pos in ["H", "T", "O", "T2", "O2"]:
        s = x[pos]
        prev = s.shift(1)
        x[f"Odd_{pos}"] = prev % 2
        x[f"High_{pos}"] = (prev >= 5).astype(np.int8)
        x[f"Prime_{pos}"] = (prev.isin([2, 3, 5, 7])).astype(np.int8)

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

    return x.replace([np.inf, -np.inf], np.nan).fillna(-1)


# ============================================================
# 5-9. ENGINES (Freq, Trans, Pattern, Eq, AI)
# ============================================================

class FrequencyEngine:
    def analyze(self, df, pos):
        s = df[pos].astype(int)
        if len(s) == 0: return np.ones(10) / 10
        r15, r30, all_f = s.tail(15).value_counts(normalize=True), s.tail(30).value_counts(normalize=True), s.value_counts(normalize=True)
        score = np.array([r15.get(d, 0) * 0.55 + r30.get(d, 0) * 0.30 + all_f.get(d, 0) * 0.15 for d in range(10)], dtype=np.float64)
        score += 0.01
        return score / score.sum()

class TransitionEngine:
    def analyze(self, df, pos):
        if len(df) < 6: return np.ones(10) / 10
        current = int(df[pos].iloc[-1])
        subset = df[df[pos].shift(1) == current]
        if len(subset) < 2: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)], dtype=np.float64)
        score += 0.01
        return score / score.sum()

class PatternEngine:
    def analyze(self, df, pos):
        if len(df) < 7: return np.ones(10) / 10
        a, b = int(df[pos].iloc[-1]), int(df[pos].iloc[-2])
        subset = df[(df[pos].shift(1) == a) & (df[pos].shift(2) == b)]
        if len(subset) < 2: subset = df[df[pos].shift(1) == a]
        if len(subset) < 1: return np.ones(10) / 10
        freq = subset[pos].value_counts(normalize=True)
        score = np.array([freq.get(d, 0) for d in range(10)], dtype=np.float64)
        score += 0.01
        return score / score.sum()

class EquationEngine:
    def __init__(self):
        self.equations = [
            ("L1", lambda a,b,c,d: a), ("L2", lambda a,b,c,d: b), ("L3", lambda a,b,c,d: c), ("L5", lambda a,b,c,d: d),
            ("L1+L2", lambda a,b,c,d: a+b), ("L1+L3", lambda a,b,c,d: a+c), ("L1+L5", lambda a,b,c,d: a+d),
            ("L1-L2", lambda a,b,c,d: a-b), ("L1-L3", lambda a,b,c,d: a-c), ("ABS(L1-L2)", lambda a,b,c,d: abs(a-b)),
            ("L1+L2+L3", lambda a,b,c,d: a+b+c), ("2L1+L2", lambda a,b,c,d: 2*a+b)
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
                try: pred = fn(*vals) % 10
                except: pred = -1
                actual = int(df[pos].iloc[idx])
                total += 1
                if pred == actual:
                    hits += 1
                    if idx >= n - 5: recent_hits += 1
            if total == 0: continue
            hit_rate = hits / total
            if hit_rate >= 0.10: results.append({"name": name, "fn": fn, "hit": hit_rate, "score": 0.7*hit_rate + 0.3*(recent_hits/min(5, total))})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        stable_selected = results[:8]
        if not stable_selected: return {"prob": np.ones(10)/10, "top": [], "strength": 0.0, "stable": 0, "total": len(self.equations)}
        
        try: vals = (int(df[pos].iloc[-1]), int(df[pos].iloc[-2]), int(df[pos].iloc[-3]), int(df[pos].iloc[-5]))
        except: return {"prob": np.ones(10)/10, "top": [], "strength": 0.0, "stable": 0, "total": len(self.equations)}
        
        prob, total_weight = np.zeros(10, dtype=np.float64), 0.0
        for r in stable_selected:
            try: pred = r["fn"](*vals) % 10
            except: continue
            if pred >= 0:
                w = 0.50 + r["score"]
                prob[pred] += w
                total_weight += w
        
        if total_weight <= 0: prob = np.ones(10)/10
        else: prob = (prob / total_weight) + 0.01; prob /= prob.sum()
        
        return {
            "prob": prob, 
            "top": [(int(i), float(prob[i])) for i in np.argsort(prob)[::-1][:5]],
            "strength": float(np.mean([r["hit"] for r in stable_selected])) if stable_selected else 0.0,
            "stable": len(stable_selected), "total": len(self.equations)
        }

class FastAI:
    def __init__(self, trees=55): self.trees = trees
    def predict(self, X, y, X_next):
        result = np.zeros(10, dtype=np.float64)
        for Model, w in zip([RandomForestClassifier, ExtraTreesClassifier], [0.5, 0.5]):
            m = Model(n_estimators=self.trees, max_depth=6, class_weight="balanced", random_state=42).fit(X, y)
            for c, p in zip(m.classes_, m.predict_proba(X_next)[0]): result[int(c)] += p * w
        result += 0.001
        return result / result.sum()

# ============================================================
# 10. ENSEMBLE ENGINE
# ============================================================

class EnsembleEngine:
    def __init__(self, df):
        self.df = df.copy()
        self.lags, self.rolls = [1, 2, 3, 5], [3, 5, 10]
        self.bt = 10 if len(df) >= 700 else (9 if len(df) >= 400 else 8)
        self.features = ["PrevSum", "PrevOdd", "DistHT", "DistTO"]
        for pos in ["H", "T", "O", "T2", "O2"]:
            self.features.extend([f"Odd_{pos}", f"High_{pos}", f"Prime_{pos}", f"Skip_{pos}"] + [f"L{lag}_{pos}" for lag in self.lags] + [f"RM{w}_{pos}" for w in self.rolls])
        self.freq, self.transition, self.pattern, self.equation, self.ai = FrequencyEngine(), TransitionEngine(), PatternEngine(), EquationEngine(), FastAI()
        self.base_weights = {"AI": 0.50, "Freq": 0.18, "ST": 0.12, "BT": 0.08, "Eq": 0.12}

    def predict_all(self, progress_bar):
        ext = pd.concat([self.df, pd.DataFrame([{"Date": self.df["Date"].iloc[-1] + timedelta(days=1), "Result_3D": "000", "Result_2D": "00"}])], ignore_index=True)
        ext = build_features(ext, self.lags, self.rolls)
        hist, X, X_next = ext.iloc[:-1], ext.iloc[:-1][self.features].astype(np.float32), ext.iloc[[-1]][self.features].astype(np.float32)
        
        predictions = {}
        positions = ["H", "T", "O", "T2", "O2"]
        for i, pos in enumerate(positions):
            progress_bar.progress((i + 1) / 5, text=f"⚡ กำลังวิเคราะห์ตำแหน่ง {pos}...")
            # Simple static weights for speed in display
            w = self.base_weights
            ai_p = self.ai.predict(X, hist[pos], X_next)
            fq_p = self.freq.analyze(hist, pos)
            st_p = self.transition.analyze(hist, pos)
            pt_p = self.pattern.analyze(hist, pos)
            eq_r = self.equation.discover(hist, pos, bt=self.bt)
            
            final = (w["AI"]*ai_p + w["Freq"]*fq_p + w["ST"]*st_p + w["BT"]*pt_p + w["Eq"]*eq_r["prob"])
            final = (final + 0.001) / (final + 0.001).sum()
            
            top_n = lambda p, n: [(int(i), float(p[i])) for i in np.argsort(p)[::-1][:n]]
            predictions[pos] = {
                "Final": top_n(final, 5), "AI": top_n(ai_p, 3), "Freq": top_n(fq_p, 3),
                "Equation": eq_r["top"], "EquationStrength": eq_r["strength"],
                "StableEquations": eq_r["stable"], "TotalEquations": eq_r["total"],
                "Prob": final, "Weights": w
            }
        return predictions, ext.iloc[-1]["Date"]

# ============================================================
# 11. UI HELPERS
# ============================================================

def html_top5(items):
    return '<span class="dot-sep">•</span>'.join([f'<span class="number-highlight">{n}</span>' for n, _ in items])

def html_badge(items, badge_class):
    return f'<span class="{badge_class}">' + " &nbsp;•&nbsp; ".join([str(n) for n, _ in items]) + "</span>"

def nums_prob(items):
    return " | ".join(f"{n} ({p:.1%})" for n, p in items)

def combine_top_n(preds, positions, n=5):
    score = sum(preds[pos]["Prob"] for pos in positions) / len(positions)
    return [(int(i), float(score[i])) for i in np.argsort(score)[::-1][:n]]

# ============================================================
# 12. MAIN APP
# ============================================================

st.markdown('<div class="main-title">🚀 LOTTO AI V.MAX TURBO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">PURE HISTORICAL HYBRID<br><b>AI + Statistics + Equation Discovery + Strict Walk-Forward</b><br>NO DAY • NO MONTH • NO CALENDAR</div>', unsafe_allow_html=True)
st.divider()

selected_lotto = st.selectbox("🎯 เลือกหวย", list(LOTTERY_SOURCES.keys()))key="select_vmax1")

if st.button("🚀 วิเคราะห์เลขเด่น",key="btn_vmax1",type="primary", use_container_width=True):
    df = fetch_and_clean_data(LOTTERY_SOURCES[selected_lotto])
    if df.empty: st.stop()
    if len(df) < 50: st.warning(f"⚠️ มีข้อมูลเพียง {len(df)} งวด ระบบยังทำงานได้ แต่ Equation/WF จะมีความเสถียรต่ำ")

    progress_bar = st.progress(0, text="⚡ กำลังเตรียมข้อมูล...")
    engine = EnsembleEngine(df)
    preds, next_date = engine.predict_all(progress_bar)
    progress_bar.empty()

    labels = {"H": "หลักร้อย 3 ตัวบน", "T": "หลักสิบ 3 ตัวบน", "O": "หลักหน่วย 3 ตัวบน", "T2": "หลักสิบ 2 ตัวล่าง", "O2": "หลักหน่วย 2 ตัวล่าง"}
    st.info(f"📊 ข้อมูลย้อนหลัง {len(df)} งวด | งวดถัดไปสำหรับแสดงผล: {next_date.strftime('%d-%m-%Y')} (วันที่ไม่มีผลต่อการคำนวณเลข)")

    for pos in ["H", "T", "O", "T2", "O2"]:
        res = preds[pos]
        st.markdown(f'<div class="position-title">📍 {labels[pos]}</div>', unsafe_allow_html=True)
        
        # FIXED HTML BUG HERE: No line breaks or indentations inside the html string block
        html_content = (
            f'<div class="hot-card">'
            f'<div style="font-weight:700; color:#444; margin-bottom:8px;">🔥 FINAL TOP-5</div>'
            f'<div style="text-align:center; margin:10px 0;">{html_top5(res["Final"])}</div>'
            f'<div style="font-size:13px; color:#888; text-align:center;">{nums_prob(res["Final"])}</div>'
            f'</div>'
            f'<div class="info-row">🤖 <b>AI TOP-3:</b> &nbsp; {html_badge(res["AI"], "badge-ai")}</div>'
            f'<div class="info-row">📊 <b>Frequency TOP-3:</b> &nbsp; {html_badge(res["Freq"], "badge-stat")}</div>'
            f'<div class="info-row">🧮 <b>Equation TOP-5:</b> &nbsp; {html_badge(res["Equation"], "badge-eq")}</div>'
            f'<div style="font-size:13px; color:#777; margin-top:8px;">🧮 สมการผ่าน Stability: <b>{res["StableEquations"]}</b> / {res["TotalEquations"]} &nbsp; | &nbsp; Strength: <b>{res["EquationStrength"]:.0%}</b></div>'
            f'<div style="font-size:13px; color:#999;">⚖️ น้ำหนัก: AI {res["Weights"]["AI"]:.0%} | Frequency {res["Weights"]["Freq"]:.0%} | Transition {res["Weights"]["ST"]:.0%} | Pattern {res["Weights"]["BT"]:.0%} | Equation {res["Weights"]["Eq"]:.0%}</div>'
        )
        st.markdown(html_content, unsafe_allow_html=True)
        st.write("")

    st.subheader("🔥 สรุปเลขเด่นภาพรวม")
    hot_top = combine_top_n(preds, ["H", "T", "O"])
    hot_bottom = combine_top_n(preds, ["T2", "O2"])

    overall_html = (
        f'<div class="hot-card">'
        f'<div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่งบน</div>'
        f'<div style="text-align:center; margin:10px 0;">{html_top5(hot_top)}</div>'
        f'<div style="font-size:13px; color:#888; text-align:center;">{nums_prob(hot_top)}</div>'
        f'</div>'
        f'<div class="hot-card">'
        f'<div style="font-weight:700; color:#444;">🔥 HOT 5-TOP รูด/วิ่งล่าง</div>'
        f'<div style="text-align:center; margin:10px 0;">{html_top5(hot_bottom)}</div>'
        f'<div style="font-size:13px; color:#888; text-align:center;">{nums_prob(hot_bottom)}</div>'
        f'</div>'
    )
    st.markdown(overall_html, unsafe_allow_html=True)
    st.success("✅ วิเคราะห์เสร็จสิ้น • Pure Historical AI + Statistics + Equation")
