import argparse
import json
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

@dataclass(frozen=True)
class Coin:
    label: str
    value: float   # dirhams
    kind: str      # display colour: gold, silver, bimetal


COINS = [
    Coin("10 cts", 0.10, "gold"),
    Coin("20 cts", 0.20, "gold"),
    Coin("1/2 DH", 0.50, "silver"),
    Coin("1 DH", 1.00, "silver"),
    Coin("2 DH", 2.00, "silver"),
    Coin("5 DH", 5.00, "bimetal"),
    Coin("10 DH", 10.0, "bimetal"),
]
BY_LABEL = {c.label: c for c in COINS}
SAMPLES_FILE = "coin_samples.json"
MAX_SAMPLES_PER_COIN = 10
BANDS = [(0.0, 0.30), (0.30, 0.55), (0.55, 0.80), (0.80, 0.95)]

# Layout
W, H = 1280, 720
FONT, DUPLEX, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
BG, HEADER, CARD, CARD2 = (30, 26, 24), (38, 33, 30), (46, 41, 38), (62, 56, 52)
HOVER, BORDER, TRACK = (84, 76, 70), (72, 66, 62), (88, 80, 76)
TEXT, MUTED, WHITE = (238, 234, 232), (160, 152, 146), (255, 255, 255)
ACCENT = (248, 189, 56)
KIND_COLORS = {"gold": (50, 190, 245), "silver": (215, 215, 215), "bimetal": (240, 130, 200)}
UNKNOWN_COLOR = (80, 80, 255)
WARN = (80, 180, 255)

VIDEO = (16, 80, 860, 624)
RESULT = (892, 80, 372, 300)
SETTINGS = (892, 392, 372, 312)


@dataclass
class Slider:
    key: str
    label: str
    lo: float
    hi: float
    default: float
    step: float = 1


SLIDERS = [
    Slider("edge", "Edge contrast (Hough param1)", 20, 200, 80),
    Slider("sens", "Sensitivity (lower = more circles)", 10, 100, 30),
    Slider("min_r", "Min radius (px)", 10, 200, 30),
    Slider("max_r", "Max radius (px)", 20, 400, 90),
    Slider("support", "Min border support", 0.3, 0.95, 0.65, 0.05),
    Slider("strict", "Match tolerance (higher = looser)", 2.0, 12.0, 5.0, 0.5),
]


# Data structures
@dataclass
class Feat:
    diam: float                 # refined diameter in pixels
    ab: np.ndarray              # 8 values: a* and b* per band 
    rel: np.ndarray             # 4 values: lightness per band minus mean lightness
    L: float                    # mean lightness


@dataclass
class Sample:
    label: str
    feat: Feat

    def to_json(self):
        f = self.feat
        return {"label": self.label, "diam": f.diam, "ab": f.ab.tolist(), "rel": f.rel.tolist(), "L": f.L}

    @staticmethod
    def from_json(d):
        return Sample(d["label"], Feat(float(d["diam"]), np.array(d["ab"], np.float32),
                                       np.array(d["rel"], np.float32), float(d["L"])))


@dataclass
class Det:
    x: float
    y: float
    r: float
    support: float
    feat: Feat = None
    coin: Coin = None
    closest: str = ""
    dist: float = 0.0
    second: float = 0.0


# Detection: Hough proposal, radial border refinement, duplicate removal
def hough_candidates(gray, v):
    h, w = gray.shape
    f = min(1.0, 960.0 / w)
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    small = cv2.GaussianBlur(small, (9, 9), 2)
    lo = max(3, int(v["min_r"] * f * 0.9))
    hi = max(lo + 2, int(v["max_r"] * f * 1.1))
    c = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(4, lo * 0.3),
                         param1=float(v["edge"]), param2=float(v["sens"]), minRadius=lo, maxRadius=hi)
    if c is None:
        return []
    return [(float(x) / f, float(y) / f, float(r) / f) for x, y, r in c[0][:80]]


def _refine_once(gray, x, y, r, min_edge=3.0):
    h, w = gray.shape
    big_r = r * 1.3
    wc = int(big_r * 2)
    if wc < 20:
        return None
    polar = cv2.warpPolar(gray, (wc, 360), (float(x), float(y)), float(big_r),
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR).astype(np.float32)
    polar = cv2.GaussianBlur(polar, (0, 0), sigmaX=2.0, sigmaY=0.8)
    g = np.abs(cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)) / 8.0

    ang = np.arange(360) * (2 * np.pi / 360)
    px, py = x + 1.12 * r * np.cos(ang), y + 1.12 * r * np.sin(ang)
    valid = (px > 2) & (px < w - 3) & (py > 2) & (py < h - 3)
    if valid.sum() < 90:
        return None

    c0, c1 = int(0.88 * r / big_r * wc), int(1.12 * r / big_r * wc)
    seg = g[:, c0:c1]
    best = np.argmax(seg, axis=1)                          # strongest edge on each ray
    ok = (seg[np.arange(360), best] > min_edge) & valid
    if ok.sum() < 40:
        return None
    rad = (c0 + best) * big_r / wc

    inl = ok.copy()
    cx, cy, rr = float(x), float(y), float(np.median(rad[ok]))
    for _ in range(3):                                    # robust circle fit, outliers dropped each pass
        res = np.abs(np.hypot(x + rad * np.cos(ang) - cx, y + rad * np.sin(ang) - cy) - rr) if _ else \
            np.abs(rad - rr)
        if _:
            pass
        thr = max(1.5, 0.025 * rr)
        inl = ok & (res < thr)
        if inl.sum() < 40:
            return None
        ex, ey = x + rad[inl] * np.cos(ang[inl]), y + rad[inl] * np.sin(ang[inl])
        A = np.column_stack([2 * ex, 2 * ey, np.ones_like(ex)])
        sol = np.linalg.lstsq(A, ex ** 2 + ey ** 2, rcond=None)[0]
        cx, cy = float(sol[0]), float(sol[1])
        rr = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    if math.hypot(cx - x, cy - y) > 0.4 * r or not (0.7 * r < rr < 1.3 * r):
        return None
    return cx, cy, rr, float(inl.sum()) / float(valid.sum())


def expand_circle(gray, cx, cy, rr, min_edge=3.0):
    h, w = gray.shape
    big_r = rr * 1.95
    wc = int(big_r * 2)
    polar = cv2.warpPolar(gray, (wc, 360), (float(cx), float(cy)), float(big_r),
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR).astype(np.float32)
    polar = cv2.GaussianBlur(polar, (0, 0), sigmaX=2.0, sigmaY=0.8)
    g = np.abs(cv2.Sobel(polar, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
    ang = np.arange(360) * (2 * np.pi / 360)
    px, py = cx + 1.85 * rr * np.cos(ang), cy + 1.85 * rr * np.sin(ang)
    valid = (px > 2) & (px < w - 3) & (py > 2) & (py < h - 3)
    if valid.sum() < 270:
        return None
    col = int(rr / big_r * wc)
    s0 = float(np.median(g[:, max(0, col - 2):col + 3].max(axis=1)[valid]))
    thr = max(2.0 * min_edge, 0.25 * s0)
    peak = (g > thr)
    peak[:, 1:-1] &= (g[:, 1:-1] >= g[:, :-2]) & (g[:, 1:-1] >= g[:, 2:])
    tol = max(2, int(0.03 * rr / big_r * wc))
    near = cv2.dilate(peak.astype(np.uint8), np.ones((1, 2 * tol + 1), np.uint8))
    coh = (near[valid] > 0).mean(axis=0)
    c0, c1 = int(1.2 * rr / big_r * wc), int(1.85 * rr / big_r * wc)
    k = int(np.argmax(coh[c0:c1])) + c0
    if coh[k] < 0.85:
        return None
    rho = k * big_r / wc
    res = _refine_once(gray, cx, cy, rho)
    if res is not None and res[3] >= 0.75 and abs(res[2] - rho) < 0.06 * rho:
        return res
    return None


def refine_circle(gray, x, y, r):
    res = _refine_once(gray, x, y, r)
    if res is None:
        return None
    res2 = _refine_once(gray, res[0], res[1], res[2])
    res = res2 if res2 is not None else res
    big = expand_circle(gray, *res[:3])
    return big if big is not None else res


def merge_circles(circs):
    kept = []
    for x, y, r, s in sorted(circs, key=lambda c: -c[3]):
        drop = False
        for i, (kx, ky, kr, ks) in enumerate(kept):
            d = math.hypot(x - kx, y - ky)
            if d < 0.25 * max(r, kr):
                if r > kr * 1.1:
                    kept[i] = (x, y, r, s)
                drop = True
                break
            if d < 0.75 * (r + kr):
                drop = True
                break
        if not drop:
            kept.append((x, y, r, s))
    return kept


# Features, learning and matching
def background_ab(lab, circles):
    small = lab[::4, ::4]
    mask = np.full(small.shape[:2], 255, np.uint8)
    for x, y, r, _ in circles:
        cv2.circle(mask, (int(x / 4), int(y / 4)), int(r * 1.15 / 4) + 1, 0, -1)
    pix = small[mask > 0]
    if len(pix) < 50:
        pix = small.reshape(-1, 3)
    return float(np.median(pix[:, 1])) - 128.0, float(np.median(pix[:, 2])) - 128.0


def extract_features(lab, x, y, r, bg):
    h, w = lab.shape[:2]
    x0, x1, y0, y1 = max(0, int(x - r)), min(w, int(x + r) + 1), max(0, int(y - r)), min(h, int(y + r) + 1)
    crop = lab[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y) / r
    valid = (crop[..., 0] < 235) & (crop[..., 0] > 15)         
    Ls, abs_ = [], []
    for lo, hi in BANDS:
        m = (d >= lo) & (d < hi) & valid
        if m.sum() < 12:
            m = (d >= lo) & (d < hi)
        if m.sum() < 4:
            return None
        Ls.append(float(np.median(crop[..., 0][m])))
        abs_.append((float(np.median(crop[..., 1][m])) - 128.0 - bg[0],
                     float(np.median(crop[..., 2][m])) - 128.0 - bg[1]))
    Lm = float(np.mean(Ls))
    ab = np.array([p[0] for p in abs_] + [p[1] for p in abs_], np.float32)
    return Feat(2 * r, ab, np.array(Ls, np.float32) - Lm, Lm)


def feat_distance(f, s):
    dz = (f.diam - s.diam) / (0.025 * s.diam)                
    az = (f.ab - s.ab) / 6.0
    lz = (f.rel - s.rel) / 18.0
    Lz = (f.L - s.L) / 45.0
    return math.sqrt(4 * dz * dz + 2 * float(np.mean(az ** 2)) + 0.5 * float(np.mean(lz ** 2)) + 0.25 * Lz * Lz)


def match(feat, samples):
    best = {}
    for s in samples:
        d = feat_distance(feat, s.feat)
        if d < best.get(s.label, 1e9):
            best[s.label] = d
    if not best:
        return "", 0.0, 0.0
    rk = sorted(best.items(), key=lambda kv: kv[1])
    return rk[0][0], rk[0][1], (rk[1][1] if len(rk) > 1 else 0.0)


def analyse(frame, S):
    v = dict(S.vals)
    if S.auto_r and S.samples:
        dm = [s.feat.diam for s in S.samples]
        v["min_r"], v["max_r"] = int(0.9 * min(dm) / 2), int(1.1 * max(dm) / 2) + 1
        S.vals["min_r"] = min(max(v["min_r"], 10), 200)
        S.vals["max_r"] = min(max(v["max_r"], 20), 400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good, rejected = [], []
    for x, y, r in hough_candidates(gray, v):
        res = refine_circle(gray, x, y, r)
        if res is None or res[3] < v["support"] or not (v["min_r"] * 0.8 <= res[2] <= v["max_r"] * 1.2):
            rejected.append((x, y, r))
        else:
            good.append(res)
    circles = merge_circles(good)
    S.rejected = rejected[:40]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    bg = background_ab(lab, circles)
    dets = []
    for x, y, r, sup in circles:
        feat = extract_features(lab, x, y, r, bg)
        if feat is None:
            continue
        d = Det(x, y, r, sup, feat)
        d.closest, d.dist, d.second = match(feat, S.samples)
        if d.closest and d.dist <= v["strict"]:
            d.coin = BY_LABEL[d.closest]
        dets.append(d)
    return dets


# State and mouse
class State:
    def __init__(self, size):
        self.vals = {s.key: s.default for s in SLIDERS}
        self.size = size
        self.samples = []
        self.auto_r = True
        self.freeze = False
        self.info = True
        self.frame = None
        self.dets = []
        self.rejected = []
        self.version = 0
        self.sig = None
        self.hits = []
        self.drag = None
        self.mouse = (0, 0)
        self.sel = None
        self.view = (0, 0, 1.0)
        self.toast = ("", 0.0)
        self.save_request = False
        self.load_samples()

    def notify(self, text, sec=2.8):
        self.toast = (text, time.time() + sec)

    def counts(self):
        out = {c.label: 0 for c in COINS}
        for s in self.samples:
            out[s.label] += 1
        return out

    def load_samples(self):
        try:
            with open(SAMPLES_FILE) as fh:
                data = json.load(fh)
            if tuple(data.get("size", ())) == tuple(self.size):
                self.samples = [Sample.from_json(d) for d in data["samples"]]
            else:
                self.notify("Saved samples ignored: different camera resolution")
        except (OSError, ValueError, KeyError):
            pass

    def save_samples(self):
        try:
            with open(SAMPLES_FILE, "w") as fh:
                json.dump({"size": list(self.size), "samples": [s.to_json() for s in self.samples]}, fh)
        except OSError:
            self.notify("Could not write " + SAMPLES_FILE)

    def selected(self):
        if self.sel is None:
            return None
        best, bd = None, 1e9
        for d in self.dets:
            dist = math.hypot(d.x - self.sel[0], d.y - self.sel[1])
            if dist < d.r and dist < bd:
                best, bd = d, dist
        return best

    def teach(self, idx):
        d = self.selected()
        if d is None:
            self.notify("Click a coin first, then press its key")
            return
        if d.support < max(0.7, self.vals["support"]):
            self.notify(f"Border not clear (support {d.support:.2f}): improve light or move the coin")
            return
        label = COINS[idx].label
        mine = [s for s in self.samples if s.label == label]
        if len(mine) >= MAX_SAMPLES_PER_COIN:
            self.samples.remove(mine[0])
        self.samples.append(Sample(label, d.feat))
        self.save_samples()
        self.version += 1
        self.notify(f"Learned {label}  ({self.counts()[label]} example(s))")

    def undo(self):
        if self.samples:
            s = self.samples.pop()
            self.save_samples()
            self.version += 1
            self.notify(f"Removed last example ({s.label})")

    def forget(self):
        self.samples.clear()
        self.save_samples()
        self.version += 1
        self.notify("All examples forgotten")

    def press(self, name):
        if name == "freeze":
            self.freeze = not self.freeze
        elif name == "save":
            self.save_request = True
        elif name == "forget":
            self.forget()
        elif name == "auto":
            self.auto_r = not self.auto_r

    def update_drag(self, x):
        sl, tx, tw = self.drag
        frac = min(max((x - tx) / tw, 0.0), 1.0)
        n = round((sl.lo + frac * (sl.hi - sl.lo) - sl.lo) / sl.step)
        v = min(max(sl.lo + n * sl.step, sl.lo), sl.hi)
        self.vals[sl.key] = round(v, 2) if sl.step < 1 else int(round(v))
        if sl.key in ("min_r", "max_r"):
            self.auto_r = False


def inside(pt, rect):
    x, y, w, h = rect
    return x <= pt[0] < x + w and y <= pt[1] < y + h


def on_mouse(event, x, y, flags, S):
    S.mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        for kind, data, rect in S.hits:
            if inside((x, y), rect):
                if kind == "button":
                    S.press(data)
                elif kind == "slider":
                    S.drag = data
                    S.update_drag(x)
                elif kind == "video":
                    ox, oy, s = S.view
                    S.sel = ((x - ox) / s, (y - oy) / s)
                return
    elif event == cv2.EVENT_MOUSEMOVE and S.drag:
        S.update_drag(x)
    elif event == cv2.EVENT_LBUTTONUP:
        S.drag = None


# Drawing tools
def put(img, s, x, y, scale=0.5, color=TEXT, thick=1, anchor="l", font=FONT):
    if anchor != "l":
        tw = cv2.getTextSize(s, font, scale, thick)[0][0]
        x = x - tw if anchor == "r" else x - tw // 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, AA)


def text_w(s, scale, thick=1, font=FONT):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


def outlined(img, s, x, y, scale, color, anchor="c"):
    put(img, s, x, y, scale, (0, 0, 0), 3, anchor)
    put(img, s, x, y, scale, color, 1, anchor)


def rrect(img, x, y, w, h, color, r=10):
    r = max(0, min(r, w // 2, h // 2))
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def card(img, x, y, w, h, title=None):
    rrect(img, x, y, w, h, BORDER, 12)
    rrect(img, x + 1, y + 1, w - 2, h - 2, CARD, 11)
    if title:
        put(img, title, x + 16, y + 24, 0.4, MUTED)


def fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, ox, oy, s


def fmt_money(v):
    return f"{v:.2f} DH"


# Panels
def draw_header(c, S, fps, ms, count, total):
    cv2.rectangle(c, (0, 0), (W, 64), HEADER, -1)
    cv2.rectangle(c, (0, 64), (W, 66), ACCENT, -1)
    cv2.circle(c, (40, 32), 15, ACCENT, 2, AA)
    cv2.circle(c, (40, 32), 8, ACCENT, 2, AA)
    put(c, "Coin Counter", 66, 31, 0.8, TEXT, 1, font=DUPLEX)
    put(c, "Moroccan dirhams & centimes  -  learns your coins, then counts them", 66, 51, 0.4, MUTED)
    x = W - 16
    for label, value, col in (("DETECTION", f"{ms:.0f} ms", TEXT), ("FPS", f"{fps:.0f}", TEXT),
                              ("TOTAL", fmt_money(total), ACCENT), ("COINS", str(count), TEXT)):
        tw = max(text_w(value, 0.55), text_w(label, 0.32)) + 24
        x -= tw
        rrect(c, x, 12, tw, 40, CARD, 8)
        put(c, label, x + 12, 26, 0.32, MUTED)
        put(c, value, x + 12, 45, 0.55, col)
        x -= 10


def draw_video(c, S, frame):
    x, y, w, h = VIDEO
    card(c, x, y, w, h)
    vw, vh = w - 8, h - 8
    img, ox, oy, s = fit(frame, vw, vh)
    S.view = (x + 4 + ox, y + 4 + oy, s)
    sel = S.selected()

    if S.info:
        for rx, ry, rr in S.rejected:
            cv2.circle(img, (int(rx * s + ox), int(ry * s + oy)), max(2, int(rr * s)), (60, 60, 150), 1, AA)
    for d in S.dets:
        cx, cy, r = int(d.x * s + ox), int(d.y * s + oy), int(d.r * s)
        col = KIND_COLORS[d.coin.kind] if d.coin else UNKNOWN_COLOR
        cv2.circle(img, (cx, cy), r, col, 2, AA)
        if d is sel:
            cv2.circle(img, (cx, cy), r + 5, WHITE, 2, AA)
        outlined(img, d.coin.label if d.coin else "?", cx, cy + 6, 0.6, WHITE)
        if S.info:
            outlined(img, f"{d.feat.diam:.0f} px", cx, cy + 24, 0.4, col)
        if d is sel:
            line1 = f"closest: {d.closest or '-'}  dist {d.dist:.1f}"
            line2 = f"border support {d.support:.2f}  (match if dist <= {S.vals['strict']:.1f})"
            outlined(img, line1, cx, cy + r + 20, 0.42, WHITE)
            outlined(img, line2, cx, cy + r + 38, 0.38, MUTED)

    n_taught = sum(1 for v in S.counts().values() if v)
    if not S.samples:
        msg, col = "TEACH ME: click a coin, then press its key (1-7)", WARN
    else:
        msg, col = f"Taught {n_taught}/7 coin types  -  click a coin + key to add examples", (110, 220, 110)
    tw = text_w(msg, 0.45) + 24
    rrect(img, 12, 12, tw, 26, (20, 20, 20), 8)
    put(img, msg, 24, 31, 0.45, col)
    if S.freeze:
        rrect(img, vw - 92, 12, 80, 26, (20, 20, 20), 8)
        put(img, "FROZEN", vw - 52, 31, 0.45, WARN, anchor="c")
    bar = 30
    img[vh - bar:] = (img[vh - bar:] * 0.3).astype(np.uint8)
    put(img, "Teach -> " + "   ".join(f"{i + 1}:{co.label}" for i, co in enumerate(COINS)), 12, vh - 10, 0.42, TEXT)
    if time.time() < S.toast[1]:
        tw = text_w(S.toast[0], 0.5) + 28
        rrect(img, (vw - tw) // 2, vh - 80, tw, 30, (40, 40, 40), 12)
        put(img, S.toast[0], vw // 2, vh - 59, 0.5, (110, 220, 110), anchor="c")
    c[y + 4:y + 4 + vh, x + 4:x + 4 + vw] = img
    S.hits.append(("video", None, (x + 4, y + 4, vw, vh)))


def draw_result(c, S):
    x, y, w, h = RESULT
    card(c, x, y, w, h, "RESULT")
    put(c, "examples", x + w - 150, y + 24, 0.34, MUTED, anchor="r")
    counts = {co.label: 0 for co in COINS}
    unknown = 0
    for d in S.dets:
        if d.coin:
            counts[d.coin.label] += 1
        else:
            unknown += 1
    taught = S.counts()
    total = sum(counts[co.label] * co.value for co in COINS)
    ry = y + 52
    for co in COINS:
        n = counts[co.label]
        col = KIND_COLORS[co.kind]
        cv2.circle(c, (x + 24, ry - 5), 6, col, -1 if n else 1, AA)
        put(c, co.label, x + 42, ry, 0.5, TEXT)
        t = taught[co.label]
        put(c, str(t) if t else "none", x + w - 150, ry, 0.4, MUTED if t else WARN, anchor="r")
        put(c, f"x {n}", x + w - 100, ry, 0.5, MUTED if n == 0 else TEXT, anchor="r")
        put(c, fmt_money(n * co.value) if n else "-", x + w - 18, ry, 0.5, MUTED if n == 0 else col, anchor="r")
        ry += 24
    put(c, "Unknown (?)", x + 42, ry, 0.5, UNKNOWN_COLOR if unknown else MUTED)
    put(c, f"x {unknown}", x + w - 100, ry, 0.5, UNKNOWN_COLOR if unknown else MUTED, anchor="r")
    cv2.line(c, (x + 16, ry + 12), (x + w - 16, ry + 12), BORDER, 1)
    put(c, "TOTAL", x + 18, ry + 42, 0.55, MUTED)
    put(c, fmt_money(total), x + w - 18, ry + 44, 0.9, ACCENT, 1, anchor="r", font=DUPLEX)
    return sum(counts.values()), total


def draw_settings(c, S):
    x, y, w, h = SETTINGS
    card(c, x, y, w, h, "DETECTION SETTINGS")
    ry = y + 38
    for sl in SLIDERS:
        v = S.vals[sl.key]
        auto = S.auto_r and S.samples and sl.key in ("min_r", "max_r")
        put(c, sl.label + ("  (auto)" if auto else ""), x + 16, ry + 10, 0.42, TEXT)
        put(c, f"{v:.2f}" if sl.step < 0.5 else (f"{v:.1f}" if sl.step < 1 else str(int(v))), x + w - 16, ry + 10, 0.45, ACCENT, anchor="r")
        tx, tw, ty = x + 25, w - 50, ry + 26
        frac = (v - sl.lo) / (sl.hi - sl.lo)
        rrect(c, tx, ty - 3, tw, 6, TRACK, 3)
        rrect(c, tx, ty - 3, max(6, int(tw * frac)), 6, ACCENT, 3)
        kx = tx + int(tw * frac)
        cv2.circle(c, (kx, ty), 9, WHITE, -1, AA)
        cv2.circle(c, (kx, ty), 4, ACCENT, -1, AA)
        S.hits.append(("slider", (sl, tx, tw), (x + 16, ry + 14, w - 32, 24)))
        ry += 36
    buttons = [("freeze", "Freeze", S.freeze), ("save", "Save", False),
               ("forget", "Forget all", False), ("auto", "Auto radius", S.auto_r)]
    bw = (w - 32 - 3 * 8) // 4
    for i, (name, label, active) in enumerate(buttons):
        rect = (x + 16 + i * (bw + 8), y + h - 40, bw, 28)
        fill = ACCENT if active else (HOVER if inside(S.mouse, rect) else CARD2)
        rrect(c, *rect, fill, 8)
        put(c, label, rect[0] + bw // 2, rect[1] + 19, 0.36, BG if active else TEXT, anchor="c")
        S.hits.append(("button", name, rect))


def render(S, frame, fps, ms):
    c = np.full((H, W, 3), BG, np.uint8)
    S.hits = []
    draw_video(c, S, frame)
    count, total = draw_result(c, S)
    draw_settings(c, S)
    draw_header(c, S, fps, ms, count, total)
    return c


# Main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="analyse a photo instead of the webcam")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap, frame = None, None
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Cannot read {args.image}")
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            raise SystemExit("Cannot open the webcam.")
        for _ in range(5):                   
            ok, frame = cap.read()
        if not ok:
            raise SystemExit("Cannot read from the webcam.")

    S = State((frame.shape[1], frame.shape[0]))
    S.frame = frame
    win = "Coin Counter"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse, S)

    fps, ms, t_prev = 0.0, 0.0, time.perf_counter()
    while True:
        if cap is not None and not S.freeze:
            ok, f = cap.read()
            if not ok:
                break
            S.frame = f
        sig = (id(S.frame), tuple(sorted(S.vals.items())), S.version, S.auto_r)
        if sig != S.sig:
            t0 = time.perf_counter()
            S.dets = analyse(S.frame, S)
            dt = (time.perf_counter() - t0) * 1000
            ms = 0.8 * ms + 0.2 * dt if ms else dt
            S.sig = (id(S.frame), tuple(sorted(S.vals.items())), S.version, S.auto_r)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        canvas = render(S, S.frame, fps, ms)
        if S.save_request:
            S.save_request = False
            name = f"coins_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(name, canvas)
            S.notify(f"Saved {name}")
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            S.press("freeze")
        elif key in (ord("s"), ord("S")):
            S.press("save")
        elif key in (ord("c"), ord("C")):
            S.forget()
        elif key in (ord("u"), ord("U")):
            S.undo()
        elif key in (ord("i"), ord("I")):
            S.info = not S.info
        elif key in (ord("a"), ord("A")):
            S.press("auto")
        elif ord("1") <= key <= ord("7"):
            S.teach(key - ord("1"))

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()