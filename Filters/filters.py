
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np


W, H = 1280, 720
WIN = "Vision Lab"
FONT = cv2.FONT_HERSHEY_SIMPLEX
DUPLEX = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

BG = (30, 26, 24)
HEADER = (38, 33, 30)
CARD = (46, 41, 38)
CARD2 = (62, 56, 52)
HOVER = (84, 76, 70)
BORDER = (72, 66, 62)
TRACK = (88, 80, 76)
TEXT = (238, 234, 232)
MUTED = (160, 152, 146)
WHITE = (255, 255, 255)

CAT_COLORS = {
    "BASE": (190, 185, 180),
    "SMOOTHING": (153, 211, 52),
    "EDGES": (60, 146, 251),
    "ENHANCE": (250, 139, 167),
}
CAT_TITLES = {
    "BASE": "BASE",
    "SMOOTHING": "SMOOTHING",
    "EDGES": "EDGES & DERIVATIVES",
    "ENHANCE": "ENHANCE & SEGMENT",
}

OUT_RECT = (252, 80, 700, 394)         
BOTTOM_X = (252, 489, 726)            
BOTTOM_Y, BOTTOM_W, BOTTOM_H = 486, 226, 218
PARAM_RECT = (968, 80, 296, 300)
THEORY_RECT = (968, 392, 296, 312)

# Parameter and filter definitions

@dataclass
class Param:
    key: str
    label: str
    lo: float = 0
    hi: float = 1
    default: float = 0
    step: float = 1
    hint: str = ""
    choices: Optional[tuple] = None             
    show_if: Optional[Callable] = None         


@dataclass
class Filter:
    name: str
    category: str
    desc: str
    formula: str
    tip: str
    run: Callable          
    kernel: Callable     
    params: list = field(default_factory=list)


def is_int(pm):
    return all(float(v).is_integer() for v in (pm.lo, pm.hi, pm.step))


def quantize(pm, raw):
    n = round((raw - pm.lo) / pm.step)
    v = min(max(pm.lo + n * pm.step, pm.lo), pm.hi)
    return int(round(v)) if is_int(pm) else round(v, 2)


def fmt_value(pm, v):
    if pm.choices:
        return pm.choices[int(v)]
    return str(int(v)) if is_int(pm) else f"{v:.1f}"



SOBEL_DIRS = ("dI/dx", "dI/dy", "Magnitude")
THRESH_METHODS = ("Binary", "Otsu", "Adaptive")


def run_original(frame, gray, p, info):
    return frame, "Original | no processing"

# Mean filter

def run_mean(frame, gray, p, info):
    k = p["k"]
    return cv2.blur(frame, (k, k)), f"Mean | k={k}"

# Gaussian filter

def run_gaussian(frame, gray, p, info):
    k, s = p["k"], p["sigma"]
    return cv2.GaussianBlur(frame, (k, k), s), f"Gaussian | k={k}, sigma={s:.1f}"

# Median filter

def run_median(frame, gray, p, info):
    k = p["k"]
    return cv2.medianBlur(frame, k), f"Median | k={k}"

# Bilateral filter

def run_bilateral(frame, gray, p, info):
    d, sc, ss = p["d"], p["sc"], p["ss"]
    out = cv2.bilateralFilter(frame, d, sc, ss)
    return out, f"Bilateral | d={d}, sigma_color={sc}, sigma_space={ss}"

# Sobel filter

def run_sobel(frame, gray, p, info):
    k, gain, mode = p["k"], p["gain"], p["dir"]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    if mode == 0:
        g = np.abs(gx)
    elif mode == 1:
        g = np.abs(gy)
    else:
        g = cv2.magnitude(gx, gy)
    out = cv2.convertScaleAbs(g, alpha=gain)
    return out, f"Sobel | {SOBEL_DIRS[mode]}, k={k}, gain={gain:.1f}"

# Laplacian filter

def run_laplacian(frame, gray, p, info):
    k, gain, pre = p["k"], p["gain"], p["pre"]
    src = cv2.GaussianBlur(gray, (0, 0), pre) if pre > 0 else gray
    lap = cv2.Laplacian(src, cv2.CV_32F, ksize=k)
    out = cv2.convertScaleAbs(lap, alpha=gain)
    name = "LoG" if pre > 0 else "Laplacian"
    return out, f"{name} | k={k}, gain={gain:.1f}, pre-blur={pre:.1f}"

# Canny edge detector

def run_canny(frame, gray, p, info):
    lo = int(p["low"])
    hi = int(max(lo + 1, p["high"]))
    out = cv2.Canny(gray, lo, hi, apertureSize=p["ap"], L2gradient=bool(p["l2"]))
    return out, f"Canny | low={lo}, high={hi}, aperture={p['ap']}, L{p['l2'] + 1}"

# Unsharp mask

def run_unsharp(frame, gray, p, info):
    s, a = p["sigma"], p["amount"]
    blur = cv2.GaussianBlur(frame, (0, 0), s)
    return cv2.addWeighted(frame, 1 + a, blur, -a, 0), f"Unsharp mask | sigma={s:.1f}, amount={a:.1f}"

# Thresholding

def run_threshold(frame, gray, p, info):
    m = p["method"]
    if m == 0:
        _, out = cv2.threshold(gray, p["t"], 255, cv2.THRESH_BINARY)
        cap = f"Threshold | binary, T={p['t']}"
    elif m == 1:
        t, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        info["otsu"] = t
        cap = f"Threshold | Otsu, T={t:.0f} (found automatically)"
    else:
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, p["block"], p["c"])
        cap = f"Threshold | adaptive, block={p['block']}, C={p['c']}"
    return out, cap

# Kernel generation functions

def kern_identity(p):
    k = np.zeros((3, 3), np.float32)
    k[1, 1] = 1
    return k, "Identity kernel: output = input"


def kern_mean(p):
    k = p["k"]
    return np.full((k, k), 1.0 / (k * k), np.float32), f"{k}x{k} box, every weight = 1/{k * k}"


def kern_gauss(p):
    g = cv2.getGaussianKernel(p["k"], p["sigma"])
    return (g @ g.T).astype(np.float32), f"{p['k']}x{p['k']} Gaussian, sigma={p['sigma']:.1f}"


def kern_sobel(p):
    kx, ky = cv2.getDerivKernels(1, 0, p["k"], normalize=False)
    gx = np.outer(ky.ravel(), kx.ravel()).astype(np.float32)
    if p["dir"] == 1:
        return gx.T.copy(), "Gy kernel (transpose of Gx)"
    return gx, "Gx kernel" + (" (Gy = transpose)" if p["dir"] == 2 else "")


def kern_laplacian(p):
    k = p["k"]
    if k == 1:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32), "4-neighbour Laplacian"
    a, b = cv2.getDerivKernels(2, 0, k, normalize=False)
    c, d = cv2.getDerivKernels(0, 2, k, normalize=False)
    m = np.outer(b.ravel(), a.ravel()) + np.outer(d.ravel(), c.ravel())
    return m.astype(np.float32), f"{k}x{k} Laplacian kernel"


def kern_none(text):
    return lambda p: (None, text)


def ksize_param(lo=3, hi=31, default=5, hint="Odd window size. Larger = blurrier"):
    return Param("k", "Kernel size (k)", lo, hi, default, 2, hint)

# Filter definitions

FILTERS = [
    Filter(
        "Original", "BASE",
        "The raw camera frame, untouched. Use it as the reference to judge what every other filter adds or removes.",
        "out(x,y) = I(x,y)",
        "Turn on Split view, then switch filters to compare them with the original.",
        run_original, kern_identity),
    Filter(
        "Mean", "SMOOTHING",
        "Replaces each pixel by the average of its k x k neighbourhood. Every neighbour counts equally, so noise AND edges are blurred.",
        "out(x,y) = 1/k^2 * sum I(x+i, y+j) over the window",
        "Raise k and watch details melt. Then compare with Gaussian at the same k.",
        run_mean, kern_mean, [ksize_param()]),
    Filter(
        "Gaussian", "SMOOTHING",
        "Weighted average: pixels near the centre weigh more, following a bell curve. Smoother and more natural than the box filter.",
        "G(x,y) = exp(-(x^2 + y^2) / (2 sigma^2)), normalised",
        "Keep k fixed and lower sigma: the filter slowly turns into the identity.",
        run_gaussian, kern_gauss,
        [ksize_param(default=7),
         Param("sigma", "Sigma", 0.1, 10, 1.5, 0.1, "Width of the bell. Larger = more blur")]),
    Filter(
        "Median", "SMOOTHING",
        "Replaces each pixel by the median of its neighbourhood. Extreme values are ignored, so edges stay sharp while outliers vanish.",
        "out(x,y) = median{ I(x+i, y+j) }",
        "Ideal against salt-and-pepper noise. Try k=3 then k=15 and compare with Mean.",
        run_median, kern_none("Non-linear: the k x k window is sorted and the middle value wins."),
        [ksize_param(default=5, hint="Odd window size. Cost grows with k")]),
    Filter(
        "Bilateral", "SMOOTHING",
        "Gaussian weights in space AND in intensity: only neighbours with a similar colour contribute. Flat areas get smooth, edges survive.",
        "w(p,q) = G_space(|p-q|) * G_range(|I(p)-I(q)|)",
        "Push sigma colour very high and it degrades into a plain Gaussian.",
        run_bilateral, kern_none("Non-linear: weights depend on the pixel values, so there is no fixed kernel."),
        [Param("d", "Diameter (d)", 1, 15, 9, 1, "Size of the pixel neighbourhood"),
         Param("sc", "Sigma colour", 1, 200, 75, 1, "Tolerance to colour difference"),
         Param("ss", "Sigma space", 1, 200, 75, 1, "How far neighbours can influence")]),
    Filter(
        "Sobel", "EDGES",
        "Approximates the first derivative of intensity. A large response means the image changes quickly there: an edge.",
        "G = sqrt(Gx^2 + Gy^2),  angle = atan2(Gy, Gx)",
        "dI/dx reveals vertical edges, dI/dy horizontal ones. Magnitude shows both.",
        run_sobel, kern_sobel,
        [Param("dir", "Direction", 0, 2, 2, 1, "Which derivative to compute", SOBEL_DIRS),
         Param("k", "Kernel size (k)", 3, 7, 3, 2, "Bigger = smoother, thicker edges"),
         Param("gain", "Gain", 0.1, 5, 0.5, 0.1, "Brightness multiplier of the result")]),
    Filter(
        "Laplacian", "EDGES",
        "Second derivative: d2I/dx2 + d2I/dy2. It reacts to edges in every direction but is very sensitive to noise.",
        "L(x,y) = d2I/dx2 + d2I/dy2",
        "Raise Pre-blur above 0 to get the classic Laplacian of Gaussian (LoG) and see the noise vanish.",
        run_laplacian, kern_laplacian,
        [Param("k", "Kernel size (k)", 1, 7, 1, 2, "1 = classic 4-neighbour kernel"),
         Param("gain", "Gain", 0.1, 5, 1, 0.1, "Brightness multiplier of the result"),
         Param("pre", "Pre-blur sigma", 0, 5, 0, 0.5, "0 = off. Gaussian blur applied first")]),
    Filter(
        "Canny", "EDGES",
        "Multi-stage detector: Sobel gradient, thinning of ridges, then hysteresis with two thresholds to keep connected, strong edges.",
        "keep if G > high, or G > low and linked to a strong edge",
        "Raise both thresholds to drop noise. A good rule: high = 2 to 3 x low.",
        run_canny, kern_none("Pipeline: Sobel gradient > thin the edges > hysteresis thresholds."),
        [Param("low", "Low threshold", 0, 255, 50, 1, "Weak edges kept only if linked"),
         Param("high", "High threshold", 0, 255, 150, 1, "Pixels above are sure edges"),
         Param("ap", "Aperture (Sobel k)", 3, 7, 3, 2, "Kernel size of the internal Sobel"),
         Param("l2", "Gradient norm", 0, 1, 0, 1, "L1 is faster, L2 is exact", ("L1", "L2"))]),
    Filter(
        "Unsharp", "ENHANCE",
        "Sharpens by adding back the detail a blur removes: out = I + a * (I - blurred). Boosts fine texture and local contrast.",
        "out = (1 + a) * I - a * Gaussian_sigma(I)",
        "A large amount creates halos around edges and amplifies noise.",
        run_unsharp, kern_none("One linear kernel: (1+a) x identity - a x Gaussian."),
        [Param("sigma", "Blur sigma", 0.5, 10, 2, 0.1, "Size of the details that get boosted"),
         Param("amount", "Amount (a)", 0, 5, 1.5, 0.1, "Sharpening strength")]),
    Filter(
        "Threshold", "ENHANCE",
        "Turns the image into pure black and white. Otsu finds the best global level automatically; adaptive decides locally.",
        "out = 255 if I > T else 0",
        "Try Adaptive under uneven lighting: a single global T fails there.",
        run_threshold, kern_none("Adaptive: a different T for every pixel, from its local mean."),
        [Param("method", "Method", 0, 2, 1, 1, "How the threshold T is chosen", THRESH_METHODS),
         Param("t", "Threshold (T)", 0, 255, 127, 1, "Global gray-level cut",
               show_if=lambda v: v["method"] == 0),
         Param("block", "Block size", 3, 51, 11, 2, "Local neighbourhood for adaptive T",
               show_if=lambda v: v["method"] == 2),
         Param("c", "Constant (C)", -10, 30, 2, 1, "Subtracted from the local mean",
               show_if=lambda v: v["method"] == 2)]),
]

# State management

class State:
    def __init__(self):
        self.idx = 0
        self.values = {f.name: {p.key: p.default for p in f.params} for f in FILTERS}
        self.hits = []
        self.drag = None
        self.mouse = (0, 0)
        self.freeze = False
        self.mirror = True
        self.split = False
        self.split_frac = 0.5
        self.frame = None
        self.info = {}
        self.save_request = False
        self.toast_text = ""
        self.toast_until = 0.0

    def current(self):
        return FILTERS[self.idx]

    def press(self, name):
        if name == "split":
            self.split = not self.split
        elif name == "freeze":
            self.freeze = not self.freeze
        elif name == "mirror":
            self.mirror = not self.mirror
            if self.frame is not None:
                self.frame = cv2.flip(self.frame, 1)
        elif name == "reset":
            f = self.current()
            self.values[f.name] = {p.key: p.default for p in f.params}
        elif name == "save":
            self.save_request = True

    def notify(self, text, seconds=2.0):
        self.toast_text, self.toast_until = text, time.time() + seconds

    def update_drag(self, x):
        kind, data = self.drag
        if kind == "slider":
            pm, tx, tw = data
            frac = min(max((x - tx) / tw, 0.0), 1.0)
            self.values[self.current().name][pm.key] = quantize(pm, pm.lo + frac * (pm.hi - pm.lo))
        else:
            ix, iw = data
            self.split_frac = min(max((x - ix) / iw, 0.02), 0.98)

# Mouse and drawing utilities

def inside(pt, rect):
    x, y, w, h = rect
    return x <= pt[0] < x + w and y <= pt[1] < y + h


def on_mouse(event, x, y, flags, S):
    S.mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        for kind, data, rect in S.hits:
            if not inside((x, y), rect):
                continue
            if kind == "filter":
                S.idx = data
            elif kind == "choice":
                pm, i = data
                S.values[S.current().name][pm.key] = i
            elif kind == "button":
                S.press(data)
            else:  # slider / split
                S.drag = (kind, data)
                S.update_drag(x)
            break
    elif event == cv2.EVENT_MOUSEMOVE and S.drag:
        S.update_drag(x)
    elif event == cv2.EVENT_LBUTTONUP:
        S.drag = None

def put(img, s, x, y, scale=0.5, color=TEXT, thick=1, anchor="l", font=FONT):
    if anchor != "l":
        tw = cv2.getTextSize(s, font, scale, thick)[0][0]
        x = x - tw if anchor == "r" else x - tw // 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, AA)


def text_w(s, scale, thick=1, font=FONT):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


def wrap(text, scale, max_w):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if text_w(t, scale) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


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
        put(img, title, x + 14, y + 22, 0.38, MUTED)


def fit(img, w, h, bg=(0, 0, 0)):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    out = np.full((h, w, 3), bg, np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out


def tag(img, text, x, y, color):
    tw = text_w(text, 0.4)
    rrect(img, x, y, tw + 18, 22, (20, 20, 20), 8)
    put(img, text, x + 9, y + 16, 0.4, color)


def heatmap(K, size):
    k = K.astype(np.float32)
    m = float(np.abs(k).max()) or 1.0
    norm = (k / m) * 0.5 + 0.5 if k.min() < 0 else k / m
    img = cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cell = max(1, size // K.shape[0])
    return cv2.resize(img, (cell * K.shape[1], cell * K.shape[0]), interpolation=cv2.INTER_NEAREST), cell


def draw_header(c, S, frame, fps, ms):
    col = CAT_COLORS[S.current().category]
    cv2.rectangle(c, (0, 0), (W, 64), HEADER, -1)
    cv2.rectangle(c, (0, 64), (W, 66), col, -1)
    cv2.circle(c, (40, 32), 15, col, 2, AA)
    cv2.circle(c, (40, 32), 6, col, -1, AA)
    put(c, "Vision Lab", 66, 31, 0.8, TEXT, 1, font=DUPLEX)
    put(c, "Spatial filtering playground  -  pick a filter, tune its hyperparameters", 66, 51, 0.4, MUTED)

    x = W - 16
    h, w = frame.shape[:2]
    for label, value, vcol in (("INPUT", f"{w}x{h}", TEXT), ("FILTER TIME", f"{ms:.1f} ms", col),
                               ("FPS", f"{fps:.0f}", TEXT)):
        tw = max(text_w(value, 0.55), text_w(label, 0.32)) + 24
        x -= tw
        rrect(c, x, 12, tw, 40, CARD, 8)
        put(c, label, x + 12, 26, 0.32, MUTED)
        put(c, value, x + 12, 45, 0.55, vcol)
        x -= 10
    status, scol = ("FROZEN", (80, 180, 255)) if S.freeze else ("LIVE", (110, 220, 110))
    tw = text_w(status, 0.45) + 36
    x -= tw
    rrect(c, x, 12, tw, 40, CARD, 8)
    cv2.circle(c, (x + 14, 32), 5, scol, -1, AA)
    put(c, status, x + 26, 37, 0.45, scol)


def draw_sidebar(c, S):
    x, y, w, h = 16, 80, 220, 624
    card(c, x, y, w, h)
    cy, last = y + 14, None
    for i, f in enumerate(FILTERS):
        if f.category != last:
            last = f.category
            put(c, CAT_TITLES[f.category], x + 14, cy + 10, 0.36, MUTED)
            cy += 22
        rect = (x + 10, cy, w - 20, 32)
        sel, col = i == S.idx, CAT_COLORS[f.category]
        fill = col if sel else (HOVER if inside(S.mouse, rect) else CARD2)
        rrect(c, *rect, fill, 8)
        cv2.circle(c, (x + 26, cy + 16), 4, BG if sel else col, -1, AA)
        put(c, f.name, x + 40, cy + 21, 0.5, BG if sel else TEXT)
        put(c, str((i + 1) % 10), x + w - 20, cy + 21, 0.4, BG if sel else MUTED, anchor="r")
        S.hits.append(("filter", i, rect))
        cy += 36

    put(c, "CONTROLS", x + 14, cy + 10, 0.36, MUTED)
    cy += 22
    accent = CAT_COLORS[S.current().category]
    buttons = [("split", "Split view", "V", S.split), ("freeze", "Freeze", "SPC", S.freeze),
               ("mirror", "Mirror", "M", S.mirror), ("reset", "Reset", "R", False),
               ("save", "Save PNG", "S", False)]
    for i, (name, label, key, active) in enumerate(buttons):
        if i < 4:
            rect = (x + 10 + (i % 2) * 105, cy + (i // 2) * 36, 95, 30)
        else:
            rect = (x + 10, cy + 72, 200, 30)
        fill = accent if active else (HOVER if inside(S.mouse, rect) else CARD2)
        rrect(c, *rect, fill, 8)
        put(c, label, rect[0] + 10, rect[1] + 20, 0.42, BG if active else TEXT)
        put(c, key, rect[0] + rect[2] - 8, rect[1] + 20, 0.34, BG if active else MUTED, anchor="r")
        S.hits.append(("button", name, rect))
    put(c, "Q / ESC to quit", x + w // 2, y + h - 10, 0.34, MUTED, anchor="c")


def draw_output(c, S, frame, result, caption):
    x, y, w, h = OUT_RECT
    card(c, x, y, w, h)
    ix, iy, iw, ih = x + 4, y + 4, w - 8, h - 8
    col = CAT_COLORS[S.current().category]
    res = fit(result, iw, ih)
    if S.split:
        orig = fit(frame, iw, ih)
        sx = int(iw * S.split_frac)
        res[:, :sx] = orig[:, :sx]
        cv2.line(res, (sx, 0), (sx, ih), WHITE, 2, AA)
        cv2.circle(res, (sx, ih // 2), 15, WHITE, -1, AA)
        put(res, "<>", sx, ih // 2 + 5, 0.5, BG, 2, anchor="c")
        tag(res, "ORIGINAL", 12, 12, TEXT)
        tag(res, "FILTERED", iw - 12 - text_w("FILTERED", 0.4) - 18, 12, col)
        S.hits.append(("split", (ix, iw), (ix, iy, iw, ih)))
    else:
        tag(res, S.current().name.upper(), 12, 12, col)
    bar = 34
    res[ih - bar:] = (res[ih - bar:] * 0.3).astype(np.uint8)
    put(res, caption, 14, ih - 12, 0.5, TEXT)
    if time.time() < S.toast_until:
        tw = text_w(S.toast_text, 0.5) + 28
        rrect(res, (iw - tw) // 2, 14, tw, 30, (40, 40, 40), 12)
        put(res, S.toast_text, iw // 2, 34, 0.5, (110, 220, 110), anchor="c")
    c[iy:iy + ih, ix:ix + iw] = res


def draw_kernel_card(c, S, x, y, f, p, col):
    card(c, x, y, BOTTOM_W, BOTTOM_H, "KERNEL / NEIGHBOURHOOD")
    K, cap = f.kernel(p)
    ax, ay, side = x + 38, y + 34, 150
    if f.name == "Threshold" and p["method"] != 2:
        t = p["t"] if p["method"] == 0 else S.info.get("otsu", 127)
        cv2.rectangle(c, (ax, ay), (ax + side, ay + side), BG, -1)
        tx = int(ax + t / 255 * side)
        pts = np.array([(ax, ay + side - 4), (tx, ay + side - 4), (tx, ay + 4), (ax + side, ay + 4)], np.int32)
        cv2.polylines(c, [pts], False, col, 2, AA)
        put(c, "in", ax + side - 4, ay + side - 8, 0.34, MUTED, anchor="r")
        put(c, "out", ax + 6, ay + 18, 0.34, MUTED)
        cap = f"Transfer curve, T = {t:.0f}"
    elif K is not None:
        hm, cell = heatmap(K, side)
        hx, hy = x + (BOTTOM_W - hm.shape[1]) // 2, ay + (side - hm.shape[0]) // 2
        c[hy:hy + hm.shape[0], hx:hx + hm.shape[1]] = hm
        n = K.shape[0]
        if n <= 5 and np.allclose(K, np.round(K)):
            for r in range(n):
                for q in range(n):
                    px, py = hx + q * cell + cell // 2, hy + r * cell + cell // 2
                    bright = int(hm[r * cell + cell // 2, q * cell + cell // 2].mean()) > 128
                    put(c, str(int(K[r, q])), px, py + 5, 0.42, (20, 20, 20) if bright else WHITE, anchor="c")
    else:
        for i, line in enumerate(wrap(cap, 0.42, BOTTOM_W - 30)):
            put(c, line, x + 15, ay + 28 + i * 20, 0.42, TEXT)
        return
    for i, line in enumerate(wrap(cap, 0.36, BOTTOM_W - 24)[:2]):
        put(c, line, x + BOTTOM_W // 2, y + 200 + i * 13, 0.36, MUTED, anchor="c")


def draw_bottom(c, S, frame, result, f, p, col):
    # 1) input thumbnail
    x, y = BOTTOM_X[0], BOTTOM_Y
    card(c, x, y, BOTTOM_W, BOTTOM_H, "INPUT (ORIGINAL)")
    c[y + 34:y + 34 + 172, x + 12:x + 12 + 202] = fit(frame, 202, 172, BG)

    # 2) histogram of the output
    x = BOTTOM_X[1]
    card(c, x, y, BOTTOM_W, BOTTOM_H, "OUTPUT HISTOGRAM")
    g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g], [0], None, [64], [0, 256]).ravel()
    hist = hist / (hist.max() + 1e-6)
    ax, ay, aw, ah = x + 12, y + 34, 202, 140
    cv2.rectangle(c, (ax, ay), (ax + aw, ay + ah), BG, -1)
    for i, v in enumerate(hist):
        bh = int(v * (ah - 10))
        if bh > 0:
            cv2.rectangle(c, (ax + 5 + i * 3, ay + ah - bh), (ax + 7 + i * 3, ay + ah), col, -1)
    put(c, "0", ax, ay + ah + 13, 0.3, MUTED)
    put(c, "255", ax + aw, ay + ah + 13, 0.3, MUTED, anchor="r")
    mean, std = cv2.meanStdDev(g)
    put(c, f"mean {mean[0][0]:.0f}    std {std[0][0]:.0f}", x + BOTTOM_W // 2, y + 207, 0.42, TEXT, anchor="c")

    # 3) kernel
    draw_kernel_card(c, S, BOTTOM_X[2], y, f, p, col)


def draw_slider(c, S, pm, v, x, y, w, col):
    put(c, pm.label, x, y + 12, 0.48, TEXT)
    put(c, fmt_value(pm, v), x + w, y + 12, 0.5, col, anchor="r")
    tx, tw, ty = x + 9, w - 18, y + 30
    frac = (v - pm.lo) / (pm.hi - pm.lo)
    rrect(c, tx, ty - 3, tw, 6, TRACK, 3)
    rrect(c, tx, ty - 3, max(6, int(tw * frac)), 6, col, 3)
    kx = tx + int(tw * frac)
    cv2.circle(c, (kx, ty), 9, WHITE, -1, AA)
    cv2.circle(c, (kx, ty), 4, col, -1, AA)
    put(c, pm.hint, x, y + 54, 0.36, MUTED)
    S.hits.append(("slider", (pm, tx, tw), (x, y + 14, w, 32)))


def draw_choice(c, S, pm, v, x, y, w, col):
    put(c, pm.label, x, y + 12, 0.48, TEXT)
    n, gap = len(pm.choices), 6
    bw = (w - gap * (n - 1)) // n
    for i, name in enumerate(pm.choices):
        rect = (x + i * (bw + gap), y + 20, bw, 26)
        sel = i == int(v)
        fill = col if sel else (HOVER if inside(S.mouse, rect) else CARD2)
        rrect(c, *rect, fill, 13)
        put(c, name, rect[0] + bw // 2, rect[1] + 18, 0.4, BG if sel else TEXT, anchor="c")
        S.hits.append(("choice", (pm, i), rect))
    put(c, pm.hint, x, y + 60, 0.36, MUTED)


def draw_params(c, S, f, col):
    x, y, w, h = PARAM_RECT
    card(c, x, y, w, h)
    put(c, "HYPERPARAMETERS", x + 16, y + 26, 0.4, MUTED)
    tw = text_w(f.name, 0.45) + 22
    rrect(c, x + w - 16 - tw, y + 10, tw, 24, col, 12)
    put(c, f.name, x + w - 16 - tw + 11, y + 27, 0.45, BG)

    vals = S.values[f.name]
    visible = [pm for pm in f.params if pm.show_if is None or pm.show_if(vals)]
    if not visible:
        put(c, "This filter has no hyperparameters.", x + w // 2, y + 150, 0.45, MUTED, anchor="c")
        put(c, "Pick another one in the sidebar.", x + w // 2, y + 172, 0.4, MUTED, anchor="c")
        return
    ry = y + 48
    for pm in visible:
        if pm.choices:
            draw_choice(c, S, pm, vals[pm.key], x + 16, ry, w - 32, col)
        else:
            draw_slider(c, S, pm, vals[pm.key], x + 16, ry, w - 32, col)
        ry += 60


def draw_theory(c, f, col):
    x, y, w, h = THEORY_RECT
    card(c, x, y, w, h)
    inner = w - 32
    put(c, "HOW IT WORKS", x + 16, y + 26, 0.4, MUTED)
    cy = y + 50
    for line in wrap(f.desc, 0.44, inner):
        put(c, line, x + 16, cy, 0.44, TEXT)
        cy += 18
    cy += 8
    put(c, "FORMULA", x + 16, cy, 0.36, MUTED)
    lines = wrap(f.formula, 0.4, inner - 16)
    box_h = len(lines) * 16 + 12
    rrect(c, x + 16, cy + 8, inner, box_h, BG, 8)
    for i, line in enumerate(lines):
        put(c, line, x + 24, cy + 25 + i * 16, 0.4, col)
    cy += box_h + 30
    put(c, "TRY THIS", x + 16, cy, 0.36, MUTED)
    cy += 20
    for line in wrap(f.tip, 0.42, inner):
        put(c, line, x + 16, cy, 0.42, TEXT)
        cy += 17


def render(S, frame, result, caption, fps, ms):
    c = np.full((H, W, 3), BG, np.uint8)
    S.hits = []
    f = S.current()
    p = S.values[f.name]
    col = CAT_COLORS[f.category]
    draw_header(c, S, frame, fps, ms)
    draw_sidebar(c, S)
    draw_output(c, S, frame, result, caption)
    draw_bottom(c, S, frame, result, f, p, col)
    draw_params(c, S, f, col)
    draw_theory(c, f, col)
    return c

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the webcam.")

    S = State()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse, S)

    fps, ms, t_prev = 0.0, 0.0, time.perf_counter()
    while True:
        if not S.freeze or S.frame is None:
            ok, raw = cap.read()
            if not ok:
                print("Cannot read a frame from the webcam.")
                break
            S.frame = cv2.flip(raw, 1) if S.mirror else raw
        frame = S.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f = S.current()
        t0 = time.perf_counter()
        result, caption = f.run(frame, gray, S.values[f.name], S.info)
        ms = 0.9 * ms + 0.1 * (time.perf_counter() - t0) * 1000
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if S.save_request:
            S.save_request = False
            name = f"capture_{f.name.lower()}_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(name, result)
            S.notify(f"Saved {name}")

        cv2.imshow(WIN, render(S, frame, result, caption, fps, ms))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            S.press("freeze")
        elif key in (ord("v"), ord("V")):
            S.press("split")
        elif key in (ord("m"), ord("M")):
            S.press("mirror")
        elif key in (ord("r"), ord("R")):
            S.press("reset")
        elif key in (ord("s"), ord("S")):
            S.press("save")
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(FILTERS):
            S.idx = key - ord("1")
        elif key == ord("0") and len(FILTERS) >= 10:
            S.idx = 9

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()