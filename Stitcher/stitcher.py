import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_IMAGES = 5
CANVAS_MAX_SIDE = 7000
CANVAS_MAX_PIXELS = 36e6
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# Parameters 
@dataclass
class Params:
    engine: int = 0        # 0 custom, 1 OpenCV panorama, 2 OpenCV scans
    model: int = 0         # 0 auto, 1 similarity, 2 homography
    nfeat: int = 4000
    ratio: float = 0.75
    min_inl: int = 15
    work: int = 1600
    gain: bool = True
    crop: bool = True


MODEL_NAMES = ("auto", "similarity", "homography")
ENGINE_NAMES = ("Custom", "CV panorama", "CV scans")


@dataclass
class Result:
    ok: bool = False
    image: np.ndarray = None
    layout: np.ndarray = None
    method: str = ""
    seconds: float = 0.0
    messages: list = field(default_factory=list)
    pairs: list = field(default_factory=list)       # (i, j, inliers, model)
    matched: list = field(default_factory=list)     # per image: True/False/None
    angles: list = field(default_factory=list)      # per image rotation (deg) or None


# Image helpers
def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def valid_mask(img):
    nz = (img.max(axis=2) >= 4).astype(np.uint8) * 255
    if 1.0 - nz.mean() / 255.0 < 0.02:
        return np.full(img.shape[:2], 255, np.uint8)
    nz = cv2.morphologyEx(nz, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nz = cv2.morphologyEx(nz, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nz)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        nz = ((lab == k) * 255).astype(np.uint8)
    return cv2.erode(nz, np.ones((5, 5), np.uint8))


def crop_black(img, thr=3):
    mask = (img.max(axis=2) > thr).astype(np.uint8)
    pts = cv2.findNonZero(mask)
    if pts is None:
        return img
    x, y, w, h = cv2.boundingRect(pts)
    return img[y:y + h, x:x + w]

# Features, matching, geometric models
def detect(img, mask, nfeat):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=int(nfeat))
    kps, des = sift.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    return pts, des


def plausible(H, w, h, lo=0.12, hi=8.0):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        t = cv2.perspectiveTransform(c, H.astype(np.float64))
    except cv2.error:
        return False
    if not np.all(np.isfinite(t)):
        return False
    if not cv2.isContourConvex(t.astype(np.float32)):
        return False
    ratio = cv2.contourArea(t.astype(np.float32)) / float(w * h)
    return lo <= ratio <= hi


def match_pair(fa, fb, shape_a, p):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < p.ratio * pair[1].distance]
    if len(good) < p.min_inl:
        return None
    need = max(p.min_inl, int(8 + 0.2 * len(good)))
    src = np.float32([pa[m.queryIdx] for m in good])
    dst = np.float32([pb[m.trainIdx] for m in good])
    h, w = shape_a[:2]
    cands = []
    model = MODEL_NAMES[p.model]
    if model in ("auto", "similarity"):
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                             maxIters=5000, confidence=0.995)
        if M is not None and inl is not None:
            H = np.vstack([M, [0, 0, 1]])
            if inl.sum() >= need and plausible(H, w, h):
                cands.append(("similarity", H, int(inl.sum())))
    if model in ("auto", "homography"):
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995)
        if H is not None and inl is not None and inl.sum() >= need and plausible(H, w, h):
            cands.append(("homography", H, int(inl.sum())))
    if not cands:
        return None
    best = max(cands, key=lambda c: c[2])
    if model == "auto" and len(cands) == 2:
        sim = [c for c in cands if c[0] == "similarity"][0]
        hom = [c for c in cands if c[0] == "homography"][0]
        best = sim if sim[2] >= 0.9 * hom[2] else hom        # prefer the simpler model when it fits as well
    return dict(model=best[0], H=best[1], inliers=best[2], matches=len(good))


# Custom engine
def stitch_custom(imgs, p):
    t0 = time.perf_counter()
    res = Result(method="Custom (SIFT + RANSAC + feather blend)")
    n = len(imgs)
    res.matched = [False] * n
    res.angles = [None] * n
    masks = [valid_mask(im) for im in imgs]
    feats = [detect(im, m, p.nfeat) for im, m in zip(imgs, masks)]

    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            r = match_pair(feats[i], feats[j], imgs[i].shape, p)
            if r:
                edges[(i, j)] = r
                res.pairs.append((i, j, r["inliers"], r["model"]))
    if not edges:
        res.messages.append("No pair of images has enough matching points. Do they overlap (>= 25 %)?")
        res.seconds = time.perf_counter() - t0
        return res

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in edges:
        parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    def comp_score(nodes_):
        return (len(nodes_), sum(r["inliers"] for (a, b), r in edges.items() if a in nodes_ and b in nodes_))

    nodes = max(comps.values(), key=comp_score)
    if len(nodes) < 2:
        res.messages.append("No pair of images could be linked.")
        res.seconds = time.perf_counter() - t0
        return res
    for i in range(n):
        if i not in nodes:
            res.messages.append(f"Image {i + 1} does not match the others (no overlap or too few details).")

    # maximum spanning tree (Kruskal)
    uf = {i: i for i in nodes}

    def ufind(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        return a

    tree = []
    for (i, j), r in sorted(edges.items(), key=lambda kv: -kv[1]["inliers"]):
        if i in nodes and j in nodes and ufind(i) != ufind(j):
            uf[ufind(i)] = ufind(j)
            tree.append((i, j))
    adj = {i: [] for i in nodes}
    for i, j in tree:
        adj[i].append(j)
        adj[j].append(i)
    weight = {i: sum(edges[(min(i, k), max(i, k))]["inliers"] for k in adj[i]) for i in nodes}
    ref = max(nodes, key=lambda i: weight[i])

    def H_map(a, b):                      # maps image a -> image b
        if (a, b) in edges:
            return edges[(a, b)]["H"]
        return np.linalg.inv(edges[(b, a)]["H"])

    Href = {ref: np.eye(3)}
    order, par = [ref], {ref: None}
    k = 0
    while k < len(order):
        a = order[k]
        k += 1
        for b in adj[a]:
            if b not in Href:
                Href[b] = Href[a] @ H_map(b, a)
                par[b] = a
                order.append(b)
    for i in list(order):
        h, w = imgs[i].shape[:2]
        if not plausible(Href[i], w, h, 0.03, 30.0):
            res.messages.append(f"Image {i + 1}: unstable transform, ignored.")
            order.remove(i)
    if len(order) < 2:
        res.messages.append("Not enough consistent images to build a result.")
        res.seconds = time.perf_counter() - t0
        return res

    # orient the whole result like the first loaded image (snapped to 90 degrees)
    anchor = min(order)
    k90 = int(round(math.degrees(math.atan2(Href[anchor][1, 0], Href[anchor][0, 0])) / 90.0))
    th = math.radians(-90.0 * k90)
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], np.float64)
    for i in order:
        Href[i] = R @ Href[i]
        res.matched[i] = True
        res.angles[i] = math.degrees(math.atan2(Href[i][1, 0], Href[i][0, 0]))

    # canvas
    allc = []
    for i in order:
        h, w = imgs[i].shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        allc.append(cv2.perspectiveTransform(c, Href[i]).reshape(-1, 2))
    allc = np.vstack(allc)
    xmin, ymin = np.floor(allc.min(0))
    xmax, ymax = np.ceil(allc.max(0))
    W0, H0 = xmax - xmin, ymax - ymin
    s = min(1.0, CANVAS_MAX_SIDE / max(W0, H0), math.sqrt(CANVAS_MAX_PIXELS / (W0 * H0)))
    T = np.array([[s, 0, -xmin * s], [0, s, -ymin * s], [0, 0, 1]], np.float64)
    CW, CH = int(math.ceil(W0 * s)), int(math.ceil(H0 * s))

    # warp every image into its own bounding box
    tiles = {}
    for i in order:
        h, w = imgs[i].shape[:2]
        M = T @ Href[i]
        c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), M).reshape(-1, 2)
        x0, y0 = max(0, int(math.floor(c[:, 0].min()))), max(0, int(math.floor(c[:, 1].min())))
        x1, y1 = min(CW, int(math.ceil(c[:, 0].max()))), min(CH, int(math.ceil(c[:, 1].max())))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        S = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ M
        img = cv2.warpPerspective(imgs[i], S, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        msk = cv2.warpPerspective(masks[i], S, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST)
        msk = cv2.erode(msk, np.ones((3, 3), np.uint8))
        pad = cv2.copyMakeBorder(msk, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        wgt = cv2.distanceTransform(pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
        tiles[i] = dict(img=img, mask=msk > 0, w=wgt, rect=(x0, y0, x1, y1), gain=np.ones(3, np.float32))

    # exposure gain compensation relative to the parent image
    if p.gain:
        for i in order:
            q = par[i]
            if q is None or i not in tiles or q not in tiles:
                continue
            a, b = tiles[i], tiles[q]
            x0, y0 = max(a["rect"][0], b["rect"][0]), max(a["rect"][1], b["rect"][1])
            x1, y1 = min(a["rect"][2], b["rect"][2]), min(a["rect"][3], b["rect"][3])
            if x1 <= x0 or y1 <= y0:
                continue
            ma = a["mask"][y0 - a["rect"][1]:y1 - a["rect"][1], x0 - a["rect"][0]:x1 - a["rect"][0]]
            mb = b["mask"][y0 - b["rect"][1]:y1 - b["rect"][1], x0 - b["rect"][0]:x1 - b["rect"][0]]
            ov = ma & mb
            if ov.sum() < 1500:
                continue
            ia = a["img"][y0 - a["rect"][1]:y1 - a["rect"][1], x0 - a["rect"][0]:x1 - a["rect"][0]][ov].astype(np.float32)
            ib = b["img"][y0 - b["rect"][1]:y1 - b["rect"][1], x0 - b["rect"][0]:x1 - b["rect"][0]][ov].astype(np.float32)
            g = ib.mean(0) * b["gain"] / np.maximum(ia.mean(0), 1.0)
            a["gain"] = np.clip(g, 0.7, 1.4).astype(np.float32)

    # feathered blending
    acc = np.zeros((CH, CW, 3), np.float32)
    wsum = np.zeros((CH, CW), np.float32)
    for i in order:
        if i not in tiles:
            continue
        t = tiles[i]
        x0, y0, x1, y1 = t["rect"]
        w = t["w"]
        acc[y0:y1, x0:x1] += t["img"].astype(np.float32) * t["gain"] * w[..., None]
        wsum[y0:y1, x0:x1] += w
    cover = wsum > 0
    out = np.zeros((CH, CW, 3), np.uint8)
    out[cover] = np.clip(acc[cover] / wsum[cover][:, None], 0, 255).astype(np.uint8)

    # layout view: footprint of every image
    lay = (out * 0.35).astype(np.uint8)
    palette = [(250, 189, 56), (153, 211, 52), (60, 146, 251), (250, 139, 167), (80, 220, 255)]
    thick = max(2, CW // 450)
    for i in order:
        h, w = imgs[i].shape[:2]
        c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), T @ Href[i])
        poly = c.reshape(-1, 2)
        cv2.polylines(lay, [poly.astype(np.int32)], True, palette[i % 5], thick, cv2.LINE_AA)
        ctr = poly.mean(0)
        cv2.putText(lay, str(i + 1), (int(ctr[0]) - thick * 6, int(ctr[1]) + thick * 6), cv2.FONT_HERSHEY_SIMPLEX,
                    thick * 0.9, palette[i % 5], thick, cv2.LINE_AA)

    if p.crop:
        pts = cv2.findNonZero(cover.astype(np.uint8))
        if pts is not None:
            x, y, bw, bh = cv2.boundingRect(pts)
            out, lay = out[y:y + bh, x:x + bw], lay[y:y + bh, x:x + bw]
    res.ok = True
    res.image, res.layout = out, lay
    res.seconds = time.perf_counter() - t0
    return res


# OpenCV engine (for comparison)
def stitch_cv(imgs, scans, p):
    t0 = time.perf_counter()
    res = Result(method="OpenCV Stitcher (" + ("scans" if scans else "panorama") + ")")
    res.matched = [None] * len(imgs)
    res.angles = [None] * len(imgs)
    st = cv2.Stitcher_create(cv2.Stitcher_SCANS if scans else cv2.Stitcher_PANORAMA)
    try:
        st.setPanoConfidenceThresh(0.3)
    except cv2.error:
        pass
    status, pano = st.stitch(imgs)
    names = {1: "Need more images / not enough matches", 2: "Homography estimation failed",
             3: "Camera parameter adjustment failed"}
    if status != 0 or pano is None:
        res.messages.append("OpenCV Stitcher: " + names.get(int(status), f"error {int(status)}"))
    else:
        out = crop_black(pano) if p.crop else pano
        res.ok, res.image, res.layout = True, out, None
    res.seconds = time.perf_counter() - t0
    return res


def stitch(raw_images, p):
    """raw_images: list of BGR images (full resolution)."""
    imgs = [resize_max(im, p.work) for im in raw_images]
    if p.engine == 0:
        return stitch_custom(imgs, p)
    return stitch_cv(imgs, p.engine == 2, p)


# GUI
W, H = 1280, 720
FONT, DUPLEX, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
BG, HEADER, CARD, CARD2 = (30, 26, 24), (38, 33, 30), (46, 41, 38), (62, 56, 52)
HOVER, BORDER, TRACK = (84, 76, 70), (72, 66, 62), (88, 80, 76)
TEXT, MUTED, WHITE = (238, 234, 232), (160, 152, 146), (255, 255, 255)
ACCENT, GOOD, WARN, BAD = (248, 189, 56), (110, 220, 110), (80, 180, 255), (90, 90, 255)

SIDEBAR = (16, 80, 250, 624)
VIEWER = (282, 80, 666, 624)
SETTINGS = (964, 80, 300, 470)
RESULTS = (964, 562, 300, 142)


@dataclass
class Slider:
    key: str
    label: str
    lo: float
    hi: float
    step: float


SLIDERS = [Slider("nfeat", "Keypoints per image", 1000, 8000, 500),
           Slider("ratio", "Match ratio test", 0.6, 0.9, 0.05),
           Slider("min_inl", "Min inliers", 8, 50, 1),
           Slider("work", "Work size (px)", 800, 3000, 200)]


def put(img, s, x, y, scale=0.5, color=TEXT, thick=1, anchor="l", font=FONT):
    if anchor != "l":
        tw = cv2.getTextSize(s, font, scale, thick)[0][0]
        x = x - tw if anchor == "r" else x - tw // 2
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, AA)


def text_w(s, scale, thick=1, font=FONT):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


def short(s, scale, max_w):
    if text_w(s, scale) <= max_w:
        return s
    while len(s) > 3 and text_w(s + "..", scale) > max_w:
        s = s[:-1]
    return s + ".."


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
        put(img, title, x + 16, y + 24, 0.4, MUTED)


def inside(pt, rect):
    x, y, w, h = rect
    return x <= pt[0] < x + w and y <= pt[1] < y + h


def fit(img, w, h, bg=(0, 0, 0)):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.full((h, w, 3), bg, np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out


def paste(dst, src, px, py):
    px, py = int(round(px)), int(round(py))
    h, w = src.shape[:2]
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(dst.shape[1], px + w), min(dst.shape[0], py + h)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = src[y0 - py:y1 - py, x0 - px:x1 - px]


class State:
    def __init__(self):
        self.items = []                    # dicts: name, path, img, thumb
        self.p = dict(engine=0, model=0, nfeat=4000, ratio=0.75, min_inl=15, work=1600, gain=0, crop=0)
        self.result = None
        self.view = "result"
        self.zoom, self.center = 1.0, None
        self.hits, self.drag, self.mouse = [], None, (0, 0)
        self.toast = ("", 0.0)
        self.open_req = self.stitch_req = self.save_req = self.busy = False
        self.vinfo = (0, 0, 1.0)           # viewport: top-left in image coords + image->screen scale
        self._cache = (None, None)

    def notify(self, text, sec=3.0):
        self.toast = (text, time.time() + sec)

    def add_paths(self, paths):
        for path in paths:
            if len(self.items) >= MAX_IMAGES:
                self.notify(f"Maximum {MAX_IMAGES} images")
                break
            img = imread_unicode(path)
            if img is None:
                self.notify("Cannot read " + os.path.basename(path))
                continue
            self.items.append(dict(name=os.path.basename(path), path=path, img=img, thumb=fit(img, 110, 74, CARD2)))
        self.result, self.zoom, self.center = None, 1.0, None

    def remove(self, i):
        if 0 <= i < len(self.items):
            self.items.pop(i)
            self.result, self.zoom, self.center = None, 1.0, None

    def remove_all(self):
        self.items.clear()
        self.result, self.zoom, self.center = None, 1.0, None

    def rotate_result(self):
        r = self.result
        if r is None or not r.ok:
            return
        r.image = cv2.rotate(r.image, cv2.ROTATE_90_CLOCKWISE)
        if r.layout is not None:
            r.layout = cv2.rotate(r.layout, cv2.ROTATE_90_CLOCKWISE)
        self.zoom, self.center = 1.0, None
        self._cache = (None, None)

    def current_image(self):
        r = self.result
        if r is None or not r.ok:
            return None
        return r.layout if (self.view == "layout" and r.layout is not None) else r.image

    def params(self):
        q = self.p
        return Params(q["engine"], q["model"], q["nfeat"], q["ratio"], q["min_inl"], q["work"],
                      q["gain"] == 0, q["crop"] == 0)

    def update_drag(self, x, y):
        d = self.drag
        if d[0] == "slider":
            _, sl, tx, tw = d
            frac = min(max((x - tx) / tw, 0.0), 1.0)
            n = round(frac * (sl.hi - sl.lo) / sl.step)
            v = min(max(sl.lo + n * sl.step, sl.lo), sl.hi)
            self.p[sl.key] = round(v, 2) if sl.step < 1 else int(round(v))
        else:
            _, x0, y0, c0 = d
            s = self.vinfo[2]
            self.center = (c0[0] - (x - x0) / s, c0[1] - (y - y0) / s)

    def zoom_at(self, factor, sx=None, sy=None):
        img = self.current_image()
        if img is None:
            return
        vx, vy, vw, vh = VIEWER
        vw, vh = vw - 8, vh - 8
        vx, vy = vx + 4, vy + 4
        ix0, iy0, s_old = self.vinfo
        if sx is None:
            sx, sy = vx + vw / 2, vy + vh / 2
        px, py = ix0 + (sx - vx) / s_old, iy0 + (sy - vy) / s_old          # image point under the cursor
        self.zoom = min(max(self.zoom * factor, 1.0), 16.0)
        s_new = min(vw / img.shape[1], vh / img.shape[0]) * self.zoom
        self.center = (px - ((sx - vx) - vw / 2) / s_new, py - ((sy - vy) - vh / 2) / s_new)
        if self.zoom <= 1.0:
            self.center = None


def on_mouse(event, x, y, flags, S):
    S.mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        for kind, data, rect in S.hits:
            if not inside((x, y), rect):
                continue
            if kind == "button":
                if data == "add":
                    S.open_req = True
                elif data == "stitch":
                    S.stitch_req = True
                elif data == "save":
                    S.save_req = True
                elif data == "clear":
                    S.remove_all()
                elif data == "rotate":
                    S.rotate_result()
            elif kind == "remove":
                S.remove(data)
            elif kind == "choice":
                key, idx = data
                S.p[key] = idx
            elif kind == "tab":
                S.view = data
            elif kind == "slider":
                S.drag = ("slider",) + data
                S.update_drag(x, y)
            elif kind == "viewer":
                if S.center:
                    c0 = S.center
                else:
                    c0 = (S.vinfo[0] + (VIEWER[2] - 8) / 2 / S.vinfo[2], S.vinfo[1] + (VIEWER[3] - 8) / 2 / S.vinfo[2])
                S.drag = ("pan", x, y, c0)
            return
    elif event == cv2.EVENT_MOUSEMOVE and S.drag:
        if S.drag[0] == "pan" and S.zoom <= 1.0:
            return
        S.update_drag(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        S.drag = None
    elif event == cv2.EVENT_MOUSEWHEEL and inside((x, y), VIEWER):
        try:
            d = cv2.getMouseWheelDelta(flags)
        except AttributeError:
            d = 1 if flags > 0 else -1
        S.zoom_at(1.25 if d > 0 else 0.8, x, y)


def render_viewer(c, S):
    x, y, w, h = VIEWER
    card(c, x, y, w, h)
    vw, vh = w - 8, h - 8
    out = np.zeros((vh, vw, 3), np.uint8)
    img = S.current_image()
    if img is None:
        if S.result is not None and not S.result.ok:
            put(out, "Stitching failed", vw // 2, vh // 2 - 14, 0.7, BAD, anchor="c")
            for i, m in enumerate(S.result.messages[:3]):
                put(out, short(m, 0.45, vw - 40), vw // 2, vh // 2 + 16 + i * 22, 0.45, MUTED, anchor="c")
        else:
            put(out, "Click '+ Add images' to load 2 to 5", vw // 2, vh // 2 - 8, 0.6, MUTED, anchor="c")
            put(out, "overlapping images, then press Stitch.", vw // 2, vh // 2 + 22, 0.6, MUTED, anchor="c")
        S.vinfo = (0, 0, 1.0)
    else:
        ih, iw = img.shape[:2]
        fit_s = min(vw / iw, vh / ih)
        s = fit_s * S.zoom
        cx, cy = S.center if S.center else (iw / 2, ih / 2)
        cx, cy = min(max(cx, 0), iw), min(max(cy, 0), ih)
        if S.center:
            S.center = (cx, cy)
        x0, y0 = cx - vw / (2 * s), cy - vh / (2 * s)
        S.vinfo = (x0, y0, s)
        if s <= 1.0:
            key = (id(img), round(s, 3))
            if S._cache[0] != key:
                S._cache = (key, cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            paste(out, S._cache[1], -x0 * s, -y0 * s)
        else:
            ix0, iy0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
            ix1, iy1 = min(iw, int(math.ceil(x0 + vw / s))), min(ih, int(math.ceil(y0 + vh / s)))
            if ix1 > ix0 and iy1 > iy0:
                roi = cv2.resize(img[iy0:iy1, ix0:ix1], None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                paste(out, roi, (ix0 - x0) * s, (iy0 - y0) * s)
    if S.result is not None and S.result.ok:
        zoom_txt = f"zoom x{S.zoom:.1f}" if S.zoom > 1.0 else "fit"
        put(out, f"{zoom_txt}   wheel: zoom   drag: move   0: fit", 14, vh - 12, 0.4, MUTED)
    if S.busy:
        out[:] = (out * 0.35).astype(np.uint8)
        put(out, "Stitching...", vw // 2, vh // 2, 0.9, WHITE, 1, "c", DUPLEX)
    c[y + 4:y + 4 + vh, x + 4:x + 4 + vw] = out
    if S.result is not None and S.result.ok:
        for i, (name, label) in enumerate((("result", "Result"), ("layout", "Layout"))):
            if name == "layout" and S.result.layout is None:
                continue
            rect = (x + 4 + 12 + i * 92, y + 4 + 12, 84, 26)
            sel = S.view == name
            rrect(c, *rect, ACCENT if sel else CARD2, 13)
            put(c, label, rect[0] + 42, rect[1] + 18, 0.45, BG if sel else TEXT, anchor="c")
            S.hits.append(("tab", name, rect))
    if time.time() < S.toast[1]:
        tw = text_w(S.toast[0], 0.5) + 28
        rrect(c, x + (w - tw) // 2, y + h - 56, tw, 30, (40, 40, 40), 12)
        put(c, S.toast[0], x + w // 2, y + h - 35, 0.5, WARN, anchor="c")


def render_sidebar(c, S):
    x, y, w, h = SIDEBAR
    card(c, x, y, w, h, f"IMAGES  ({len(S.items)}/{MAX_IMAGES})")
    rect = (x + 10, y + 34, w - 20, 32)
    full = len(S.items) >= MAX_IMAGES
    rrect(c, *rect, CARD2 if full else (HOVER if inside(S.mouse, rect) else ACCENT), 8)
    put(c, "+ Add images", rect[0] + rect[2] // 2, rect[1] + 21, 0.5, MUTED if full else BG, anchor="c")
    S.hits.append(("button", "add", rect))
    palette = [(250, 189, 56), (153, 211, 52), (60, 146, 251), (250, 139, 167), (80, 220, 255)]
    r = S.result
    for i in range(MAX_IMAGES):
        sy = y + 78 + i * 96
        slot = (x + 10, sy, w - 20, 90)
        rrect(c, *slot, CARD2 if i < len(S.items) else (40, 36, 33), 8)
        if i >= len(S.items):
            put(c, f"empty slot {i + 1}", slot[0] + slot[2] // 2, sy + 50, 0.42, (95, 90, 86), anchor="c")
            continue
        it = S.items[i]
        c[sy + 8:sy + 8 + 74, slot[0] + 8:slot[0] + 8 + 110] = it["thumb"]
        cv2.circle(c, (slot[0] + 20, sy + 20), 10, palette[i % 5], -1, AA)
        put(c, str(i + 1), slot[0] + 20, sy + 25, 0.45, BG, anchor="c")
        tx = slot[0] + 126
        put(c, short(it["name"], 0.4, slot[0] + slot[2] - tx - 24), tx, sy + 22, 0.4, TEXT)
        ih, iw = it["img"].shape[:2]
        put(c, f"{iw} x {ih}", tx, sy + 40, 0.38, MUTED)
        if r is not None and i < len(r.matched) and r.matched[i] is not None:
            if r.matched[i]:
                put(c, "matched", tx, sy + 58, 0.4, GOOD)
                if r.angles[i] is not None:
                    put(c, f"rot {r.angles[i]:+.0f} deg", tx, sy + 76, 0.38, MUTED)
            else:
                put(c, "NOT MATCHED", tx, sy + 58, 0.4, BAD)
        rm = (slot[0] + slot[2] - 24, sy + 4, 20, 20)
        rrect(c, *rm, HOVER if inside(S.mouse, rm) else (70, 64, 60), 6)
        put(c, "x", rm[0] + 10, rm[1] + 15, 0.45, TEXT, anchor="c")
        S.hits.append(("remove", i, rm))
    rect = (x + 10, y + h - 40, w - 20, 28)
    rrect(c, *rect, HOVER if inside(S.mouse, rect) else CARD2, 8)
    put(c, "Clear all", rect[0] + rect[2] // 2, rect[1] + 19, 0.42, TEXT, anchor="c")
    S.hits.append(("button", "clear", rect))


def draw_pills(c, S, label, key, names, ry, x0, wi, enabled=True):
    put(c, label, x0, ry + 12, 0.45, TEXT if enabled else MUTED)
    n, gap = len(names), 6
    bw = (wi - gap * (n - 1)) // n
    for i, name in enumerate(names):
        rect = (x0 + i * (bw + gap), ry + 20, bw, 26)
        sel = S.p[key] == i
        fill = (ACCENT if enabled else (120, 110, 90)) if sel else (HOVER if inside(S.mouse, rect) else CARD2)
        rrect(c, *rect, fill, 13)
        put(c, name, rect[0] + bw // 2, rect[1] + 18, 0.38, BG if sel else (TEXT if enabled else MUTED), anchor="c")
        S.hits.append(("choice", (key, i), rect))


def render_settings(c, S):
    x, y, w, h = SETTINGS
    card(c, x, y, w, h, "STITCH SETTINGS")
    x0, wi = x + 16, w - 32
    custom = S.p["engine"] == 0
    ry = y + 38
    draw_pills(c, S, "Engine", "engine", ENGINE_NAMES, ry, x0, wi)
    ry += 50
    draw_pills(c, S, "Geometric model", "model", ("Auto", "Similarity", "Homography"), ry, x0, wi, custom)
    ry += 50
    for sl in SLIDERS:
        v = S.p[sl.key]
        en = custom or sl.key == "work"
        put(c, sl.label, x0, ry + 10, 0.42, TEXT if en else MUTED)
        put(c, f"{v:.2f}" if sl.step < 1 else str(int(v)), x0 + wi, ry + 10, 0.45, ACCENT if en else MUTED, anchor="r")
        tx, tw, ty = x0 + 9, wi - 18, ry + 26
        frac = (v - sl.lo) / (sl.hi - sl.lo)
        col = ACCENT if en else (120, 110, 90)
        rrect(c, tx, ty - 3, tw, 6, TRACK, 3)
        rrect(c, tx, ty - 3, max(6, int(tw * frac)), 6, col, 3)
        kx = tx + int(tw * frac)
        cv2.circle(c, (kx, ty), 9, WHITE, -1, AA)
        cv2.circle(c, (kx, ty), 4, col, -1, AA)
        S.hits.append(("slider", (sl, tx, tw), (x0, ry + 14, wi, 24)))
        ry += 38
    half = (wi - 8) // 2
    draw_pills(c, S, "Exposure fix", "gain", ("On", "Off"), ry, x0, half, custom)
    draw_pills(c, S, "Crop edges", "crop", ("On", "Off"), ry, x0 + half + 8, half)
    rect = (x0, y + h - 52, wi, 38)
    ok = len(S.items) >= 2
    rrect(c, *rect, ACCENT if ok else CARD2, 10)
    put(c, "STITCH  (Enter)", rect[0] + wi // 2, rect[1] + 25, 0.6, BG if ok else MUTED, 1, "c", DUPLEX)
    S.hits.append(("button", "stitch", rect))


def render_results(c, S):
    x, y, w, h = RESULTS
    card(c, x, y, w, h, "RESULT")
    r = S.result
    if r is None:
        for i, ln in enumerate(wrap("Images must overlap by about 25 % or more. Order and orientation do not matter.", 0.42, w - 32)):
            put(c, ln, x + 16, y + 50 + i * 18, 0.42, MUTED)
        return
    cy = y + 46
    if r.ok:
        ih, iw = r.image.shape[:2]
        put(c, f"{iw} x {ih} px     {r.seconds:.1f} s", x + 16, cy, 0.48, GOOD)
        cy += 20
    put(c, short(r.method, 0.38, w - 32), x + 16, cy, 0.38, MUTED)
    cy += 18
    if r.pairs:
        txt = "   ".join(f"{i + 1}-{j + 1}: {n} ({m[:3]})" for i, j, n, m in sorted(r.pairs, key=lambda t: -t[2])[:6])
        for ln in wrap("Matches  " + txt, 0.38, w - 32)[:2]:
            put(c, ln, x + 16, cy, 0.38, TEXT)
            cy += 16
    for m in r.messages[:2]:
        for ln in wrap(m, 0.38, w - 32)[:2]:
            put(c, ln, x + 16, cy, 0.38, WARN)
            cy += 16


def render(S):
    c = np.full((H, W, 3), BG, np.uint8)
    S.hits = []
    cv2.rectangle(c, (0, 0), (W, 64), HEADER, -1)
    cv2.rectangle(c, (0, 64), (W, 66), ACCENT, -1)
    cv2.circle(c, (34, 32), 12, ACCENT, 2, AA)
    cv2.circle(c, (46, 32), 12, ACCENT, 2, AA)
    put(c, "Image Stitcher", 72, 31, 0.8, TEXT, 1, font=DUPLEX)
    put(c, "Up to 5 overlapping images, any order, any orientation", 72, 51, 0.4, MUTED)
    stats = [("IMAGES", f"{len(S.items)}/{MAX_IMAGES}")]
    if S.result is not None and S.result.ok:
        ih, iw = S.result.image.shape[:2]
        stats = [("TIME", f"{S.result.seconds:.1f} s"), ("OUTPUT", f"{iw}x{ih}")] + stats
    x = W - 16
    for label, value in stats:
        tw = max(text_w(value, 0.55), text_w(label, 0.32)) + 24
        x -= tw
        rrect(c, x, 12, tw, 40, CARD, 8)
        put(c, label, x + 12, 26, 0.32, MUTED)
        put(c, value, x + 12, 45, 0.55, TEXT)
        x -= 10
    render_sidebar(c, S)
    render_viewer(c, S)
    render_settings(c, S)
    render_results(c, S)
    if S.result is not None and S.result.ok:
        rect = (VIEWER[0] + VIEWER[2] - 108, VIEWER[1] + 16, 92, 26)
        rrect(c, *rect, HOVER if inside(S.mouse, rect) else CARD2, 13)
        put(c, "Save (S)", rect[0] + 46, rect[1] + 18, 0.42, TEXT, anchor="c")
        S.hits.append(("button", "save", rect))
        rect = (VIEWER[0] + VIEWER[2] - 214, VIEWER[1] + 16, 98, 26)
        rrect(c, *rect, HOVER if inside(S.mouse, rect) else CARD2, 13)
        put(c, "Rotate (R)", rect[0] + 49, rect[1] + 18, 0.42, TEXT, anchor="c")
        S.hits.append(("button", "rotate", rect))
    S.hits.append(("viewer", None, (VIEWER[0] + 4, VIEWER[1] + 4, VIEWER[2] - 8, VIEWER[3] - 8)))
    return c


def ask_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(title=f"Select up to {MAX_IMAGES} images",
                                            filetypes=[("Images", " ".join("*" + e for e in IMG_EXT)), ("All files", "*.*")])
        root.destroy()
        return list(paths)
    except Exception:
        return None


def save_result(S, folder=None):
    img = S.result.image
    name = f"stitched_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(folder or ".", name)
    cv2.imencode(".png", img)[1].tofile(path)
    return path


def run_gui(paths):
    S = State()
    S.add_paths(paths[:MAX_IMAGES])
    win = "Image Stitcher"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse, S)
    if not paths:
        S.open_req = True
    while True:
        cv2.imshow(win, render(S))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("o"), ord("O")):
            S.open_req = True
        elif key in (13, 10):
            S.stitch_req = True
        elif key in (ord("s"), ord("S")):
            S.save_req = True
        elif key in (ord("r"), ord("R")):
            S.rotate_result()
        elif key in (ord("v"), ord("V")) and S.result is not None and S.result.layout is not None:
            S.view = "layout" if S.view == "result" else "result"
        elif key in (ord("+"), ord("=")):
            S.zoom_at(1.25)
        elif key in (ord("-"), ord("_")):
            S.zoom_at(0.8)
        elif key == ord("0"):
            S.zoom, S.center = 1.0, None
        elif key in (8, 127) and S.items:
            S.remove(len(S.items) - 1)

        if S.open_req:
            S.open_req = False
            picked = ask_files()
            if picked is None:
                S.notify("File dialog unavailable: pass images on the command line")
            else:
                S.add_paths(picked)
        if S.stitch_req:
            S.stitch_req = False
            if len(S.items) < 2:
                S.notify("Load at least 2 images")
            else:
                S.busy = True
                cv2.imshow(win, render(S))
                cv2.waitKey(1)
                try:
                    S.result = stitch([it["img"] for it in S.items], S.params())
                except cv2.error as e:
                    S.result = Result(messages=["OpenCV error: " + str(e).splitlines()[0]])
                S.busy = False
                S.view, S.zoom, S.center = "result", 1.0, None
                S.notify("Done" if S.result.ok else "Stitching failed")
        if S.save_req:
            S.save_req = False
            if S.result is not None and S.result.ok:
                S.notify("Saved " + save_result(S))
            else:
                S.notify("Nothing to save yet")
    cv2.destroyAllWindows()


def run_headless(paths, out, p):
    imgs = []
    for path in paths[:MAX_IMAGES]:
        im = imread_unicode(path)
        if im is None:
            sys.exit(f"Cannot read {path}")
        imgs.append(im)
    if len(imgs) < 2:
        sys.exit("Give at least 2 images")
    r = stitch(imgs, p)
    print(r.method, f"- {r.seconds:.1f} s")
    for i, j, n, m in r.pairs:
        print(f"  pair {i + 1}-{j + 1}: {n} inliers ({m})")
    for i, (ok, ang) in enumerate(zip(r.matched, r.angles)):
        if ok is not None:
            print(f"  image {i + 1}: " + (f"matched, rotation {ang:+.0f} deg" if ok else "NOT MATCHED"))
    for m in r.messages:
        print("  !", m)
    if not r.ok:
        sys.exit("Stitching failed")
    cv2.imencode(".png", r.image)[1].tofile(out)
    print("Saved", out, f"({r.image.shape[1]}x{r.image.shape[0]})")


def main():
    ap = argparse.ArgumentParser(description="Stitch up to 5 overlapping images")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--headless", action="store_true", help="no window, write --out")
    ap.add_argument("--out", default="stitched.png")
    ap.add_argument("--engine", type=int, default=0, choices=(0, 1, 2), help="0 custom, 1 OpenCV panorama, 2 OpenCV scans")
    ap.add_argument("--model", type=int, default=0, choices=(0, 1, 2), help="0 auto, 1 similarity, 2 homography")
    args = ap.parse_args()
    if len(args.images) > MAX_IMAGES:
        print(f"Only the first {MAX_IMAGES} images are used.")
    if args.headless:
        run_headless(args.images, args.out, Params(engine=args.engine, model=args.model))
    else:
        run_gui(args.images)


if __name__ == "__main__":
    main()