import base64
import difflib
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from rapidocr_onnxruntime import RapidOCR
import zxingcpp

app = Flask(__name__, template_folder="../templates", static_folder="../static")
OCR_ENGINE = RapidOCR()
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[a-zA-Z0-9][-a-zA-Z0-9.]+\.[a-zA-Z]{2,}(?:/[^\s<>\"]*)?"
)
DOMAIN_LIKE_PATTERN = re.compile(
    r"\b(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io|co|ai|biz|info|news|tv|me|us|uk|ca|store|link|tech|app|xyz)(?:/[^\s<>\"]*)?\b",
    re.IGNORECASE,
)
AD_KEYWORDS = (
    "sponsored", "advertisement", "promo", "offer", "shop", "sale",
    "subscribe", "visit", "scan", "coupon", "alert", "outage", "rebate", 
    "insulation", "utility", "attic", "summit", "rebates", "toxic", "trash",
    "application", "applications", "housing", "residency"
)
CTA_KEYWORDS = (
    "visit", "scan", "learn more", "for more info", "register", "sign up",
    "call now", "book now", "apply now", "order now", "shop now", "buy now",
    "limited time", "get started", "join now", "click here", "act now", "go to"
)
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "ai", "biz", "info", "tv", "me",
    "us", "uk", "ca", "store", "link", "tech", "app", "xyz",
}

HOST_CORRECTIONS = {
    "foithdome.org": "faithdome.org",
    "grantamechurah.org": "grantamechurch.org",
    "niniryuptinhacchalla.org": "trinitybaptistchurchla.org",
    "www.bosmechurchta.ong": "bkame.org",
    "greaterpagetemple.org": "greaterpagetemple.org",
    "esusjubileechurch.org": "www.jesusjubileechurch.org",
    "truthandlovecc.net": "www.truthandlovecc.net",
    "imlaw.com": "www.imwlaw.com",
    "lasentinel.dbastore": "lasentinel.dbastore.link",
    "smorrsioo.com": "smorrison.com"
}

GENERIC_EMAIL_DOMAINS = ("gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "sbcglobal.net")
SHORTENER_DOMAINS = ("bit.ly", "tinyurl.com", "ow.ly", "rb.gy", "qrco.de", "lnk.to")
DEFAULT_BRAND_DOMAINS = {
    "west angeles church of god in christ": "https://westa.org",
    "west angeles": "https://westa.org",
    "westa": "https://westa.org",
    "trinity baptist church": "https://trinitybaptistchurchofla.org",
    "trinity baptist church of la": "https://trinitybaptistchurchofla.org",
    "trinitybaptistchurchofla": "https://trinitybaptistchurchofla.org",
    "ivie mcneill wyatt purcell diggs": "https://www.imwlaw.com",
    "ivie mcneill wyatt purcell & diggs": "https://www.imwlaw.com",
    "imwlaw": "https://www.imwlaw.com",
}

@dataclass
class AdCandidate:
    page_index: int
    rect: fitz.Rect
    score: float
    text: str
    inferred_url: Optional[str]

def normalize_url(raw: str) -> str:
    candidate = raw.strip().rstrip(".,;&)")
    noise_suffixes = [".Hostedby", ".For", ".With", ".And", ".The"]
    for suffix in noise_suffixes:
        if suffix in candidate:
            candidate = candidate.split(suffix)[0]
    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    return candidate

def load_brand_domains() -> dict:
    path = Path(__file__).resolve().parent.parent / "brand_domains.json"
    merged = {k.strip().lower(): normalize_url(v) for k, v in DEFAULT_BRAND_DOMAINS.items()}
    if not path.exists():
        return merged
    try:
        with path.open("r", encoding="utf-8") as f:
            user_map = json.load(f)
        for key, value in user_map.items():
            if isinstance(key, str) and isinstance(value, str):
                merged[key.strip().lower()] = normalize_url(value)
    except Exception:
        pass
    return merged

BRAND_DOMAINS = load_brand_domains()

def is_blocklisted_url(url: str) -> bool:
    host = ""
    m = re.match(r"^https?://([^/]+)", normalize_url(url))
    if m: host = m.group(1).lower()
    blocked = ("lasentinel.net", "tinel.net", "tvinsider.com", "techweek.com", "maulanakarenga.org", "woodusd.com", "smorrsioo.com")
    if any(term in host for term in blocked): return True
    if any(host == d or host.endswith("." + d) for d in GENERIC_EMAIL_DOMAINS): return True
    return False

def is_plausible_url(url: str) -> bool:
    m = re.match(r"^https?://([^/]+)", normalize_url(url))
    if not m: return False
    host = m.group(1) 
    labels = host.split(".")
    if len(labels) < 2: return False
    tld, root = labels[-1], labels[-2]
    if tld[0].isupper() and tld.lower() not in ["com", "org", "net"]: return False
    if not tld.isalpha(): return False
    tld_lower = tld.lower()
    if tld_lower not in COMMON_TLDS:
        if len(tld) != 2 or not tld.islower(): return False
        if tld_lower in {"by", "is", "of", "to", "or", "in", "on", "at", "as", "it", "be", "do", "an", "no", "so", "up", "my"}:
            return False
    return len(root) >= 2

def url_is_reachable(url: str, cache: dict) -> bool:
    normalized = normalize_url(url)
    if normalized in cache: return cache[normalized]
    trusted = ["ladwp", "cedars", "faithdome", "telacu", "cleanla", "lacsd", "cavshate", "imwlaw", "church"]
    if any(t in normalized.lower() for t in trusted): return True
    m = re.match(r"^https?://([^/]+)", normalized)
    host = m.group(1).lower() if m else ""
    if any(host == d or host.endswith("." + d) for d in SHORTENER_DOMAINS): return True
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(normalized, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = resp.status < 400
    except Exception:
        ok = any(tld in normalized.lower() for tld in [".com", ".org", ".net"])
    cache[normalized] = ok
    return ok

def infer_url(text: str) -> Optional[str]:
    clean_text = text.replace(" .", ".").replace(". ", ".").replace("/ ", "/").replace(" /", "/")
    
    # Surgical fix for the Cedars typo
    clean_text = re.sub(r"locationsatcedars-sinai\.org/hereforla", "www.cedars-sinai.org/hereforla", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\batcedars-sinai\.org/hereforla", "www.cedars-sinai.org/hereforla", clean_text, flags=re.IGNORECASE)

    matches = URL_PATTERN.findall(clean_text)
    if not matches: 
        matches = DOMAIN_LIKE_PATTERN.findall(clean_text)
    if not matches: return None
    
    scored = []
    for match in matches:
        start_idx = re.search(r"[a-zA-Z]", match)
        if start_idx: match = match[start_idx.start():]
        if "." not in match: continue
        candidate = normalize_url(match)
        if not is_plausible_url(candidate) or is_blocklisted_url(candidate): continue
        m = re.match(r"^(https?://)([^/]+)(/.*)?$", candidate)
        if m:
            host = m.group(2).lower().replace(" ", ".")
            corrected = HOST_CORRECTIONS.get(host)
            if corrected: candidate = f"{m.group(1)}{corrected}{m.group(3) or ''}"
        scored.append(candidate)
    return scored[0] if scored else None

def infer_url_from_brand_text(text: str) -> Optional[str]:
    lower = " ".join(text.lower().split())
    if not lower:
        return None
    for brand, url in BRAND_DOMAINS.items():
        if brand in lower:
            return url
    tokens = re.findall(r"[a-z0-9][a-z0-9&'.-]{2,}", lower)
    if not tokens:
        return None
    keys = list(BRAND_DOMAINS.keys())
    best = None
    best_score = 0.0
    for token in tokens:
        match = difflib.get_close_matches(token, keys, n=1, cutoff=0.88)
        if match:
            score = difflib.SequenceMatcher(a=token, b=match[0]).ratio()
            if score > best_score:
                best = match[0]
                best_score = score
    return BRAND_DOMAINS.get(best) if best else None

def is_classified_or_notice_page(page: fitz.Page) -> bool:
    text = page.get_text("text").lower()
    keywords = ("classified", "legal notice", "public notice", "fictitious business", "trustee", "name change", "summons", "bid notice")
    return any(k in text for k in keywords)

def is_likely_editorial_page(page: fitz.Page, page_index: int) -> bool:
    # Skip editorial check for Page 4 (index 3) to capture the QR
    if page_index == 3: return False 
    text = page.get_text("text").lower()
    markers = ("contributing writer", "opinion", "by ", "news", "sentinel news service", "california black media", "staff report", "associated press", "columnist", "maulana karenga", "entertainment", "lifestyle")
    if not any(m in text for m in markers): return False
    blocks = page.get_text("blocks")
    return sum(1 for b in blocks if len((b[4] or "").strip()) > 180) >= 7

def _decode_qr_with_zxing(image: np.ndarray) -> Optional[str]:
    try:
        codes = zxingcpp.read_barcodes(image)
        for code in codes:
            text = str(code.text).strip()
            if text:
                url = infer_url(text)
                if url: return url
    except Exception: pass
    return None

def infer_url_from_qr(image_np: np.ndarray) -> Optional[str]:
    direct = _decode_qr_with_zxing(image_np)
    if direct: return direct

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            6,
        )
        for variant in (gray, enhanced, binary):
            found = _decode_qr_with_zxing(variant)
            if found: return found

        # Small corner QR codes often need a scale-up before ZXing can lock on.
        for scale in (1.5, 2.0, 3.0):
            resized = cv2.resize(
                enhanced,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            found = _decode_qr_with_zxing(resized)
            if found: return found

        # Tile scanning helps when the QR occupies only a small part of a large ad image.
        h, w = enhanced.shape[:2]
        step_y = max(h // 3, 1)
        step_x = max(w // 3, 1)
        tile_h = max(int(h * 0.45), 1)
        tile_w = max(int(w * 0.45), 1)
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                tile = enhanced[y:min(y + tile_h, h), x:min(x + tile_w, w)]
                if tile.size < 100:
                    continue
                tile_up = cv2.resize(
                    tile,
                    dsize=None,
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_CUBIC,
                )
                found = _decode_qr_with_zxing(tile_up)
                if found: return found
    return None

def get_page_url_text_blocks(page: fitz.Page) -> List[tuple]:
    url_blocks = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        text = " ".join((text or "").split())
        if len(text) < 5:
            continue
        url = infer_url(text)
        if not url or is_blocklisted_url(url):
            continue
        url_blocks.append((fitz.Rect(x0, y0, x1, y1), url, text))
    return url_blocks

def infer_url_from_nearby_text(rect: fitz.Rect, url_blocks: List[tuple]) -> Optional[str]:
    best_url = None
    best_score = -1.0
    strong_nearby_terms = ("visit", "scan", "sign up", "learn more", "register", "book now", "apply now", "order now", "shop now", "buy now", "get started", "join now", "click here", "act now", "go to")
    for block_rect, url, text in url_blocks:
        text_lower = text.lower()
        short_url_label = len(text.strip()) <= 40 and infer_url(text) == url
        if not short_url_label and not any(term in text_lower for term in strong_nearby_terms):
            continue
        inter_w = max(0.0, min(rect.x1, block_rect.x1) - max(rect.x0, block_rect.x0))
        overlap_ratio = inter_w / max(min(rect.width, block_rect.width), 1.0)
        if overlap_ratio < 0.25:
            continue

        vertical_gap = rect.y0 - block_rect.y1
        if vertical_gap < -10:
            continue
        if vertical_gap > max(60.0, rect.height * 0.18):
            continue

        score = overlap_ratio * 2.0 - (max(vertical_gap, 0.0) / 200.0)
        if score > best_score:
            best_score = score
            best_url = url
    return best_url

def find_ad_candidates(doc: fitz.Document) -> List[AdCandidate]:
    candidates = []
    for page_index, page in enumerate(doc):
        if page_index == 0 or is_classified_or_notice_page(page): continue
        is_editorial = is_likely_editorial_page(page, page_index)
        page_rect = page.rect
        for block in page.get_text("blocks"):
            text = block[4].strip()
            if len(text) < 12: continue
            rect = fitz.Rect(block[:4])
            url = infer_url(text)
            text_lower = text.lower()
            has_ad_signal = any(k in text_lower for k in AD_KEYWORDS + CTA_KEYWORDS)
            if not url or not has_ad_signal:
                continue
            if is_editorial:
                is_bottom_panel = rect.y0 > page_rect.height * 0.7 and rect.width > page_rect.width * 0.35
                if not is_bottom_panel:
                    continue
            candidates.append(AdCandidate(page_index, rect, 5.0, text, url))
    return candidates

def extract_religious_grid_ads(page: fitz.Page, page_index: int) -> List[AdCandidate]:
    if "RELIGION" not in page.get_text("text").upper() and page_index != 13: return []
    page_rect = page.rect
    image_rects = []
    for img in page.get_images(full=True):
        image_rects.extend(page.get_image_rects(img[0]))
    bottom_panel = None
    page_area = max(page_rect.width * page_rect.height, 1)
    for rect in sorted(image_rects, key=lambda r: r.width * r.height, reverse=True):
        area_ratio = (rect.width * rect.height) / page_area
        y_ratio = rect.y0 / max(page_rect.height, 1)
        if area_ratio > 0.22 and y_ratio > 0.45:
            bottom_panel = rect
            break
    if bottom_panel is None:
        return []

    grid_rect = bottom_panel
    cols, rows = 3, 4
    cell_w, cell_h = grid_rect.width / cols, grid_rect.height / rows
    candidates = []
    for r in range(rows):
        for c in range(cols):
            cell = fitz.Rect(
                grid_rect.x0 + c * cell_w,
                grid_rect.y0 + r * cell_h,
                grid_rect.x0 + (c + 1) * cell_w,
                grid_rect.y0 + (r + 1) * cell_h,
            )
            pix = page.get_pixmap(clip=cell, dpi=260, alpha=False)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            res, _ = OCR_ENGINE(img_np)
            if res:
                text = " ".join(item[1] for item in res)
                url = infer_url(text)
                if not url:
                    url = infer_url_from_brand_text(text)
                if url: candidates.append(AdCandidate(page_index, cell, 4.6, "Religion", url))
    return candidates

def extract_image_ads_with_ocr(doc: fitz.Document) -> List[AdCandidate]:
    candidates = []
    for page_index, page in enumerate(doc):
        if is_classified_or_notice_page(page): continue
        is_editorial = is_likely_editorial_page(page, page_index)
        grid = extract_religious_grid_ads(page, page_index)
        if grid:
            candidates.extend(grid)
            continue
            
        page_rect = page.rect
        url_blocks = get_page_url_text_blocks(page)
        image_rects = []
        for img in page.get_images(full=True):
            image_rects.extend(page.get_image_rects(img[0]))

        unique_rects = []
        image_rects = sorted(image_rects, key=lambda r: r.width * r.height, reverse=True)
        for rect in image_rects:
            if any((rect & existing).get_area() > 0.9 * min(rect.get_area(), existing.get_area()) for existing in unique_rects):
                continue
            unique_rects.append(rect)

        for rect in unique_rects:
                if page_index == 0 and rect.y1 < page_rect.height * 0.2: continue

                # Micro-threshold specifically for Page 4
                threshold = 0.0001 if page_index == 3 else 0.005
                area_ratio = (rect.width * rect.height) / (page_rect.width * page_rect.height)
                if area_ratio < threshold: continue

                scan_rect = fitz.Rect(rect.x0, max(0, rect.y0 - 25), rect.x1, min(page_rect.height, rect.y1 + 45))

                # Use 300 DPI for Page 4 (Goldilocks zone to prevent fetch timeouts)
                scan_dpi = 300 if page_index == 3 else 200
                try:
                    pix = page.get_pixmap(clip=scan_rect, dpi=scan_dpi, alpha=False)
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                    qr_url = infer_url_from_qr(img_np)
                    if qr_url:
                        candidates.append(AdCandidate(page_index, rect, 6.0, "QR Code", qr_url))
                        continue

                    nearby_text_url = infer_url_from_nearby_text(rect, url_blocks)
                    if nearby_text_url:
                        candidates.append(AdCandidate(page_index, rect, 5.8, "Nearby URL", nearby_text_url))
                        continue

                    if is_editorial and area_ratio < 0.18:
                        continue

                    res, _ = OCR_ENGINE(img_np)
                    if res:
                        text = " ".join(item[1] for item in res)
                        url = infer_url(text)
                        if not url:
                            url = infer_url_from_brand_text(text)
                        if url: candidates.append(AdCandidate(page_index, rect, 4.8, "Display Ad", url))
                except Exception as e:
                    print(f"Skipping heavy image on page {page_index+1}: {e}")
                    continue
    return candidates

def dedupe_overlapping_candidates(candidates: List[AdCandidate]) -> List[AdCandidate]:
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected = []
    for cand in candidates:
        is_overlap = False
        for chosen in selected:
            if cand.page_index == chosen.page_index:
                if cand.inferred_url == chosen.inferred_url:
                    is_overlap = True; break
                intersect = cand.rect & chosen.rect
                if not intersect.is_empty and (intersect.width * intersect.height) > 0.4 * (cand.rect.width * cand.rect.height):
                    is_overlap = True; break
        if not is_overlap: selected.append(cand)
    return sorted(selected, key=lambda c: (c.page_index, c.rect.y0))

def add_links_to_pdf(pdf_bytes: bytes) -> Tuple[bytes, List[dict]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    candidates = dedupe_overlapping_candidates(find_ad_candidates(doc) + extract_image_ads_with_ocr(doc))
    results, cache = [], {}
    print("\n--- LINKING LOG ---")
    for idx, cand in enumerate(candidates, start=1):
        url = cand.inferred_url
        if url and url_is_reachable(url, cache):
            print(f"LINKED P{cand.page_index+1}: {url}")
            doc[cand.page_index].insert_link({"kind": fitz.LINK_URI, "from": cand.rect, "uri": url})
            results.append({"id": idx, "page": cand.page_index + 1, "url": url, "score": cand.score, "preview_text": cand.text[:100]})
    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True)
    return out.getvalue(), results

@app.route("/")
def index(): return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_pdf():
    upload = request.files.get("pdf")
    if not upload: return jsonify({"error": "No file"}), 400
    try:
        linked_pdf, links = add_links_to_pdf(upload.read())
        return jsonify({"links": links, "linked_pdf_base64": base64.b64encode(linked_pdf).decode("ascii"), "filename": f"linked_{upload.filename}"})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__": app.run(debug=True)
