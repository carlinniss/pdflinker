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
    r"(?:(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"]*)?)"
)
DOMAIN_LIKE_PATTERN = re.compile(
    r"\b(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|edu|gov|io|co|ai|biz|info|news|tv|me)\b",
    re.IGNORECASE,
)
AD_KEYWORDS = (
    "sponsored",
    "advertisement",
    "promo",
    "offer",
    "shop",
    "sale",
    "subscribe",
    "visit",
    "scan",
    "coupon",
)
AD_SIZE_TARGETS = (
    (1.0, 0.22),    # full page
    (0.75, 0.16),   # 3/4 page
    (2 / 3, 0.14),  # 2/3 page
    (0.5, 0.12),    # 1/2 page
    (1 / 3, 0.10),  # 1/3 page
    (0.25, 0.08),   # 1/4 page
    (0.125, 0.08),  # 1/8 page
)
CTA_KEYWORDS = (
    "visit",
    "scan",
    "learn more",
    "for more info",
    "register",
    "sign up",
    "call now",
    "book now",
    "apply now",
    "order now",
    "shop now",
    "buy now",
    "limited time",
)
SOCIAL_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "youtube.com",
)
COMMON_TLDS = {
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "io",
    "co",
    "ai",
    "biz",
    "info",
    "tv",
    "me",
}
GENERIC_EMAIL_DOMAINS = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "aol.com",
    "icloud.com",
)
DEFAULT_BRAND_DOMAINS = {
    "walmart": "https://www.walmart.com",
    "target": "https://www.target.com",
    "amazon": "https://www.amazon.com",
    "costco": "https://www.costco.com",
    "best buy": "https://www.bestbuy.com",
    "home depot": "https://www.homedepot.com",
    "lowes": "https://www.lowes.com",
    "mcdonalds": "https://www.mcdonalds.com",
    "burger king": "https://www.bk.com",
    "subway": "https://www.subway.com",
    "cvs": "https://www.cvs.com",
    "walgreens": "https://www.walgreens.com",
    "kroger": "https://www.kroger.com",
    "ford": "https://www.ford.com",
    "toyota": "https://www.toyota.com",
    "cedars sinai": "https://www.cedars-sinai.org",
    "cedars-sinai": "https://www.cedars-sinai.org",
    "ivie mcneill wyatt purcell diggs": "https://www.imwlaw.com",
    "ivie mcneill wyatt purcell & diggs": "https://www.imwlaw.com",
    "imwpd": "https://www.imwlaw.com",
    "imwlaw": "https://www.imwlaw.com",
}


def load_brand_domains() -> dict:
    path = Path(__file__).resolve().parent.parent / "brand_domains.json"
    if not path.exists():
        return DEFAULT_BRAND_DOMAINS
    try:
        with path.open("r", encoding="utf-8") as f:
            user_map = json.load(f)
        normalized = {}
        for key, value in user_map.items():
            if isinstance(key, str) and isinstance(value, str):
                normalized[key.strip().lower()] = normalize_url(value)
        return {**DEFAULT_BRAND_DOMAINS, **normalized}
    except Exception:
        return DEFAULT_BRAND_DOMAINS


BRAND_DOMAINS = load_brand_domains()


@dataclass
class AdCandidate:
    page_index: int
    rect: fitz.Rect
    score: float
    text: str
    inferred_url: Optional[str]


def normalize_url(raw: str) -> str:
    candidate = raw.strip().rstrip(".,;)")
    if not candidate.startswith(("http://", "https://")):
        candidate = f"https://{candidate}"
    return candidate


def _score_url_candidate(raw: str) -> float:
    candidate = normalize_url(raw)
    m = re.match(r"^https?://([^/]+)(/.*)?$", candidate)
    if not m:
        return 0.0
    host = m.group(1).lower()
    path = m.group(2) or ""
    labels = host.split(".")
    if len(labels) < 2:
        return 0.0
    tld = labels[-1]
    root = labels[-2]

    score = 0.0
    if tld in {"com", "org", "net", "edu", "gov", "io", "co", "ai"}:
        score += 1.5
    if len(root) >= 5:
        score += 1.0
    if "-" in root:
        score += 0.8
    if len(labels) > 2:
        score += 0.2
    if path and len(path) >= 3:
        score += 0.9
    if len(root) <= 3:
        score -= 0.8
    if root.isupper():
        score -= 0.4
    # Penalize suspicious short OCR fragments like "me.fory".
    if len(root) <= 4 and len(tld) >= 4:
        score -= 1.0
    return score


def _canonicalize_with_known_domains(raw: str) -> str:
    candidate = normalize_url(raw)
    m = re.match(r"^(https?://)([^/]+)(/.*)?$", candidate)
    if not m:
        return candidate
    scheme, host, path = m.group(1), m.group(2).lower(), m.group(3) or ""
    for known_url in BRAND_DOMAINS.values():
        km = re.match(r"^https?://([^/]+)", normalize_url(known_url))
        if not km:
            continue
        known_host = km.group(1).lower()
        host_cmp = host[4:] if host.startswith("www.") else host
        known_cmp = known_host[4:] if known_host.startswith("www.") else known_host
        if host_cmp.endswith(known_cmp) and host_cmp != known_cmp:
            return f"{scheme}{known_host}{path}"
    return candidate


def infer_url(text: str) -> Optional[str]:
    matches = URL_PATTERN.findall(text)
    canonical_matches = [_canonicalize_with_known_domains(match) for match in matches if "." in match]
    scored = [(match, _score_url_candidate(match)) for match in canonical_matches]
    if scored:
        best_match, best_score = max(scored, key=lambda item: item[1])
        if best_score >= 0.6:
            return normalize_url(best_match)

    domain_matches = DOMAIN_LIKE_PATTERN.findall(text)
    if domain_matches:
        scored_domains = [(m, _score_url_candidate(m)) for m in domain_matches]
        best_domain, best_score = max(scored_domains, key=lambda item: item[1])
        if best_score >= 0.6:
            return normalize_url(best_domain)
    return None


def is_blocklisted_url(url: str) -> bool:
    m = re.match(r"^https?://([^/]+)", normalize_url(url))
    if not m:
        return False
    host = m.group(1).lower()
    blocked = (
        "lasentinel.net",
        "lasentinel.dbastore",
        "lasentinel.legaladstore",
        "legaladstore.com",
    )
    return any(term in host for term in blocked)


def host_from_url(url: str) -> str:
    m = re.match(r"^https?://([^/]+)", normalize_url(url))
    return m.group(1).lower() if m else ""


def is_plausible_url(url: str) -> bool:
    host = host_from_url(url)
    labels = host.split(".")
    if len(labels) < 2:
        return False
    tld = labels[-1]
    root = labels[-2]
    if not tld.isalpha():
        return False
    if tld not in COMMON_TLDS and len(tld) != 2:
        return False
    if len(root) < 2 or len(root) > 30:
        return False
    digit_ratio = sum(ch.isdigit() for ch in root) / max(len(root), 1)
    if digit_ratio > 0.4:
        return False
    return True


def is_weak_social_candidate(url: str, score: float, text: str) -> bool:
    host = host_from_url(url)
    if not any(host.endswith(domain) for domain in SOCIAL_DOMAINS):
        return False
    clean_text = " ".join(text.lower().split())
    has_cta = any(k in clean_text for k in CTA_KEYWORDS)
    # Drop incidental social links unless there is stronger ad evidence.
    return score < 4.2 and not has_cta


def is_generic_email_domain_url(url: str) -> bool:
    host = host_from_url(url)
    return any(host.endswith(domain) for domain in GENERIC_EMAIL_DOMAINS)


def url_is_reachable(url: str, cache: dict) -> bool:
    normalized = normalize_url(url)
    if normalized in cache:
        return cache[normalized]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    methods = ("HEAD", "GET")
    ok = False
    saw_network_error = False
    host = host_from_url(normalized)
    variants = [normalized]
    if host.startswith("www."):
        variants.append(normalized.replace(f"://{host}", f"://{host[4:]}", 1))

    for candidate in variants:
        for method in methods:
            req = urllib.request.Request(candidate, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    status = getattr(resp, "status", 200)
                    final_url = normalize_url(resp.geturl() or candidate)
                    if status < 400 and not is_blocklisted_url(final_url):
                        ok = True
                        break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 405):
                    # Many sites block bots or HEAD; treat as reachable.
                    ok = True
                    break
            except Exception:
                saw_network_error = True
                continue
        if ok:
            break

    # If live check is inconclusive due network/bot blocking, keep plausible domains.
    if not ok and saw_network_error:
        labels = host.split(".")
        if len(labels) >= 2 and len(labels[-1]) >= 2 and not is_blocklisted_url(normalized):
            ok = True

    cache[normalized] = ok
    return ok


def infer_url_from_brand_text(text: str) -> Optional[str]:
    lower = " ".join(text.lower().split())
    if not lower:
        return None

    for brand, url in BRAND_DOMAINS.items():
        if re.search(rf"\b{re.escape(brand)}\b", lower):
            return url

    tokens = re.findall(r"[a-z0-9][a-z0-9&'.-]{2,}", lower)
    if not tokens:
        return None

    best_match = None
    best_score = 0.0
    keys = list(BRAND_DOMAINS.keys())
    for token in tokens:
        match = difflib.get_close_matches(token, keys, n=1, cutoff=0.88)
        if match:
            score = difflib.SequenceMatcher(a=token, b=match[0]).ratio()
            if score > best_score:
                best_score = score
                best_match = match[0]
    if best_match:
        return BRAND_DOMAINS[best_match]
    return None


def infer_url_from_ocr_result(ocr_result, image_height: int) -> Optional[str]:
    candidates = []
    for item in ocr_result or []:
        if len(item) < 2:
            continue
        points = item[0] if isinstance(item[0], (list, tuple)) else None
        text = item[1] if isinstance(item[1], str) else ""
        if not text:
            continue

        for raw in URL_PATTERN.findall(text):
            if "." not in raw:
                continue
            normalized = _canonicalize_with_known_domains(raw)
            score = _score_url_candidate(normalized)
            y_weight = 0.0
            if points and image_height > 0:
                ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                if ys:
                    y_center = sum(ys) / len(ys)
                    y_weight = min(max(y_center / image_height, 0.0), 1.0) * 1.2
            candidates.append((normalized, score + y_weight))

        for raw in DOMAIN_LIKE_PATTERN.findall(text):
            normalized = _canonicalize_with_known_domains(raw)
            score = _score_url_candidate(normalized)
            y_weight = 0.0
            if points and image_height > 0:
                ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                if ys:
                    y_center = sum(ys) / len(ys)
                    y_weight = min(max(y_center / image_height, 0.0), 1.0) * 1.2
            candidates.append((normalized, score + y_weight))

    if not candidates:
        return None
    best_url, best_score = max(candidates, key=lambda x: x[1])
    if best_score < 0.9:
        return None
    return normalize_url(best_url)


def infer_bottom_cta_url(image_np: np.ndarray) -> Optional[str]:
    h = image_np.shape[0]
    if h < 60:
        return None
    bottom = image_np[int(h * 0.7) :, :]
    try:
        bottom_ocr_result, _ = OCR_ENGINE(bottom)
    except Exception:
        return None
    if not bottom_ocr_result:
        return None
    url = infer_url_from_ocr_result(bottom_ocr_result, bottom.shape[0])
    if url:
        return url
    bottom_text = " ".join(item[1] for item in bottom_ocr_result if len(item) >= 2)
    return infer_url(bottom_text)


def infer_top_cta_url(image_np: np.ndarray) -> Optional[str]:
    h = image_np.shape[0]
    if h < 60:
        return None
    top = image_np[: int(h * 0.3), :]
    try:
        top_ocr_result, _ = OCR_ENGINE(top)
    except Exception:
        return None
    if not top_ocr_result:
        return None
    url = infer_url_from_ocr_result(top_ocr_result, top.shape[0])
    if url:
        return url
    top_text = " ".join(item[1] for item in top_ocr_result if len(item) >= 2)
    return infer_url(top_text)


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
    for block_rect, url, _ in url_blocks:
        inter_w = max(0.0, min(rect.x1, block_rect.x1) - max(rect.x0, block_rect.x0))
        overlap_ratio = inter_w / max(min(rect.width, block_rect.width), 1.0)
        if overlap_ratio < 0.25:
            continue

        # Prefer URL text immediately above ad artwork.
        vertical_gap = rect.y0 - block_rect.y1
        if vertical_gap < -10:
            continue
        if vertical_gap > max(140.0, rect.height * 0.35):
            continue

        score = overlap_ratio * 2.0 - (max(vertical_gap, 0.0) / 200.0)
        if score > best_score:
            best_score = score
            best_url = url
    return best_url


def extract_religious_grid_ads(page: fitz.Page, page_index: int) -> List[AdCandidate]:
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1)
    image_rects: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        image_rects.extend(page.get_image_rects(img[0]))
    if not image_rects:
        return []

    # Look for a large bottom panel that usually contains a 12-ad religious grid.
    bottom_panel = None
    for rect in sorted(image_rects, key=lambda r: r.width * r.height, reverse=True):
        area_ratio = (rect.width * rect.height) / page_area
        y_ratio = rect.y0 / max(page_rect.height, 1)
        if area_ratio > 0.22 and y_ratio > 0.45:
            bottom_panel = rect
            break
    if bottom_panel is None:
        return []

    cols, rows = 4, 3
    cell_w = bottom_panel.width / cols
    cell_h = bottom_panel.height / rows
    candidates: List[AdCandidate] = []
    for r in range(rows):
        for c in range(cols):
            cell = fitz.Rect(
                bottom_panel.x0 + c * cell_w,
                bottom_panel.y0 + r * cell_h,
                bottom_panel.x0 + (c + 1) * cell_w,
                bottom_panel.y0 + (r + 1) * cell_h,
            )
            pix = page.get_pixmap(clip=cell, dpi=280, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            image_np = np.array(image)
            try:
                ocr_result, _ = OCR_ENGINE(image_np)
            except Exception:
                continue
            if not ocr_result:
                continue
            ocr_text = " ".join(item[1] for item in ocr_result if len(item) >= 2)
            inferred = infer_url_from_ocr_result(ocr_result, image_np.shape[0])
            if not inferred:
                inferred = infer_url(ocr_text)
            if not inferred:
                inferred = infer_url_from_brand_text(ocr_text)
            if not inferred or is_blocklisted_url(inferred):
                continue

            candidates.append(
                AdCandidate(
                    page_index=page_index,
                    rect=cell,
                    score=4.6,
                    text="Religious grid URL detected",
                    inferred_url=inferred,
                )
            )
    return candidates


def score_block(text: str, rect: fitz.Rect, page_rect: fitz.Rect) -> float:
    clean_text = " ".join(text.split()).lower()
    area_ratio = (rect.width * rect.height) / max(page_rect.width * page_rect.height, 1)
    has_url = bool(infer_url(clean_text))
    keyword_hits = sum(1 for kw in AD_KEYWORDS if kw in clean_text)
    text_density = len(clean_text) / max(rect.width * rect.height, 1)

    score = 0.0
    if area_ratio > 0.03:
        score += 1.0
    if area_ratio > 0.08:
        score += 1.0
    if has_url:
        score += 2.0
    score += min(keyword_hits * 0.75, 2.0)
    if text_density < 0.002:
        score += 0.5
    return score


def is_likely_text_ad(text: str, rect: fitz.Rect, page_rect: fitz.Rect) -> bool:
    clean_text = " ".join(text.split()).lower()
    area_ratio = (rect.width * rect.height) / max(page_rect.width * page_rect.height, 1)
    keyword_hits = sum(1 for kw in AD_KEYWORDS if kw in clean_text)
    cta_hits = sum(1 for kw in CTA_KEYWORDS if kw in clean_text)

    # Filter out editorial/news snippets that happen to contain a URL.
    if area_ratio < 0.04 and cta_hits == 0 and keyword_hits < 2:
        return False
    if "news" in clean_text and "thursday" in clean_text and area_ratio < 0.08:
        return False

    return area_ratio >= 0.12 or keyword_hits >= 2 or (cta_hits >= 1 and area_ratio >= 0.04)


def matches_ad_size(area_ratio: float) -> bool:
    if area_ratio < 0.03:
        return False
    return any(abs(area_ratio - target) <= tol for target, tol in AD_SIZE_TARGETS)


def is_classified_or_notice_page(page: fitz.Page) -> bool:
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1)
    blocks = page.get_text("blocks")
    merged_text = " ".join((b[4] or "") for b in blocks).lower()
    strong_keywords = (
        "classified",
        "legal notices",
        "public notices",
        "order to show cause",
        "change of name",
        "fictitious business",
    )
    has_strong_keyword = any(k in merged_text for k in strong_keywords)
    has_legal_public_combo = ("legal notices" in merged_text and "public notices" in merged_text)

    small_blocks = 0
    large_blocks = 0
    for b in blocks:
        x0, y0, x1, y1, txt, *_ = b
        if not txt or len(txt.strip()) < 20:
            continue
        ratio = ((x1 - x0) * (y1 - y0)) / page_area
        if ratio < 0.03:
            small_blocks += 1
        if ratio > 0.12:
            large_blocks += 1

    image_rects: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        image_rects.extend(page.get_image_rects(img[0]))
    large_image_count = sum(
        1 for r in image_rects if ((r.width * r.height) / page_area) > 0.08
    )

    if has_strong_keyword and small_blocks >= 20 and large_image_count == 0:
        return True
    if has_strong_keyword and small_blocks >= 30 and large_blocks == 0:
        return True
    if has_legal_public_combo and small_blocks >= 18:
        return True
    return False


def is_cartoon_page(page: fitz.Page) -> bool:
    text = page.get_text("text").lower()
    if "cartoon:" in text:
        return True
    if "cartoon" in text and "david g. brown" in text:
        return True
    return False


def is_likely_editorial_page(page: fitz.Page) -> bool:
    text = page.get_text("text").lower()
    if not text:
        return False
    editorial_markers = (
        "contributing writer",
        "opinion",
        "entertainment",
        "by ",
    )
    has_editorial_marker = any(marker in text for marker in editorial_markers)
    ad_marker_count = sum(1 for k in AD_KEYWORDS + CTA_KEYWORDS if k in text)

    blocks = page.get_text("blocks")
    dense_text_blocks = sum(1 for b in blocks if len((b[4] or "").strip()) > 100)
    image_rects: List[fitz.Rect] = []
    for img in page.get_images(full=True):
        image_rects.extend(page.get_image_rects(img[0]))
    page_area = max(page.rect.width * page.rect.height, 1)
    ad_sized_images = sum(
        1
        for r in image_rects
        if matches_ad_size((r.width * r.height) / page_area)
    )

    return has_editorial_marker and dense_text_blocks >= 8 and ad_marker_count < 2 and ad_sized_images == 0


def find_ad_candidates(doc: fitz.Document) -> List[AdCandidate]:
    candidates: List[AdCandidate] = []
    for page_index, page in enumerate(doc):
        if is_cartoon_page(page):
            continue
        if is_classified_or_notice_page(page):
            continue
        if is_likely_editorial_page(page):
            continue
        page_rect = page.rect
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if len(text) < 8:
                continue
            rect = fitz.Rect(x0, y0, x1, y1)
            score = score_block(text, rect, page_rect)
            inferred = infer_url(text)
            area_ratio = (rect.width * rect.height) / max(page_rect.width * page_rect.height, 1)
            if (
                score >= 2.5
                and inferred
                and not is_blocklisted_url(inferred)
                and is_likely_text_ad(text, rect, page_rect)
                and matches_ad_size(area_ratio)
            ):
                candidates.append(
                    AdCandidate(
                        page_index=page_index,
                        rect=rect,
                        score=score,
                        text=text,
                        inferred_url=inferred,
                    )
                )

    return dedupe_overlapping_candidates(candidates)


def rect_iou(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    inter_area = inter.width * inter.height
    union_area = (a.width * a.height) + (b.width * b.height) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def dedupe_overlapping_candidates(candidates: List[AdCandidate]) -> List[AdCandidate]:
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    selected: List[AdCandidate] = []
    for cand in candidates:
        overlaps = any(
            (cand.page_index == chosen.page_index and rect_iou(cand.rect, chosen.rect) > 0.4)
            for chosen in selected
        )
        if not overlaps:
            selected.append(cand)
    return sorted(selected, key=lambda c: (c.page_index, c.rect.y0, c.rect.x0))


def _decode_qr_with_zxing(image: np.ndarray) -> Optional[str]:
    try:
        codes = zxingcpp.read_barcodes(image)
    except Exception:
        return None
    for code in codes:
        text = str(code.text).strip()
        if not text:
            continue
        inferred = infer_url(text)
        if inferred:
            return inferred
    return None


def infer_url_from_qr(image_np: np.ndarray) -> Optional[str]:
    direct = _decode_qr_with_zxing(image_np)
    if direct:
        return direct

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.equalizeHist(gray)
    bin_img = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 6
    )

    for variant in (gray, enhanced, bin_img):
        zxing_found = _decode_qr_with_zxing(variant)
        if zxing_found:
            return zxing_found

    # Scale up for very small QRs inside full-page ads.
    for scale in (1.5, 2.0, 3.0):
        resized = cv2.resize(
            enhanced,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        zxing_found = _decode_qr_with_zxing(resized)
        if zxing_found:
            return zxing_found

    # Tile scan catches QRs in a small corner of a large ad.
    h, w = enhanced.shape[:2]
    step_y = max(h // 3, 1)
    step_x = max(w // 3, 1)
    tile_h = max(int(h * 0.45), 1)
    tile_w = max(int(w * 0.45), 1)
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            y2 = min(y + tile_h, h)
            x2 = min(x + tile_w, w)
            tile = enhanced[y:y2, x:x2]
            if tile.size < 100:
                continue
            tile_up = cv2.resize(tile, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            zxing_found = _decode_qr_with_zxing(tile_up)
            if zxing_found:
                return zxing_found
    return None


def extract_image_ads_with_ocr(doc: fitz.Document) -> List[AdCandidate]:
    candidates: List[AdCandidate] = []
    for page_index, page in enumerate(doc):
        if is_cartoon_page(page):
            continue
        if is_classified_or_notice_page(page):
            continue
        grid_candidates = extract_religious_grid_ads(page, page_index)
        if grid_candidates:
            candidates.extend(grid_candidates)
            # For this page layout, rely on the 12-slot religious grid pass only.
            continue
        page_rect = page.rect
        page_area = max(page_rect.width * page_rect.height, 1)
        url_blocks = get_page_url_text_blocks(page)
        image_rects: List[fitz.Rect] = []
        for img in page.get_images(full=True):
            xref = img[0]
            for rect in page.get_image_rects(xref):
                image_rects.append(rect)

        if not image_rects:
            continue

        # Deduplicate repeated image placements before OCR.
        unique_rects: List[fitz.Rect] = []
        image_rects = sorted(image_rects, key=lambda r: r.width * r.height, reverse=True)
        for rect in image_rects:
            if any(rect_iou(rect, existing) > 0.9 for existing in unique_rects):
                continue
            unique_rects.append(rect)

        for rect in unique_rects:
            area_ratio = (rect.width * rect.height) / page_area
            nearby_text_url = infer_url_from_nearby_text(rect, url_blocks)
            if not matches_ad_size(area_ratio) and not nearby_text_url:
                continue

            # Render the ad rect and a slightly expanded rect to catch tiny URL text above ads.
            pix = page.get_pixmap(clip=rect, dpi=260, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            image_np = np.array(image)
            scan_rect = fitz.Rect(
                rect.x0,
                max(page_rect.y0, rect.y0 - rect.height * 0.25),
                rect.x1,
                min(page_rect.y1, rect.y1 + rect.height * 0.05),
            )
            scan_pix = page.get_pixmap(clip=scan_rect, dpi=280, alpha=False)
            scan_image = Image.frombytes("RGB", (scan_pix.width, scan_pix.height), scan_pix.samples)
            scan_np = np.array(scan_image)

            if nearby_text_url:
                candidates.append(
                    AdCandidate(
                        page_index=page_index,
                        rect=rect,
                        score=4.85 + min(area_ratio * 6, 1.5),
                        text="Nearby URL text detected",
                        inferred_url=nearby_text_url,
                    )
                )
                continue

            qr_url = infer_url_from_qr(scan_np)
            if qr_url and not is_blocklisted_url(qr_url):
                candidates.append(
                    AdCandidate(
                        page_index=page_index,
                        rect=rect,
                        score=5.0 + min(area_ratio * 6, 1.5),
                        text="QR code detected",
                        inferred_url=qr_url,
                    )
                )
                continue

            top_url = infer_top_cta_url(scan_np)
            if top_url and not is_blocklisted_url(top_url):
                candidates.append(
                    AdCandidate(
                        page_index=page_index,
                        rect=rect,
                        score=4.9 + min(area_ratio * 6, 1.5),
                        text="Top CTA URL detected",
                        inferred_url=top_url,
                    )
                )
                continue

            bottom_url = infer_bottom_cta_url(image_np)
            if bottom_url and not is_blocklisted_url(bottom_url):
                candidates.append(
                    AdCandidate(
                        page_index=page_index,
                        rect=rect,
                        score=4.8 + min(area_ratio * 6, 1.5),
                        text="Bottom CTA URL detected",
                        inferred_url=bottom_url,
                    )
                )
                continue
            try:
                ocr_result, _ = OCR_ENGINE(image_np)
            except Exception:
                continue
            if not ocr_result:
                continue

            ocr_text = " ".join(item[1] for item in ocr_result if len(item) >= 2)
            inferred = infer_url_from_ocr_result(ocr_result, image_np.shape[0])
            if not inferred:
                inferred = infer_url(ocr_text)
            if not inferred:
                inferred = infer_url_from_brand_text(ocr_text)
            if not inferred or is_blocklisted_url(inferred):
                continue

            candidates.append(
                AdCandidate(
                    page_index=page_index,
                    rect=rect,
                    score=3.6 + min(area_ratio * 6, 1.5),
                    text=ocr_text,
                    inferred_url=inferred,
                )
            )
    return candidates


def add_links_to_pdf(pdf_bytes: bytes) -> Tuple[bytes, List[dict]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_candidates = find_ad_candidates(doc)
    image_ocr_candidates = extract_image_ads_with_ocr(doc)
    candidates = dedupe_overlapping_candidates(text_candidates + image_ocr_candidates)
    results = []
    url_validation_cache = {}

    for idx, cand in enumerate(candidates, start=1):
        page = doc[cand.page_index]
        url = cand.inferred_url
        if not url:
            continue
        if not is_plausible_url(url):
            continue
        if is_weak_social_candidate(url, cand.score, cand.text):
            continue
        if is_generic_email_domain_url(url):
            continue
        if not url_is_reachable(url, url_validation_cache):
            continue
        page.insert_link(
            {
                "kind": fitz.LINK_URI,
                "from": cand.rect,
                "uri": url,
            }
        )
        results.append(
            {
                "id": idx,
                "page": cand.page_index + 1,
                "url": url,
                "score": round(cand.score, 2),
                "preview_text": " ".join(cand.text.split())[:140],
            }
        )

    out = io.BytesIO()
    doc.save(out, garbage=4, deflate=True)
    doc.close()
    return out.getvalue(), results


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process_pdf():
    upload = request.files.get("pdf")
    if upload is None or not upload.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400

    data = upload.read()
    if not data:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        linked_pdf, links = add_links_to_pdf(data)
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        return jsonify({"error": f"Failed to process PDF: {exc}"}), 400

    encoded_pdf = base64.b64encode(linked_pdf).decode("ascii")
    return jsonify(
        {
            "links": links,
            "linked_pdf_base64": encoded_pdf,
            "filename": f"linked_{upload.filename}",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
