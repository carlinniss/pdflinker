# PDF Ad Linker (Vercel + Python)

This app uploads an e-paper PDF, detects likely ad blocks, infers destination URLs from page text, QR codes, OCR on image ads, and brand/logo text mapping, and writes clickable link annotations over the full ad area.

## Local run

1. Create and activate a virtualenv.
2. Install deps:

   ```bash
   pip install -r requirements.txt
   ```

3. Run:

   ```bash
   python api/index.py
   ```

4. Open http://127.0.0.1:5000

## Deploy to Vercel

1. Install Vercel CLI and login.
2. From this folder:

   ```bash
   vercel
   ```

3. Vercel uses `vercel.json` and deploys `api/index.py` via `@vercel/python`.

## Notes on detection quality

- Current ad detection is heuristic (block size + ad keywords + URL/domain extraction).
- Ad candidate sizing now follows newspaper-style fractions (full, 3/4, 2/3, 1/2, 1/3, 1/4, 1/8) with tolerance for layout gutters.
- Classified/legal-notice pages are suppressed using page-level signals (dense small text + notice keywords).
- Pages that match legal/public notice structure are skipped entirely (no URL extraction).
- Religious-page bottom ad grids are scanned as 12 slots (4x3), and only slots with a valid URL are linked.
- Image ads are checked for QR codes first, and QR-decoded URLs are used when present.
- QR decoding now uses `zxing-cpp` across multiple image variants/scales/tiles for better small-QR recovery.
- If an ad image has a URL line above it, nearby page text URL extraction is used as a fallback.
- Image-heavy ads are scanned with OCR (`rapidocr-onnxruntime`) to detect URLs from rendered ad images.
- When OCR text contains a recognizable brand but no explicit URL, the app maps it using `brand_domains.json`.
- URLs are validated for reachability before annotation; unreachable URLs are skipped.
- Weak social-only links (for example incidental `facebook.com` references without clear ad CTA intent) are filtered out.
- Generic email-host domains (for example `gmail.com`) are not linked as ad destinations.
- OCR and URL inference are best-effort, so some ads may still need manual review.

## Brand/logo mapping

- Edit `brand_domains.json` to add brands that commonly appear in your e-paper ads.
- Format: `"brand text": "https://destination-url"`.
- The system checks direct matches and close spelling matches from OCR output.
