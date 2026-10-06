"""
Bulbapedia Pokemon image/text pair scraper.

Builds a dataset of (600x600 image, biology-paragraph text) pairs for all
National Pokedex entries, using the MediaWiki API that powers Bulbapedia
rather than raw HTML scraping wherever possible.

Output layout:
    output/
        images/<dex_num>_<name>.png
        manifest.jsonl        # one JSON object per pokemon, resumable
        failures.jsonl        # anything that errored, for manual review

Run:
    python scrape_bulbapedia.py --step list      # build the master name list
    python scrape_bulbapedia.py --step scrape     # fetch text+images
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from tqdm import tqdm

API_URL = "https://bulbapedia.bulbagarden.net/w/api.php"
BASE_WIKI = "https://bulbapedia.bulbagarden.net/wiki/"
LIST_PAGE = "List of Pokémon by National Pokédex number"

OUT_DIR = Path("output")
IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.jsonl"
FAILURES = OUT_DIR / "failures.jsonl"
NAME_LIST = OUT_DIR / "pokemon_pages.json"

# Bulbapedia (and MediaWiki API etiquette generally) asks for a descriptive
# User-Agent that identifies who/what is making requests and how to reach you.
HEADERS = {
    "User-Agent": "PokedexEmbeddingResearch/0.1 (contact: your_email@example.com)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

REQUEST_DELAY = 0.6  # seconds between API calls -- be polite, tune as needed


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(3):
        resp = SESSION.get(API_URL, params=params, timeout=20)
        if resp.status_code == 200:
            time.sleep(REQUEST_DELAY)
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MediaWiki API error for params={params}: {data['error']}")
            return data
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Step 1: build the master list of (dex_number, name, page_title)
# ---------------------------------------------------------------------------

def build_pokemon_list():
    """Parse the National Pokedex list page's tables to get every species'
    display name, dex number, and *actual wiki page title* (these differ,
    e.g. 'Mr. Mime (Pokémon)')."""
    data = api_get({"action": "parse", "page": LIST_PAGE, "prop": "text"})
    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    entries = []
    seen = set()
    for table in soup.find_all("table", class_="roundy"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            # Find the pokemon name link (skip type icon links etc.)
            name_link = None
            for a in row.find_all("a", href=True):
                href = a["href"]
                if "/wiki/" in href and "(Pok" in href:
                    name_link = a
                    break
            if not name_link:
                continue
            page_title = href_to_title(name_link["href"])
            display_name = name_link.get("title", name_link.text).strip()
            dex_match = re.search(r"#?(\d{3,4})", cells[0].get_text())
            dex_num = dex_match.group(1) if dex_match else None
            key = (dex_num, page_title)
            if dex_num and page_title and key not in seen:
                seen.add(key)
                entries.append(
                    {"dex_num": dex_num, "name": display_name, "page_title": page_title}
                )

    entries.sort(key=lambda e: e["dex_num"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NAME_LIST.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Found {len(entries)} entries -> {NAME_LIST}")


def href_to_title(href: str) -> str:
    title = href.split("/wiki/")[-1]
    from urllib.parse import unquote
    return unquote(title).replace("_", " ")


# ---------------------------------------------------------------------------
# Step 2: per-Pokemon text (Biology section) + image
# ---------------------------------------------------------------------------

def get_biology_text(page_title: str) -> str | None:
    data = api_get({"action": "parse", "page": page_title, "prop": "sections"})
    if "error" in data:
        return None
    sections = data["parse"]["sections"]
    target = next((s for s in sections if s["line"].strip() == "Biology"), None)
    if not target:
        return None

    sec_data = api_get(
        {"action": "parse", "page": page_title, "prop": "text", "section": target["index"]}
    )
    html = sec_data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    return " ".join(paragraphs) if paragraphs else None


def get_main_image_url(page_title: str, dex_num: str) -> str | None:
    imgs = api_get({"action": "query", "titles": page_title, "prop": "images", "imlimit": 50})
    pages = imgs.get("query", {}).get("pages", {})
    candidates = []
    for page in pages.values():
        for im in page.get("images", []):
            candidates.append(im["title"])  # e.g. "File:001Bulbasaur.png"

    # Prefer the canonical "<dexnum><name>.png" artwork file over icons/sprites
    padded = dex_num.zfill(3)
    best = next((c for c in candidates if c.startswith(f"File:{padded}")), None)
    if not best:
        best = next((c for c in candidates if c.lower().endswith(".png")), None)
    if not best:
        return None

    info = api_get(
        {
            "action": "query",
            "titles": best,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 600,
        }
    )
    ipages = info.get("query", {}).get("pages", {})
    for p in ipages.values():
        ii = p.get("imageinfo")
        if ii:
            return ii[0].get("thumburl") or ii[0].get("url")
    return None


def download_and_resize(url: str, out_path: Path, size=(600, 600)):
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()
    from io import BytesIO
    img = Image.open(BytesIO(resp.content)).convert("RGBA")
    # Pad to square instead of stretching, then resize to target
    img = ImageOps.pad(img, size, color=(255, 255, 255, 0))
    # Flatten transparency onto a white background before saving as RGB.
    # Most multimodal embedding models (ImageBind included) expect 3-channel
    # RGB input -- an alpha channel either gets silently dropped or errors
    # depending on the preprocessing pipeline, so flatten explicitly here.
    background = Image.new("RGB", size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    background.save(out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def already_done(dex_num: str) -> bool:
    if not MANIFEST.exists():
        return False
    with open(MANIFEST) as f:
        return any(json.loads(line)["dex_num"] == dex_num for line in f)


def scrape_all(limit: int | None = None):
    entries = json.loads(NAME_LIST.read_text())
    if limit:
        entries = entries[:limit]
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(entries):
        dex_num, name, title = entry["dex_num"], entry["name"], entry["page_title"]
        if already_done(dex_num):
            continue
        try:
            text = get_biology_text(title)
            img_url = get_main_image_url(title, dex_num)
            if not text or not img_url:
                raise ValueError(f"missing text={bool(text)} image={bool(img_url)}")

            img_path = IMG_DIR / f"{dex_num}_{re.sub(r'[^A-Za-z0-9]+', '', name)}.png"
            download_and_resize(img_url, img_path)

            with open(MANIFEST, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "dex_num": dex_num,
                            "name": name,
                            "page_title": title,
                            "image_path": str(img_path),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            with open(FAILURES, "a") as f:
                f.write(json.dumps({"dex_num": dex_num, "title": title, "error": str(e)}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["list", "scrape"], required=True)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N entries (use this for a test run before the full 1025)."
    )
    args = parser.parse_args()

    if args.step == "list":
        build_pokemon_list()
    elif args.step == "scrape":
        scrape_all(limit=args.limit)
