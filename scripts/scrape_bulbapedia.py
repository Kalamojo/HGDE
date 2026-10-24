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
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Optional

POKEAPI_POKEMON_URL = "https://pokeapi.co/api/v2/pokemon/"
WIKI_API_URL = "https://bulbapedia.bulbagarden.net/w/api.php"
BASE_WIKI = "https://bulbapedia.bulbagarden.net/wiki/"
LIST_PAGE = "List of Pokémon by National Pokédex number"

OUT_DIR = Path("data/pokemon_data")
IMG_DIR = OUT_DIR / "images"
METADATA = OUT_DIR / "metadata.jsonl"
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


def wiki_api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(3):
        resp = SESSION.get(WIKI_API_URL, params=params, timeout=20)
        if resp.status_code == 200:
            time.sleep(REQUEST_DELAY)
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MediaWiki API error for params={params}: {data['error']}")
            return data
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()

def poke_api_get(url: str) -> dict:
    resp = SESSION.get(url)
    resp.raise_for_status()

    return resp.json()


# ---------------------------------------------------------------------------
# Step 1: build the master list of (dex_number, name, page_title)
# ---------------------------------------------------------------------------

def build_pokemon_list():
    """Parse the National Pokedex list page's tables to get every species'
    display name, dex number, and *actual wiki page title* (these differ,
    e.g. 'Mr. Mime (Pokémon)')."""
    data = wiki_api_get({"action": "parse", "page": LIST_PAGE, "prop": "text"})
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
            dex_match = re.search(r"#?(\d{3,4})", cells[0].get_text()) # heck naw, gonna replace this regex logic
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
    data = wiki_api_get({"action": "parse", "page": page_title, "prop": "sections"})
    if "error" in data:
        return None
    sections = data["parse"]["sections"]
    target = next((s for s in sections if s["line"].strip() == "Biology"), None)
    if not target:
        return None

    sec_data = wiki_api_get(
        {"action": "parse", "page": page_title, "prop": "text", "section": target["index"]}
    )
    html = sec_data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraph = next(p for p in paragraphs if p)
    return paragraph if paragraph else None


def get_main_image_url(page_title: str, dex_num: str, name: str) -> str | None:
    imgs = wiki_api_get({"action": "query", "titles": page_title, "prop": "images", "imlimit": 50})
    pages = imgs.get("query", {}).get("pages", {})

    # Prefer the canonical "<dexnum><name>.png" artwork file over icons/sprites
    padded = dex_num.zfill(4)
    target_file = f"File:{padded}{name}.png"
    last_candidate = None
    best = None
    for page in pages.values():
        for im in page.get("images", []):
            if im["title"] == target_file:
                best = im["title"]
            if im["title"].startswith(f"File:{padded}"):
                last_candidate = im["title"] # e.g. "File:0001Bulbasaur.png"

    if not best:
        best = last_candidate
    if not best:
        return None

    info = wiki_api_get(
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
            img_url = get_main_image_url(title, dex_num, name)
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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def parse_stats(stats: dict) -> dict:
    flattened = {}
    for stat in stats:
        stat_name = stat["stat"]["name"]
        flattened[stat_name] = stat["base_stat"]

    return flattened

def parse_types(types: dict) -> dict:
    flattened = {
        "primary_type": types[0]["type"]["name"]
    }
    if len(types) > 2:
        raise ValueError("There shouldn't be more than 2 types for any pokemon")
    elif len(types) == 2:
        flattened["secondary_type"] = types[1]["type"]["name"]

    return flattened

def fetch_details(dex_num: str) -> tuple[dict, str, Optional[str]]:
    dex_id = dex_num.lstrip('0')
    pokemon_url = urljoin(POKEAPI_POKEMON_URL, dex_id)
    all_details = poke_api_get(pokemon_url)

    species_url = all_details["species"]["url"]
    cries_url = all_details["cries"]["latest"]

    details_subset = {
        "height": all_details["height"],
        "weight": all_details["weight"],
    }

    details_subset = details_subset | parse_stats(all_details["stats"])
    details_subset = details_subset | parse_types(all_details["types"])

    return details_subset, species_url, cries_url

def fetch_species(species_url: str) -> tuple[dict, str, str]:
    all_species = poke_api_get(species_url)

    evolutions_url = all_species["evolution_chain"]["url"]

    species_subset = {
        "is_baby": all_species["is_baby"],
        "is_legendary": all_species["is_legendary"],
        "is_mythical": all_species["is_mythical"],
        "egg_groups": [group["name"] for group in all_species["egg_groups"]],
        "color": all_species["color"]["name"],
        "shape": all_species["shape"]["name"],
        "habitat": all_species["habitat"]["name"] if all_species["habitat"] else None,
        "generation": all_species["generation"]["name"],
    }

    return species_subset, evolutions_url, all_species["name"]

def find_stage(name, evolution_chain: dict) -> int:
    if name == evolution_chain["species"]["name"]:
        return 0
    if len(evolution_chain["evolves_to"]) == 0:
        return -1

    for evolution in evolution_chain["evolves_to"]:
        stage_depth = find_stage(name, evolution)
        if stage_depth != -1:
            return 1 + stage_depth

    return -1

def fetch_evolutions(name: str, evolutions_url: str) -> dict:
    all_evolutions = poke_api_get(evolutions_url)
    evolution_stage = find_stage(name, all_evolutions["chain"])

    return {"stage": evolution_stage}

def fetch_audio(cries_url: str) -> dict:
    raise NotImplementedError("Still need to implement fetch audio")

def fetch_metadata(get_audio: bool = False, limit: int | None = None):
    entries = json.loads(NAME_LIST.read_text())
    if limit:
        entries = entries[:limit]

    for entry in tqdm(entries):
        dex_num, name = entry["dex_num"], entry["name"]
        try:
            metadata_dict = {
                "dex_num": dex_num,
                "name": name,
            }
            details_dict, species_url, cries_url = fetch_details(dex_num)
            species_dict, evolutions_url, lower_name = fetch_species(species_url)
            evolutions_dict = fetch_evolutions(lower_name, evolutions_url)
            audio_dict = {}
            if get_audio:
                audio_dict = fetch_audio(cries_url)

            #print(type(metadata_dict), type(details_dict), type(species_dict), type(evolutions_dict), type(audio_dict))
            metadata_dict = metadata_dict | details_dict | species_dict | evolutions_dict | audio_dict

            with open(METADATA, "a") as f:
                f.write(
                    json.dumps(
                        metadata_dict,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            print("Metadata fetch failure:", {"dex_num": dex_num, "name": name, "error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["list", "metadata", "scrape"], required=True)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N entries (use this for a test run before the full 1025)."
    )
    args = parser.parse_args()

    if args.step == "list":
        build_pokemon_list()
    elif args.step == "metadata":
        fetch_metadata(limit=args.limit)
    elif args.step == "scrape":
        scrape_all(limit=args.limit)
