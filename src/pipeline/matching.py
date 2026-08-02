"""
Step 1 - Matching. Derives category/style_tag for records that don't
natively carry them, and passes through records that already do.

Competitor/self records already carry category + style_tag -> pass through.
IG posts do NOT carry them by design (raw, undecorated) -> must be derived.

For this mock dataset, image_manifest.json records the ground-truth
style_tag assigned to every image at generation time, so IG matching is a
direct join against the manifest (tag_source="manifest_ground_truth").

keyword_match() and vision_match() below are the production path for real
(non-mock) IG data, where no manifest ground truth exists. They are kept
correct and ready to swap in but are not called in match_ig_posts() while
manifest lookups succeed - see run_matching()'s use_fallback flag.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "final_data_v3"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"

# Full taxonomy for all 4 categories. Only streetwear_tops has data today;
# the other 3 stay in this map with zero data so adding real records to
# them later requires no code change here.
TAG_TO_CATEGORY = {
    # streetwear_tops
    "oversized-hoodie": "streetwear_tops",
    "boxy-tee": "streetwear_tops",
    "cropped-zip-up": "streetwear_tops",
    "graphic-print": "streetwear_tops",
    "ribbed-tank": "streetwear_tops",
    # denim_bottoms
    "low-rise": "denim_bottoms",
    "cargo-pant": "denim_bottoms",
    "baggy": "denim_bottoms",
    "wide-leg": "denim_bottoms",
    "raw-hem": "denim_bottoms",
    # athleisure
    "bike-short": "athleisure",
    "tech-jogger": "athleisure",
    "crop-top": "athleisure",
    "track-pant": "athleisure",
    "co-ord-set": "athleisure",
    # utility_outerwear
    "multi-pocket-jacket": "utility_outerwear",
    "techwear-shell": "utility_outerwear",
    "quilted": "utility_outerwear",
    "oversized-parka": "utility_outerwear",
    "boxy-jacket": "utility_outerwear",
}

# Keyword sets for the production keyword-match path (caption_text + hashtags).
KEYWORDS_BY_TAG = {
    "oversized-hoodie": ["oversized hoodie", "hoodie", "pullover", "drop-shoulder", "drop shoulder"],
    "boxy-tee": ["boxy tee", "boxy t-shirt", "boxy", "tee", "t-shirt", "tshirt"],
    "cropped-zip-up": ["cropped zip", "zip-up", "zip up", "zipup"],
    "graphic-print": ["graphic tee", "graphic print", "graphic"],
    "ribbed-tank": ["ribbed tank", "tank top", "ribbed", "tank"],
    "low-rise": ["low-rise", "low rise"],
    "cargo-pant": ["cargo pant", "cargo pants", "cargo"],
    "baggy": ["baggy jeans", "baggy"],
    "wide-leg": ["wide-leg", "wide leg"],
    "raw-hem": ["raw hem", "raw-hem", "rawhem"],
    "bike-short": ["bike short", "bike shorts"],
    "tech-jogger": ["tech jogger", "jogger"],
    "crop-top": ["crop top", "cropped top"],
    "track-pant": ["track pant", "track pants"],
    "co-ord-set": ["co-ord", "co ord", "coord set", "matching set"],
    "multi-pocket-jacket": ["multi-pocket", "multi pocket", "cargo jacket"],
    "techwear-shell": ["techwear", "tech shell", "shell jacket"],
    "quilted": ["quilted jacket", "quilted", "puffer"],
    "oversized-parka": ["oversized parka", "parka"],
    "boxy-jacket": ["boxy jacket"],
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def build_manifest_lookup(manifest: list, dataset_name: str = "ig_posts.json") -> dict:
    """post_id -> manifest entry, restricted to entries used by the given dataset."""
    return {
        entry["used_by_record"]: entry
        for entry in manifest
        if entry["used_by_dataset"] == dataset_name
    }


def keyword_match(post: dict) -> dict | None:
    """
    Production path for real (non-mock) IG data: guess style_tag from
    caption_text + hashtags. Returns None if no keyword hits.
    """
    haystack = " ".join([post.get("caption_text") or ""] + (post.get("hashtags") or [])).lower()

    best_tag, best_len = None, 0
    for tag, keywords in KEYWORDS_BY_TAG.items():
        for kw in keywords:
            if kw in haystack and len(kw) > best_len:
                best_tag, best_len = tag, len(kw)

    if best_tag is None:
        return None

    return {
        "category": TAG_TO_CATEGORY[best_tag],
        "style_tag": best_tag,
        "tag_source": "keyword_match",
        "tag_confidence": "medium",
    }


def vision_match(post: dict, bedrock_client=None) -> dict | None:
    """
    Production fallback for real IG data when keyword_match() finds nothing:
    a Claude vision call over the post's image to classify category/style_tag
    against the full taxonomy. Not exercised in this run (manifest ground
    truth resolves 100% of this mock dataset), but kept correct so it can be
    wired in without touching match_ig_posts().
    """
    import base64
    import requests

    from src.utils.aws_cred import call_cred
    from src.utils.chat_prompt_loader import safe_json_parse

    image_url = post.get("thumbnail_url") or (post.get("media_urls") or [None])[0]
    if not image_url:
        logger.warning(f"vision_match: no image available for post_id={post.get('post_id')}")
        return None

    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception as e:
        logger.warning(f"vision_match: failed to fetch image for post_id={post.get('post_id')}: {e}")
        return None

    client = bedrock_client or call_cred()
    taxonomy = {cat: [] for cat in set(TAG_TO_CATEGORY.values())}
    for tag, cat in TAG_TO_CATEGORY.items():
        taxonomy[cat].append(tag)

    prompt = (
        "Classify this fashion image against the taxonomy below. "
        "Respond with ONLY a JSON object: {\"category\": ..., \"style_tag\": ...}.\n\n"
        f"Taxonomy: {json.dumps(taxonomy)}"
    )

    try:
        response = client.converse(
            modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                    {"text": prompt},
                ],
            }],
        )
        text = response["output"]["message"]["content"][0]["text"]
        parsed = safe_json_parse(None, text, context=f"vision_match:{post.get('post_id')}")
        if not parsed or "category" not in parsed or "style_tag" not in parsed:
            return None
        return {
            "category": parsed["category"],
            "style_tag": parsed["style_tag"],
            "tag_source": "vision_match",
            "tag_confidence": "medium",
        }
    except Exception as e:
        logger.warning(f"vision_match: Bedrock call failed for post_id={post.get('post_id')}: {e}")
        return None


def match_ig_posts(ig_posts: list, manifest: list, use_fallback: bool = False, bedrock_client=None) -> list:
    """
    Join ig_posts on post_id against the manifest's ground-truth style_tag.
    Falls back to keyword_match() then vision_match() only if use_fallback
    is True (real, non-mock data) - never crashes the whole run on one
    unmatched record, just logs it by post_id and moves on.
    """
    lookup = build_manifest_lookup(manifest, dataset_name="ig_posts.json")

    matched, unmatched_ids = [], []
    for post in ig_posts:
        post_id = post["post_id"]
        record = dict(post)

        entry = lookup.get(post_id)
        if entry is not None:
            style_tag = entry["style_tag"]
            record.update({
                "category": TAG_TO_CATEGORY[style_tag],
                "style_tag": style_tag,
                "tag_source": "manifest_ground_truth",
                "tag_confidence": "high",
            })
            matched.append(record)
            continue

        result = None
        if use_fallback:
            result = keyword_match(post)
            if result is None:
                result = vision_match(post, bedrock_client=bedrock_client)

        if result is not None:
            record.update(result)
            matched.append(record)
        else:
            record.update({
                "category": None,
                "style_tag": None,
                "tag_source": "unmatched",
                "tag_confidence": "none",
            })
            unmatched_ids.append(post_id)
            matched.append(record)

    if unmatched_ids:
        logger.warning(f"match_ig_posts: {len(unmatched_ids)} unmatched post_id(s): {unmatched_ids}")

    return matched


def match_passthrough(records: list) -> list:
    """Competitor/self records already carry category + style_tag natively."""
    out = []
    for record in records:
        r = dict(record)
        r["tag_source"] = "raw"
        r["tag_confidence"] = "high"
        out.append(r)
    return out


def run_matching(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR, use_fallback: bool = False) -> dict:
    competitor_pdp = load_json(data_dir / "competitor_pdp.json")
    self_catalog = load_json(data_dir / "self_catalog.json")
    ig_posts = load_json(data_dir / "ig_posts.json")
    manifest = load_json(data_dir / "image_manifest.json")

    ig_posts_matched = match_ig_posts(ig_posts, manifest, use_fallback=use_fallback)
    competitor_matched = match_passthrough(competitor_pdp)
    self_matched = match_passthrough(self_catalog)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ig_posts_matched.json", "w") as f:
        json.dump(ig_posts_matched, f, indent=2)
    with open(output_dir / "competitor_pdp_matched.json", "w") as f:
        json.dump(competitor_matched, f, indent=2)
    with open(output_dir / "self_catalog_matched.json", "w") as f:
        json.dump(self_matched, f, indent=2)

    ig_matched_count = sum(1 for r in ig_posts_matched if r["tag_source"] != "unmatched")
    ig_unmatched_count = len(ig_posts_matched) - ig_matched_count
    style_tag_dist = {}
    for r in ig_posts_matched:
        if r["style_tag"]:
            style_tag_dist[r["style_tag"]] = style_tag_dist.get(r["style_tag"], 0) + 1

    summary = {
        "ig_posts_matched": ig_matched_count,
        "ig_posts_unmatched": ig_unmatched_count,
        "ig_posts_total": len(ig_posts_matched),
        "competitor_records": len(competitor_matched),
        "self_records": len(self_matched),
        "ig_style_tag_distribution": style_tag_dist,
    }
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = run_matching()
    print(json.dumps(summary, indent=2))
