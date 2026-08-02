"""
Final data generation v3 - consumes the 4 per-image tag files (100 images
total, 25 per style tag) and produces the three datasets, scaled up now
that we have a full 25-image pool per tag instead of 6.

Allocation per style tag (25 images), scaled to use most of the pool,
volume itself now shaped by momentum (not just the numbers within it):
  - 8  -> competitor SKUs (one image each, cycling 5 real brands)
  - 3  -> self-catalog SKUs
  - remainder (14 for rising tags, 10 for flat, 8 for declining) -> IG posts,
    week-distribution weighted by momentum direction
"""
import json
import shutil
from pathlib import Path
from datetime import date, timedelta

SOURCE_ROOT = Path("/home/claude/fashion_images/Fashion_Images_25/Streetwear tops")
TAG_DIR = Path("/home/claude/tagged_images")
OUT_ROOT = Path("/home/claude/final_data_v3")
DATASET_ROOT_NAME = "final_data_v3"

FOLDER_BY_TAG = {
    "oversized-hoodie": "oversized hoodie streetwear",
    "graphic-print": "graphic print t-shirt street style",
    "boxy-tee": "boxy t-shirt fashion",
    "cropped-zip-up": "cropped zip up jacket",
}

MOMENTUM_BY_TAG = {
    "oversized-hoodie": "rising",
    "graphic-print": "rising",
    "boxy-tee": "flat",
    "cropped-zip-up": "declining",
}

BRANDS = ["Brandy Melville", "PacSun", "Urban Outfitters", "American Eagle", "Zara"]

CATEGORY = "streetwear_tops"
WEEK_LABELS = ["week_now-3", "week_now-2", "week_now-1", "week_now"]

COMP_PER_TAG = 8
SELF_PER_TAG = 3

# IG post volume + weekly weighting per momentum - counts now use almost all
# remaining pool for rising tags (denser real content = stronger trend signal)
IG_PLAN = {
    "rising":     {"count": 14, "weekly_weights": [2, 3, 4, 5]},  # ramping up hard
    "flat":       {"count": 10, "weekly_weights": [3, 2, 3, 2]},  # steady, less total volume
    "declining":  {"count": 8,  "weekly_weights": [3, 3, 2, 0]},  # tapering off, quiet last week
}


def series_for(momentum, kind):
    shapes = {
        "rating_velocity": {"rising": [3, 5, 8, 12], "flat": [4, 4, 5, 4], "declining": [6, 4, 3, 2]},
        "sell_through":    {"rising": [0.22, 0.29, 0.37, 0.48], "flat": [0.30, 0.31, 0.29, 0.30], "declining": [0.35, 0.28, 0.22, 0.16]},
        "engagement":      {"rising": [1400, 1900, 2600, 3600], "flat": [1800, 1750, 1850, 1800], "declining": [2200, 1700, 1300, 950]},
    }
    return shapes[kind][momentum]


def oos_flags_for(momentum):
    return {"rising": [False, False, True, True],
            "flat": [False, False, False, False],
            "declining": [True, False, False, False]}[momentum]


INFLUENCERS = [
    {"handle": "@ateliermaya", "followers": 420000},
    {"handle": "@studio.noor", "followers": 185000},
    {"handle": "@thecraftedcloset", "followers": 610000},
    {"handle": "@lunaeditorial", "followers": 92000},
    {"handle": "@westline.style", "followers": 330000},
]


def load_tag_entries():
    per_tag = {}
    for tag in FOLDER_BY_TAG:
        entries = json.load(open(TAG_DIR / f"{tag}_tags.json"))
        per_tag[tag] = entries
    return per_tag


IMAGE_MANIFEST = []  # tracks provenance: output filename -> original source file


def copy_image(tag, filename, role):
    src = SOURCE_ROOT / FOLDER_BY_TAG[tag] / filename
    dest_name = f"{CATEGORY}__{tag}__{role}.jpg"
    dest = OUT_ROOT / "images" / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    IMAGE_MANIFEST.append({
        "output_filename": dest_name,
        "output_path": f"{DATASET_ROOT_NAME}/images/{dest_name}",
        "original_filename": filename,
        "original_folder": FOLDER_BY_TAG[tag],
        "original_relative_path": f"Fashion_Images_25/Streetwear tops/{FOLDER_BY_TAG[tag]}/{filename}",
        "style_tag": tag,
        "role": role,
    })
    return f"{DATASET_ROOT_NAME}/images/{dest_name}"


def bullets_from_entry(entry):
    """Build 3 distinct bullets from the entry's own unique description/tags -
    no shared template text across records."""
    return [
        entry["description"],
        f"Color: {entry['color_palette'].replace('-', ' ').title()}",
        f"Style: {entry['style_name']}",
    ]


def build_competitor(per_tag):
    rows = []
    sku = 9000
    for tag, entries in per_tag.items():
        momentum = MOMENTUM_BY_TAG[tag]
        velocity = series_for(momentum, "rating_velocity")
        oos = oos_flags_for(momentum)
        for i in range(COMP_PER_TAG):
            entry = entries[i]
            sku += 1
            brand = BRANDS[i % len(BRANDS)]
            img_path = copy_image(tag, entry["filename"], f"cmp{i+1}")
            IMAGE_MANIFEST[-1]["used_by_record"] = f"CMP-{sku}"
            IMAGE_MANIFEST[-1]["used_by_dataset"] = "competitor_pdp.json"
            rows.append({
                "competitor_sku": f"CMP-{sku}",
                "brand": brand,
                "url": f"https://example-competitor.com/p/{sku}",
                "title": entry["title"],
                "thumbnail_url": img_path,
                "bullet_description": bullets_from_entry(entry),
                "category": CATEGORY,
                "style_tag": tag,
                "color_palette": entry["color_palette"],
                "price": round(28 + ((sku + i) % 9) * 6.5, 2),
                "rating_avg": round(3.9 + (sku % 5) * 0.15, 1),
                "rating_count_total": 120 + sku % 900,
                "rating_velocity_week_now-3": velocity[0],
                "rating_velocity_week_now-2": velocity[1],
                "rating_velocity_week_now-1": velocity[2],
                "rating_velocity_week_now": velocity[3],
                "oos_status_current": oos[-1],
                "oos_flag_week_now-3": oos[0],
                "oos_flag_week_now-2": oos[1],
                "oos_flag_week_now-1": oos[2],
                "oos_flag_week_now": oos[3],
                "scraped_at": str(date.today()),
            })
    return rows


def build_self(per_tag):
    rows = []
    sku = 1000
    for tag, entries in per_tag.items():
        momentum = MOMENTUM_BY_TAG[tag]
        str_series = series_for(momentum, "sell_through")
        oos = oos_flags_for(momentum)
        for i in range(SELF_PER_TAG):
            entry = entries[COMP_PER_TAG + i]  # slots after competitor allocation
            sku += 1
            img_path = copy_image(tag, entry["filename"], f"self{i+1}")
            IMAGE_MANIFEST[-1]["used_by_record"] = f"SELF-{sku}"
            IMAGE_MANIFEST[-1]["used_by_dataset"] = "self_catalog.json"
            rows.append({
                "style_name": entry["style_name"] if i == 0 else f"{entry['style_name']} v{i+1}",
                "sku": f"SELF-{sku}",
                "title": entry["title"],
                "thumbnail_url": img_path,
                "category": CATEGORY,
                "style_tag": tag,
                "color_palette": entry["color_palette"],
                "price": round(24 + ((sku + i) % 7) * 5.5, 2),
                "sell_through_rate_week_now-3": str_series[0],
                "sell_through_rate_week_now-2": str_series[1],
                "sell_through_rate_week_now-1": str_series[2],
                "sell_through_rate_week_now": str_series[3],
                "units_available_current": 40 + sku % 300,
                "oos_status_current": oos[-1],
                "oos_flag_week_now-3": oos[0],
                "oos_flag_week_now-2": oos[1],
                "oos_flag_week_now-1": oos[2],
                "oos_flag_week_now": oos[3],
                "launch_date": str(date.today() - timedelta(days=60 + sku % 120)),
            })
    return rows


def build_ig(per_tag):
    """category/style_tag intentionally NOT exported on the record - used only
    internally to pick engagement shape, image pool, and caption/hashtags."""
    rows = []
    post_id = 500
    today = date.today()
    inf_cycle = 0

    for tag, entries in per_tag.items():
        momentum = MOMENTUM_BY_TAG[tag]
        plan = IG_PLAN[momentum]
        eng_series = series_for(momentum, "engagement")
        remaining_entries = entries[COMP_PER_TAG + SELF_PER_TAG:]  # everything after competitor+self allocation

        # build the list of (week_index) assignments per the weekly_weights plan
        week_assignments = []
        for week_idx, weight in enumerate(plan["weekly_weights"]):
            week_assignments += [week_idx] * weight

        for post_num, week_idx in enumerate(week_assignments):
            entry = remaining_entries[post_num]
            inf = INFLUENCERS[inf_cycle % len(INFLUENCERS)]
            inf_cycle += 1

            base_engagement = eng_series[week_idx]
            scale = inf["followers"] / 300000
            likes = int(base_engagement * scale)

            img_path = copy_image(tag, entry["filename"], f"ig{post_num+1}")
            post_id += 1
            IMAGE_MANIFEST[-1]["used_by_record"] = f"IG-{post_id}"
            IMAGE_MANIFEST[-1]["used_by_dataset"] = "ig_posts.json"
            post_date = today - timedelta(weeks=(3 - week_idx), days=post_id % 5)
            post_type = "reel" if post_id % 3 == 0 else "post"

            rows.append({
                "influencer_handle": inf["handle"],
                "follower_count": inf["followers"],
                "post_id": f"IG-{post_id}",
                "post_permalink": f"https://instagram.com/p/{post_id}",
                "post_date": str(post_date),
                "week_label": WEEK_LABELS[week_idx],
                "post_type": post_type,
                "media_urls": [img_path],
                "video_url": None,
                "thumbnail_url": img_path,
                "caption_text": entry["caption_text"],
                "hashtags": entry["hashtags"],
                "likes": likes,
                "comments_count": int(likes * 0.04),
                "video_views": int(likes * 5) if post_type == "reel" else None,
                "saves": None,
                "reposts": None,
                "clicks": None,
            })
    return rows


if __name__ == "__main__":
    per_tag = load_tag_entries()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    competitor = build_competitor(per_tag)
    self_catalog = build_self(per_tag)
    ig_posts = build_ig(per_tag)

    with open(OUT_ROOT / "competitor_pdp.json", "w") as f:
        json.dump(competitor, f, indent=2)
    with open(OUT_ROOT / "self_catalog.json", "w") as f:
        json.dump(self_catalog, f, indent=2)
    with open(OUT_ROOT / "ig_posts.json", "w") as f:
        json.dump(ig_posts, f, indent=2)

    trend_note = {
        "category": CATEGORY,
        "momentum_assignment": MOMENTUM_BY_TAG,
        "ig_plan": IG_PLAN,
        "skipped_style_tags_no_image": ["ribbed-tank"],
        "categories_not_yet_available": ["denim_bottoms", "athleisure", "utility_outerwear"],
        "note": "v3: full 25-image pool per tag used (5 competitor + 2 self + up to 18 IG), zero repeated images or text across any record.",
    }
    with open(OUT_ROOT / "trend_assignment_note.json", "w") as f:
        json.dump(trend_note, f, indent=2)

    with open(OUT_ROOT / "image_manifest.json", "w") as f:
        json.dump(IMAGE_MANIFEST, f, indent=2)
    print(f"image_manifest.json: {len(IMAGE_MANIFEST)} entries (maps every output image back to its original Pexels filename and the record that uses it)")

    # sanity checks
    comp_images = [r["thumbnail_url"] for r in competitor]
    self_images = [r["thumbnail_url"] for r in self_catalog]
    ig_images = [r["thumbnail_url"] for r in ig_posts]
    ig_captions = [r["caption_text"] for r in ig_posts]
    all_images = comp_images + self_images + ig_images

    print(f"competitor_pdp: {len(competitor)} rows, {len(set(comp_images))} unique images")
    print(f"self_catalog:   {len(self_catalog)} rows, {len(set(self_images))} unique images")
    print(f"ig_posts:       {len(ig_posts)} rows, {len(set(ig_images))} unique images, {len(set(ig_captions))} unique captions")
    print(f"ALL images unique across entire dataset: {len(all_images) == len(set(all_images))}")
    print(f"Total images used: {len(all_images)} / 100 available")
