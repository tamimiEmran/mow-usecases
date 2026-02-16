"""Bus stop proximity filter for AV fleet video occurrences.

Each CSV row is an independent event with a lat/lon snapshot.
We query Google Places API for nearby bus stops and filter by
straight-line distance — if a stop is within ~50m it's on the
same road corridor and visible to the vehicle's cameras.

No heading estimation — each row is a standalone occurrence.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two WGS-84 points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(a))


# ── Data models ──────────────────────────────────────────────────

@dataclass
class Occurrence:
    """One row from the occurrence CSV."""
    datetime_str: str
    trip_id: str
    video_src: str
    lat: float
    lon: float
    auto_comment: str
    vehicle_id: str
    source_id: str  # derived from video filename


@dataclass
class BusStop:
    """A bus stop from Places API."""
    name: str
    address: str | None
    lat: float
    lon: float
    distance_m: float


@dataclass
class OccurrenceWithStops:
    """An occurrence with nearby bus stops within camera range."""
    occurrence: Occurrence
    nearby_stops: list[BusStop]


# ── CSV loader ───────────────────────────────────────────────────

def _extract_source_id(video_src: str) -> str:
    """'…/m002-20260202-1770000093_video.mp4' → 'm002-20260202-1770000093'"""
    name = Path(video_src).stem
    if name.endswith("_video"):
        name = name[: -len("_video")]
    return name


def load_occurrences(csv_path: str) -> list[Occurrence]:
    """Load occurrence rows from CSV."""
    rows: list[Occurrence] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"]) if row.get("lat") else None
                lon = float(row["lon"]) if row.get("lon") else None
                if lat is None or lon is None:
                    continue
                rows.append(Occurrence(
                    datetime_str=row.get("datetime", ""),
                    trip_id=str(row.get("trip_id", "")),
                    video_src=row.get("video_src", ""),
                    lat=lat,
                    lon=lon,
                    auto_comment=row.get("auto_comment", ""),
                    vehicle_id=str(row.get("vehicle_id", "")),
                    source_id=_extract_source_id(row.get("video_src", "")),
                ))
            except (ValueError, KeyError):
                continue

    logger.info("Loaded %d occurrences from %s", len(rows), csv_path)
    return rows


# ── Google Places API ────────────────────────────────────────────

def query_nearby_bus_stops(
    lat: float,
    lon: float,
    api_key: str,
    radius_m: float = 200.0,
) -> list[dict]:
    """Query Google Places API (New) for nearby bus stops."""
    url = "https://places.googleapis.com/v1/places:searchNearby"

    payload = {
        "includedTypes": ["bus_station", "bus_stop"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius_m,
            }
        },
        "rankPreference": "DISTANCE",
    }

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "places" not in data:
            return []

        return [
            {
                "name": p.get("displayName", {}).get("text", "Unknown"),
                "address": p.get("formattedAddress"),
                "lat": p["location"]["latitude"],
                "lon": p["location"]["longitude"],
            }
            for p in data["places"]
        ]
    except requests.RequestException as e:
        logger.error("Places API error at (%.5f, %.5f): %s", lat, lon, e)
        return []


# ── Filter by distance ───────────────────────────────────────────

def filter_by_distance(
    veh_lat: float,
    veh_lon: float,
    raw_stops: list[dict],
    max_distance_m: float = 50.0,
) -> list[BusStop]:
    """Keep only stops within max_distance_m of the vehicle.

    50m default covers the road width plus pavement on both sides,
    which is what the side-facing and front/rear cameras can see.
    """
    filtered = []
    for raw in raw_stops:
        dist = haversine(veh_lat, veh_lon, raw["lat"], raw["lon"])
        if dist <= max_distance_m:
            filtered.append(BusStop(
                name=raw["name"],
                address=raw.get("address"),
                lat=raw["lat"],
                lon=raw["lon"],
                distance_m=round(dist, 1),
            ))
    filtered.sort(key=lambda s: s.distance_m)
    return filtered


# ── Main pipeline ────────────────────────────────────────────────

def find_bus_stop_occurrences(
    csv_path: str,
    api_key: str,
    api_search_radius_m: float = 200.0,
    max_distance_m: float = 50.0,
    rate_limit_s: float = 0.1,
    progress: Any = None,
) -> list[OccurrenceWithStops]:
    """Load CSV → query API → distance filter.

    Parameters
    ----------
    csv_path : str
        Path to the occurrence CSV.
    api_key : str
        Google Places API key.
    api_search_radius_m : float
        Radius for the Places API search (cast a wide net, default 200m).
    max_distance_m : float
        Hard filter — only keep stops within this distance of the vehicle
        (default 50m, roughly camera visibility on a road).
    rate_limit_s : float
        Delay between API calls.

    Returns
    -------
    List of OccurrenceWithStops (only those with ≥1 nearby stop).
    """
    occurrences = load_occurrences(csv_path)
    if not occurrences:
        return []

    # De-duplicate API calls for identical locations
    location_cache: dict[str, list[dict]] = {}

    def _key(lat: float, lon: float) -> str:
        return f"{lat:.6f},{lon:.6f}"

    results: list[OccurrenceWithStops] = []
    total = len(occurrences)
    api_calls = 0

    for i, occ in enumerate(occurrences):
        key = _key(occ.lat, occ.lon)

        if key in location_cache:
            raw_stops = location_cache[key]
        else:
            raw_stops = query_nearby_bus_stops(
                occ.lat, occ.lon, api_key,
                radius_m=api_search_radius_m,
            )
            location_cache[key] = raw_stops
            api_calls += 1
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)

        nearby = filter_by_distance(occ.lat, occ.lon, raw_stops, max_distance_m)

        if nearby:
            results.append(OccurrenceWithStops(
                occurrence=occ,
                nearby_stops=nearby,
            ))

        if progress and (i + 1) % 10 == 0:
            progress((i + 1) / total, f"Processed {i + 1}/{total}")

    logger.info(
        "Done: %d/%d occurrences have bus stops within %dm "
        "(%d unique locations, %d API calls)",
        len(results), total, max_distance_m,
        len(location_cache), api_calls,
    )

    return results


# ── Export helpers ────────────────────────────────────────────────

def results_to_dicts(results: list[OccurrenceWithStops]) -> list[dict]:
    """Flatten results for CSV/JSON export."""
    rows = []
    for r in results:
        occ = r.occurrence
        for stop in r.nearby_stops:
            rows.append({
                "source_id": occ.source_id,
                "vehicle_id": occ.vehicle_id,
                "trip_id": occ.trip_id,
                "datetime": occ.datetime_str,
                "vehicle_lat": occ.lat,
                "vehicle_lon": occ.lon,
                "auto_comment": occ.auto_comment,
                "video_src": occ.video_src,
                "stop_name": stop.name,
                "stop_address": stop.address,
                "stop_lat": stop.lat,
                "stop_lon": stop.lon,
                "distance_m": stop.distance_m,
            })
    return rows


def print_summary(results: list[OccurrenceWithStops]) -> None:
    """Print readable summary."""
    print(f"\n{'=' * 70}")
    print(f"BUS STOP PROXIMITY — {len(results)} occurrences near bus stops")
    print(f"{'=' * 70}\n")

    for r in results:
        occ = r.occurrence
        print(f"📍 {occ.source_id}  |  vehicle {occ.vehicle_id}")
        print(f"   ({occ.lat:.6f}, {occ.lon:.6f})  —  {occ.auto_comment[:80]}")
        for stop in r.nearby_stops:
            print(f"     🚏 {stop.name} — {stop.distance_m}m away")
            if stop.address:
                print(f"        {stop.address}")
        print()


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find bus stops near AV occurrences")
    parser.add_argument("csv_path", help="Path to occurrence CSV")
    parser.add_argument("--api-key", required=True, help="Google Places API key")
    parser.add_argument("--search-radius", type=float, default=200, help="API search radius (m)")
    parser.add_argument("--max-distance", type=float, default=50, help="Max stop distance (m)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    results = find_bus_stop_occurrences(
        csv_path=args.csv_path,
        api_key=args.api_key,
        api_search_radius_m=args.search_radius,
        max_distance_m=args.max_distance,
    )

    print_summary(results)

    if args.output:
        rows = results_to_dicts(results)
        Path(args.output).write_text(json.dumps(rows, indent=2, default=str))
        print(f"Exported {len(rows)} rows to {args.output}")
