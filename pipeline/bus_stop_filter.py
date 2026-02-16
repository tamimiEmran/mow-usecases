"""Bus stop proximity filter for AV fleet video occurrences.

Reads occurrence CSV data with lat/lon, queries Google Places API for
nearby bus stops, then filters to stops that are on either side of the
road (within camera field of view) based on estimated vehicle heading.

Heading estimation:
    - Groups occurrences by trip_id, sorted by datetime
    - Computes bearing between consecutive GPS points
    - Falls back to bearing from the nearest bus stop direction when
      only one point exists in a trip

Lateral filtering:
    - Decomposes vehicle→stop vector into along-track and cross-track
    - Keeps stops where |cross-track| < max_lateral_m (default 25m)
      and along-track is within [-behind_m, +ahead_m] window
    - This ensures only stops actually alongside the road are kept
"""

from __future__ import annotations

import csv
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Geo helpers ──────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two WGS-84 points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees [0, 360) from point 1 → point 2.  0 = North, 90 = East."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def cross_track_along_track(
    veh_lat: float, veh_lon: float,
    stop_lat: float, stop_lon: float,
    heading_deg: float,
) -> tuple[float, float]:
    """Decompose vehicle→stop vector into along-track and cross-track distances.

    Returns (along_track_m, cross_track_m).
      along_track > 0  → stop is ahead of vehicle
      cross_track > 0  → stop is to the RIGHT of travel direction
      cross_track < 0  → stop is to the LEFT
    """
    dist = haversine(veh_lat, veh_lon, stop_lat, stop_lon)
    bearing_to_stop = compute_bearing(veh_lat, veh_lon, stop_lat, stop_lon)

    # Relative angle (radians)
    rel_rad = math.radians(bearing_to_stop - heading_deg)

    along = dist * math.cos(rel_rad)
    cross = dist * math.sin(rel_rad)
    return along, cross


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
    heading: float | None = None  # estimated later


@dataclass
class BusStop:
    """A bus stop returned by Places API."""
    name: str
    address: str | None
    lat: float
    lon: float
    distance_m: float = 0.0
    along_track_m: float = 0.0
    cross_track_m: float = 0.0
    side: str = ""  # "left" | "right" | "ahead" | "behind"


@dataclass
class OccurrenceWithStops:
    """An occurrence enriched with nearby filtered bus stops."""
    occurrence: Occurrence
    heading: float | None
    all_stops: list[BusStop]       # raw API results
    filtered_stops: list[BusStop]  # roadside-only


# ── CSV loader ───────────────────────────────────────────────────

def _extract_source_id(video_src: str) -> str:
    """Extract source_id from video URL or filename.

    e.g. '.../m002-20260202-1770000093_video.mp4' → 'm002-20260202-1770000093'
    """
    name = Path(video_src).stem  # 'm002-20260202-1770000093_video'
    if name.endswith("_video"):
        name = name[: -len("_video")]
    return name


def load_occurrences(csv_path: str) -> list[Occurrence]:
    """Load occurrence data from CSV."""
    rows: list[Occurrence] = []
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV not found: %s", csv_path)
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"]) if row.get("lat") else None
                lon = float(row["lon"]) if row.get("lon") else None
                if lat is None or lon is None:
                    continue

                occ = Occurrence(
                    datetime_str=row.get("datetime", ""),
                    trip_id=str(row.get("trip_id", "")),
                    video_src=row.get("video_src", ""),
                    lat=lat,
                    lon=lon,
                    auto_comment=row.get("auto_comment", ""),
                    vehicle_id=str(row.get("vehicle_id", "")),
                    source_id=_extract_source_id(row.get("video_src", "")),
                )
                rows.append(occ)
            except (ValueError, KeyError) as e:
                logger.debug("Skipping row: %s", e)
                continue

    logger.info("Loaded %d occurrences from %s", len(rows), csv_path)
    return rows


# ── Heading estimation ───────────────────────────────────────────

def estimate_headings(occurrences: list[Occurrence]) -> list[Occurrence]:
    """Estimate vehicle heading for each occurrence from consecutive trip GPS points.

    Groups by trip_id, sorts by datetime, computes bearing between
    consecutive points.  First point in a trip inherits the heading
    of the second point (forward-fill).  Single-point trips get None.
    """
    from collections import defaultdict

    trips: dict[str, list[Occurrence]] = defaultdict(list)
    for occ in occurrences:
        trips[occ.trip_id].append(occ)

    for trip_id, trip_occs in trips.items():
        # Sort by datetime string (ISO-ish, so lexicographic works)
        trip_occs.sort(key=lambda o: o.datetime_str)

        if len(trip_occs) == 1:
            trip_occs[0].heading = None
            continue

        # Compute heading from each point to the next
        for i in range(len(trip_occs) - 1):
            curr = trip_occs[i]
            nxt = trip_occs[i + 1]
            dist = haversine(curr.lat, curr.lon, nxt.lat, nxt.lon)
            if dist > 1.0:  # at least 1m apart to get meaningful bearing
                bearing = compute_bearing(curr.lat, curr.lon, nxt.lat, nxt.lon)
                curr.heading = bearing
            else:
                curr.heading = None

        # Last point: inherit from previous
        if trip_occs[-1].heading is None and len(trip_occs) >= 2:
            trip_occs[-1].heading = trip_occs[-2].heading

        # Forward-fill None headings
        last_known = None
        for occ in trip_occs:
            if occ.heading is not None:
                last_known = occ.heading
            elif last_known is not None:
                occ.heading = last_known

    return occurrences


# ── Google Places API ────────────────────────────────────────────

def query_nearby_bus_stops(
    lat: float,
    lon: float,
    api_key: str,
    radius_m: float = 200.0,
    max_results: int = 20,
) -> list[dict]:
    """Query Google Places API (New) for nearby bus stops.

    Parameters
    ----------
    radius_m : float
        Search radius in metres.  Default 200m — bus stops on the same
        road should be within this range.  Increase for sparse areas.
    """
    url = "https://places.googleapis.com/v1/places:searchNearby"

    payload = {
        "includedTypes": ["bus_station", "bus_stop"],
        "maxResultCount": max_results,
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

        results = []
        for place in data["places"]:
            loc = place.get("location", {})
            results.append({
                "name": place.get("displayName", {}).get("text", "Unknown"),
                "address": place.get("formattedAddress"),
                "lat": loc.get("latitude", 0),
                "lon": loc.get("longitude", 0),
            })
        return results

    except requests.RequestException as e:
        logger.error("Places API error for (%.5f, %.5f): %s", lat, lon, e)
        return []


# ── Roadside filtering ───────────────────────────────────────────

def filter_roadside_stops(
    veh_lat: float,
    veh_lon: float,
    heading: float | None,
    raw_stops: list[dict],
    max_lateral_m: float = 25.0,
    max_ahead_m: float = 100.0,
    max_behind_m: float = 30.0,
) -> tuple[list[BusStop], list[BusStop]]:
    """Filter bus stops to those on either side of the road.

    When heading is available:
        Decomposes each stop's position into along-track and cross-track
        relative to the vehicle's travel direction.  Keeps stops within
        the lateral and longitudinal thresholds.

    When heading is unavailable:
        Falls back to simple distance threshold (max_lateral_m * 2).

    Parameters
    ----------
    max_lateral_m : float
        Maximum perpendicular distance from road centreline (default 25m).
        Typical road width is 6-7m per lane; 25m covers a 4-lane road
        plus pavement on both sides.
    max_ahead_m : float
        Maximum distance ahead along travel direction (default 100m).
        Front camera FOV at typical AV speeds.
    max_behind_m : float
        Maximum distance behind (default 30m).  Rear camera coverage.

    Returns
    -------
    (all_stops, filtered_stops) — both as BusStop dataclass lists
    """
    all_stops: list[BusStop] = []
    filtered: list[BusStop] = []

    for raw in raw_stops:
        dist = haversine(veh_lat, veh_lon, raw["lat"], raw["lon"])

        stop = BusStop(
            name=raw["name"],
            address=raw.get("address"),
            lat=raw["lat"],
            lon=raw["lon"],
            distance_m=round(dist, 1),
        )

        if heading is not None:
            along, cross = cross_track_along_track(
                veh_lat, veh_lon, raw["lat"], raw["lon"], heading,
            )
            stop.along_track_m = round(along, 1)
            stop.cross_track_m = round(cross, 1)

            if cross > 0:
                stop.side = "right"
            else:
                stop.side = "left"

            # Filter: within the road corridor
            in_lateral = abs(cross) <= max_lateral_m
            in_longitudinal = -max_behind_m <= along <= max_ahead_m

            if in_lateral and in_longitudinal:
                filtered.append(stop)
        else:
            # No heading — simple distance fallback
            stop.side = "unknown"
            if dist <= max_lateral_m * 2:
                filtered.append(stop)

        all_stops.append(stop)

    return all_stops, filtered


# ── Main pipeline ────────────────────────────────────────────────

def find_bus_stop_occurrences(
    csv_path: str,
    api_key: str,
    search_radius_m: float = 200.0,
    max_lateral_m: float = 25.0,
    max_ahead_m: float = 100.0,
    max_behind_m: float = 30.0,
    rate_limit_s: float = 0.1,
    progress: Any = None,
) -> list[OccurrenceWithStops]:
    """Full pipeline: load CSV → estimate headings → query API → filter.

    Parameters
    ----------
    csv_path : str
        Path to the occurrence CSV.
    api_key : str
        Google Places API key.
    search_radius_m : float
        API search radius (default 200m).
    max_lateral_m : float
        Max perpendicular distance for roadside filter (default 25m).
    rate_limit_s : float
        Delay between API calls to stay under quota.

    Returns
    -------
    List of OccurrenceWithStops, sorted by number of filtered stops (descending).
    Only occurrences with at least one filtered stop are included.
    """
    # 1. Load and estimate headings
    occurrences = load_occurrences(csv_path)
    if not occurrences:
        logger.warning("No occurrences loaded from %s", csv_path)
        return []

    occurrences = estimate_headings(occurrences)

    # 2. De-duplicate locations (some trips share GPS points)
    #    Key by rounded lat/lon to avoid redundant API calls
    seen_locations: dict[str, list[dict]] = {}

    def _loc_key(lat: float, lon: float) -> str:
        return f"{lat:.6f},{lon:.6f}"

    results: list[OccurrenceWithStops] = []
    total = len(occurrences)

    for i, occ in enumerate(occurrences):
        key = _loc_key(occ.lat, occ.lon)

        if key in seen_locations:
            raw_stops = seen_locations[key]
        else:
            raw_stops = query_nearby_bus_stops(
                occ.lat, occ.lon, api_key,
                radius_m=search_radius_m,
            )
            seen_locations[key] = raw_stops
            time.sleep(rate_limit_s)

        all_stops, filtered_stops = filter_roadside_stops(
            occ.lat, occ.lon, occ.heading, raw_stops,
            max_lateral_m=max_lateral_m,
            max_ahead_m=max_ahead_m,
            max_behind_m=max_behind_m,
        )

        if filtered_stops:
            results.append(OccurrenceWithStops(
                occurrence=occ,
                heading=occ.heading,
                all_stops=all_stops,
                filtered_stops=filtered_stops,
            ))

        if progress and (i + 1) % 10 == 0:
            progress((i + 1) / total, f"Processed {i + 1}/{total} occurrences")

        logger.debug(
            "%s: %d raw stops, %d roadside — heading=%.0f°",
            occ.source_id,
            len(all_stops),
            len(filtered_stops),
            occ.heading or 0,
        )

    # Sort: most roadside stops first
    results.sort(key=lambda r: len(r.filtered_stops), reverse=True)

    logger.info(
        "Bus stop filter complete: %d/%d occurrences have roadside bus stops "
        "(%d unique locations queried)",
        len(results), total, len(seen_locations),
    )

    return results


# ── Pretty print / export ────────────────────────────────────────

def results_to_dicts(results: list[OccurrenceWithStops]) -> list[dict]:
    """Convert results to flat dicts for CSV/JSON export or Gradio display."""
    rows = []
    for r in results:
        occ = r.occurrence
        for stop in r.filtered_stops:
            rows.append({
                "source_id": occ.source_id,
                "vehicle_id": occ.vehicle_id,
                "trip_id": occ.trip_id,
                "datetime": occ.datetime_str,
                "vehicle_lat": occ.lat,
                "vehicle_lon": occ.lon,
                "vehicle_heading": round(r.heading, 1) if r.heading else None,
                "auto_comment": occ.auto_comment,
                "video_src": occ.video_src,
                "stop_name": stop.name,
                "stop_address": stop.address,
                "stop_lat": stop.lat,
                "stop_lon": stop.lon,
                "distance_m": stop.distance_m,
                "along_track_m": stop.along_track_m,
                "cross_track_m": stop.cross_track_m,
                "side": stop.side,
            })
    return rows


def print_summary(results: list[OccurrenceWithStops]) -> None:
    """Print a readable summary to stdout."""
    print(f"\n{'=' * 70}")
    print(f"BUS STOP PROXIMITY RESULTS — {len(results)} occurrences with roadside stops")
    print(f"{'=' * 70}\n")

    for r in results:
        occ = r.occurrence
        heading_str = f"{r.heading:.0f}°" if r.heading else "unknown"
        print(f"📍 {occ.source_id}  |  vehicle {occ.vehicle_id}  |  heading {heading_str}")
        print(f"   ({occ.lat:.6f}, {occ.lon:.6f})  —  {occ.auto_comment[:80]}")
        print(f"   Stops found: {len(r.all_stops)} total, {len(r.filtered_stops)} roadside")

        for stop in r.filtered_stops:
            print(
                f"     🚏 {stop.name} — {stop.distance_m}m "
                f"({stop.side}, along={stop.along_track_m}m, cross={stop.cross_track_m}m)"
            )
            if stop.address:
                print(f"        {stop.address}")
        print()


# ── CLI entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Find bus stops near AV fleet occurrences")
    parser.add_argument("csv_path", help="Path to occurrence CSV")
    parser.add_argument("--api-key", required=True, help="Google Places API key")
    parser.add_argument("--radius", type=float, default=200, help="Search radius in metres")
    parser.add_argument("--lateral", type=float, default=25, help="Max lateral distance in metres")
    parser.add_argument("--ahead", type=float, default=100, help="Max distance ahead in metres")
    parser.add_argument("--behind", type=float, default=30, help="Max distance behind in metres")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    results = find_bus_stop_occurrences(
        csv_path=args.csv_path,
        api_key=args.api_key,
        search_radius_m=args.radius,
        max_lateral_m=args.lateral,
        max_ahead_m=args.ahead,
        max_behind_m=args.behind,
    )

    print_summary(results)

    if args.output:
        rows = results_to_dicts(results)
        Path(args.output).write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nExported {len(rows)} rows to {args.output}")
