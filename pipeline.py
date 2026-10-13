import json
import numpy as np
import ee
import time
import os
from google.oauth2 import service_account
import requests

# Global variable to track progress for the frontend
PIPELINE_PROGRESS = {
    "status": "idle",
    "stage": "Ready",
    "progress": 0,
    "error": None
}

# t0 = time.time()

def init_ee():
    gee_json = os.getenv("GEE_KEY_JSON")
    if gee_json:
        key_info = json.loads(gee_json)
        credentials = service_account.Credentials.from_service_account_info(
            key_info,
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
    else:
        key_path = os.getenv("GEE_KEY_PATH", "key.json")
        if not os.path.exists(key_path) and os.path.exists("/etc/secrets/key.json"):
            key_path = "/etc/secrets/key.json"
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
    ee.Initialize(credentials)


def km2(area_m2):
    return area_m2.divide(1e6)


def extract_geometry(geojson_data):
    """
    Extracts a standard geometry dictionary from FeatureCollection, Feature,
    Polygon, or MultiPolygon. Ignores non-geometry metadata (such as @circle).
    """
    if not isinstance(geojson_data, dict):
        raise ValueError("GeoJSON must be a valid JSON object/dictionary")
    
    geo_type = geojson_data.get("type")
    if not geo_type:
        raise ValueError("GeoJSON input is missing required 'type' field")
    
    if geo_type == "FeatureCollection":
        features = geojson_data.get("features")
        if not features or not isinstance(features, list):
            raise ValueError("FeatureCollection must contain a non-empty 'features' list")
        
        geometries = []
        for idx, feat in enumerate(features):
            if not isinstance(feat, dict):
                raise ValueError(f"Feature at index {idx} is invalid")
            geom = feat.get("geometry")
            if not geom or not isinstance(geom, dict):
                raise ValueError(f"Feature at index {idx} is missing a valid 'geometry' object")
            geometries.append(geom)
            
        if len(geometries) == 1:
            return geometries[0]
        else:
            all_polys = []
            for g in geometries:
                g_type = g.get("type")
                if g_type == "Polygon":
                    all_polys.append(g["coordinates"])
                elif g_type == "MultiPolygon":
                    all_polys.extend(g["coordinates"])
                else:
                    raise ValueError(f"FeatureCollection contains unsupported geometry type '{g_type}'")
            return {"type": "MultiPolygon", "coordinates": all_polys}

    elif geo_type == "Feature":
        geom = geojson_data.get("geometry")
        if not geom or not isinstance(geom, dict):
            raise ValueError("Feature is missing a valid 'geometry' object")
        return geom

    elif geo_type in ["Polygon", "MultiPolygon"]:
        return geojson_data

    elif geo_type in ["Point", "MultiPoint", "LineString", "MultiLineString"]:
        raise ValueError(
            f"Unsupported geometry type '{geo_type}'. A Polygon or MultiPolygon boundary is required."
        )
    else:
        raise ValueError(f"Unrecognized GeoJSON type: '{geo_type}'")


def validate_coordinates(coords, ring_name="ring"):
    if not coords or len(coords) < 4:
        raise ValueError(f"Polygon {ring_name} must contain at least 4 coordinates")
    first, last = coords[0], coords[-1]
    if len(first) < 2 or len(last) < 2:
        raise ValueError(f"Coordinates in {ring_name} must contain at least [longitude, latitude]")
    if first[0] != last[0] or first[1] != last[1]:
        raise ValueError(
            f"Polygon {ring_name} is not closed: start coordinate {first} does not match end coordinate {last}"
        )
    for pt in coords:
        lon, lat = pt[0], pt[1]
        if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"Coordinate [{lon}, {lat}] is out of bounds: longitude must be in [-180, 180], latitude in [-90, 90]"
            )


def validate_geometry(geom):
    g_type = geom.get("type")
    coords = geom.get("coordinates")
    if not coords or not isinstance(coords, list):
        raise ValueError(f"Geometry '{g_type}' is missing valid coordinates array")
    
    if g_type == "Polygon":
        for i, ring in enumerate(coords):
            validate_coordinates(ring, f"ring {i}")
    elif g_type == "MultiPolygon":
        for p_idx, poly in enumerate(coords):
            for r_idx, ring in enumerate(poly):
                validate_coordinates(ring, f"polygon {p_idx} ring {r_idx}")
    else:
        raise ValueError(f"Expected Polygon or MultiPolygon, got '{g_type}'")
    return geom


def geojson_to_ee(geojson_dict):
    clean_geom = extract_geometry(geojson_dict)
    validate_geometry(clean_geom)
    return ee.Geometry(clean_geom)

def mask_s2_clouds(img):
    qa = img.select("QA60")
    cloud  = 1 << 10
    cirrus = 1 << 11
    mask = qa.bitwiseAnd(cloud).eq(0).And(
           qa.bitwiseAnd(cirrus).eq(0))
    # This divides by 10000, resulting in 0.0 to 1.0 float range
    return img.updateMask(mask).divide(10000)

def get_monthly_s2_indices(year, month, geom):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(mask_s2_clouds)
    )

    size = col.size()

    def build(img):
        B4  = img.select("B4")
        B8  = img.select("B8")
        B11 = img.select("B11")

        ndvi = B8.subtract(B4).divide(B8.add(B4)).rename("NDVI")
        nbr  = B8.subtract(B11).divide(B8.add(B11)).rename("NBR")

        return ee.Image.cat([ndvi, nbr])

    return ee.Image(
        ee.Algorithms.If(
            size.gt(0),
            build(col.median()),
            # FIX: Use select([]) to create 0-band image instead of rename([])
            ee.Image().select([])
        )
    )

def export_png(
    image,
    geom,
    out_path,
    bands=None,
    min=0,
    max=0.3, 
    dimensions=512,
    region=None
):
    """
    Exports an EE Image (single or RGB) as PNG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if region is None:
        region = geom.bounds().getInfo()["coordinates"]

    params = {
        "min": min,
        "max": max,
        "dimensions": dimensions,
        "region": region,
        "format": "png"
    }

    if bands:
        params["bands"] = bands

    try:
        url = image.clip(geom).getThumbURL(params)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Warning: Map export skipped for {out_path}: {e}")



def get_monthly_s1(year, month, geom):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    size = col.size()

    def build(img):
        vv = img.select("VV")
        vh = img.select("VH")
        ratio = vv.subtract(vh).rename("RATIO")
        return ee.Image.cat([vv, vh, ratio])

    return ee.Image(
        ee.Algorithms.If(
            size.gt(0),
            build(col.median()),
            # FIX: Use select([]) to create 0-band image instead of rename([])
            ee.Image().select([])
        )
    )

def get_monthly_s2_rgb(year, month, geom):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(mask_s2_clouds) 
    )

    return ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.median().select(["B4", "B3", "B2"]),
            # FIX: Use select([]) to create 0-band image instead of rename([])
            ee.Image().select([])
        )
    )


def run_detection(geom, valid_keys, monthly_cache, threshold_cache, out_id="", generate_maps=True, chunk_size=4):
    """
    Optimized detection run.
    Uses pre-computed threshold_cache.
    Batches area calculations in chunks to prevent 'Too many concurrent aggregations'.
    Handles missing optical or SAR sensor thresholds gracefully.
    """
    rows = []
    masks_raw = {}
    quantified_maps = {}

    prev_s2 = None
    prev_s1 = None
    prev_change = None
    cum_mask = None
    
    # Pre-compute region boundary once to avoid repeated getInfo network calls
    region_coords = geom.bounds().getInfo()["coordinates"] if generate_maps else None

    # Temporary list to store server-side area numbers for batch resolution
    area_promises = []

    for y, m in valid_keys:
        date_key = f"{y}-{m:02d}-01"

        # Cache is unclipped, so we clip here for local analysis
        s2_idx = monthly_cache[(y, m)]["s2_idx"].clip(geom)
        s2_rgb = monthly_cache[(y, m)]["s2_rgb"].clip(geom)
        s1     = monthly_cache[(y, m)]["s1"].clip(geom)

        if prev_s2 is None or prev_s1 is None:
            prev_s2, prev_s1 = s2_idx, s1
            continue

        # ---------- CVA (Local Change Magnitude) ----------
        d_s2 = s2_idx.subtract(prev_s2).pow(2).reduce("sum").sqrt()
        d_s1 = s1.subtract(prev_s1).abs().reduce("sum")

        thresh_entry = threshold_cache.get((y, m), {})
        opt_val = thresh_entry.get("opt")
        sar_val = thresh_entry.get("sar")

        # Multi-sensor fusion with fallback when one sensor is unavailable / cloud-covered
        conditions = []
        if opt_val is not None:
            conditions.append(d_s2.gt(ee.Image.constant(opt_val)))
        if sar_val is not None:
            conditions.append(d_s1.gt(ee.Image.constant(sar_val)))

        if len(conditions) == 2:
            candidate = conditions[0].Or(conditions[1])
        elif len(conditions) == 1:
            candidate = conditions[0]
        else:
            candidate = ee.Image.constant(0)

        stable = candidate if prev_change is None else candidate.And(prev_change)
        cum_mask = stable if cum_mask is None else cum_mask.Or(stable)

        if generate_maps:
            masks_raw[date_key] = cum_mask.rename("mine_mask")

            # ---------- VISUALIZATION LOGIC ----------
            ndvi = s2_idx.select("NDVI")
            viz_mask = cum_mask.And(ndvi.lte(0.3)).selfMask()
            viz_rgb = s2_rgb.updateMask(viz_mask)

            png_path = f"outputs/{out_id}/{date_key}.png"
            export_png(
                viz_rgb, geom, png_path,
                bands=["B4", "B3", "B2"], min=0, max=0.3,
                region=region_coords
            )

            quantified_maps[date_key] = png_path.replace("outputs", "/static")

        # ---------- AREA CALCULATION (BATCHED) ----------
        cum_mask_named = cum_mask.rename("mine_mask")
        area_img = cum_mask_named.multiply(ee.Image.pixelArea())

        area = (
            area_img.reduceRegion(
                ee.Reducer.sum(),
                geom, 10, maxPixels=1e13
            ).get("mine_mask")
        )

        area_ee = ee.Algorithms.If(area, ee.Number(area).divide(1e6), 0.0)
        
        rows.append({
            "date": date_key,
            "area_km2": 0.0
        })
        area_promises.append(area_ee)

        prev_s2, prev_s1, prev_change = s2_idx, s1, candidate

    # ---------- RESOLVE BATCHED AREAS IN CHUNKS ----------
    # Chunking resolves aggregations safely without tripping GEE concurrency limits
    if area_promises:
        resolved_areas = []
        for i in range(0, len(area_promises), chunk_size):
            chunk = area_promises[i:i + chunk_size]
            res_chunk = ee.List(chunk).getInfo()
            resolved_areas.extend(res_chunk)

        for i, val in enumerate(resolved_areas):
            rows[i]["area_km2"] = val if val is not None else 0.0

    return {
        "timeseries": rows,
        "masks_raw": masks_raw,
        "quantified_maps": quantified_maps,
        "analysis_start": rows[0]["date"] if rows else None,
        "analysis_end": rows[-1]["date"] if rows else None
    }


def first_violation(timeseries):
    for row in timeseries:
        if row["area_km2"] > 0:
            return row["date"]
    return None

def compute_monthly_growth(timeseries):
    growth = []
    for i in range(1, len(timeseries)):
        g = timeseries[i]["area_km2"] - timeseries[i-1]["area_km2"]
        growth.append({
            "date": timeseries[i]["date"],
            "growth_km2": g
        })
    return growth


def predict_next_month(timeseries):
    if len(timeseries) < 2:
        return None

    x = np.arange(len(timeseries))
    y = np.array([t["area_km2"] for t in timeseries])

    coeffs = np.polyfit(x, y, 1)   # linear trend
    next_area = coeffs[0] * (len(timeseries)) + coeffs[1]

    return float(next_area)


def classify_alert(area_km2, zone_area_km2):
    pct = (area_km2 / zone_area_km2) * 100 if zone_area_km2 > 0 else 0

    if pct == 0:
        return "none"
    elif pct < 1:
        return "soft"
    else:
        return "hard"


# ---------- CORE PIPELINE ----------
def run_pipeline(mine_geojson, no_go_geojson_list=None):
    global PIPELINE_PROGRESS
    PIPELINE_PROGRESS.update({
        "status": "running",
        "stage": "Validating input boundaries...",
        "progress": 5,
        "error": None
    })
    try:
        t0 = time.time()
        
        PIPELINE_PROGRESS.update({
            "stage": "Initializing Google Earth Engine...",
            "progress": 10
        })
        init_ee()

        PIPELINE_PROGRESS.update({
            "stage": "Parsing and validating GeoJSON boundaries...",
            "progress": 15
        })
        mine_geom = geojson_to_ee(mine_geojson)
        no_go_geoms = []
        if no_go_geojson_list:
            no_go_geoms = [geojson_to_ee(g) for g in no_go_geojson_list]
        
        PIPELINE_PROGRESS.update({
            "stage": "Checking candidate satellite collection dates...",
            "progress": 25
        })
        years  = range(2020, 2025)
        months = [1, 3, 5, 7, 9, 11]
        candidate_months = [(y, m) for y in years for m in months]

        # --- STEP 1: POPULATE CACHE & CHECK VALIDITY (BATCHED) ---
        monthly_cache = {}
        valid_keys = []
        
        check_imgs = []
        for y, m in candidate_months:
            check_imgs.append(get_monthly_s2_indices(y, m, mine_geom))
            
        PIPELINE_PROGRESS.update({
            "stage": "Filtering clear-sky Sentinel-2 and Sentinel-1 scenes...",
            "progress": 35
        })
        band_counts = ee.List([img.bandNames().size() for img in check_imgs]).getInfo()
        
        for i, count in enumerate(band_counts):
            if count > 0:
                y, m = candidate_months[i]
                s2_idx = check_imgs[i]
                s2_rgb = get_monthly_s2_rgb(y, m, mine_geom)
                s1     = get_monthly_s1(y, m, mine_geom)
                
                monthly_cache[(y, m)] = {
                    "s2_idx": s2_idx,
                    "s2_rgb": s2_rgb,
                    "s1": s1
                }
                valid_keys.append((y, m))

        # --- STEP 2: PRE-COMPUTE THRESHOLDS (BATCHED IN CHUNKS) ---
        PIPELINE_PROGRESS.update({
            "stage": "Computing adaptive Change Vector Analysis (CVA) thresholds...",
            "progress": 45
        })
        threshold_cache = {}
        threshold_ops = []
        threshold_indices = []

        prev_s2 = None
        prev_s1 = None

        for y, m in valid_keys:
            s2 = monthly_cache[(y, m)]["s2_idx"]
            s1 = monthly_cache[(y, m)]["s1"]

            if prev_s2 is None:
                prev_s2, prev_s1 = s2, s1
                continue
            
            d_s2 = s2.subtract(prev_s2).pow(2).reduce("sum").sqrt()
            d_s1 = s1.subtract(prev_s1).abs().reduce("sum")

            opt_reducer = d_s2.reduceRegion(
                ee.Reducer.percentile([85]), mine_geom, 10, maxPixels=1e13
            )
            sar_reducer = d_s1.reduceRegion(
                ee.Reducer.percentile([80]), mine_geom, 10, maxPixels=1e13
            )
            
            # Guard against completely cloudy/empty observations
            opt_val = ee.Algorithms.If(
                opt_reducer.values().size().gt(0), opt_reducer.values().get(0), None
            )
            sar_val = ee.Algorithms.If(
                sar_reducer.values().size().gt(0), sar_reducer.values().get(0), None
            )

            threshold_ops.append({
                "opt": opt_val,
                "sar": sar_val
            })
            threshold_indices.append((y, m))

            prev_s2, prev_s1 = s2, s1
        
        # Chunked network resolution to prevent "Too many concurrent aggregations"
        CHUNK_SIZE = 4
        if threshold_ops:
            for i in range(0, len(threshold_ops), CHUNK_SIZE):
                chunk = threshold_ops[i:i + CHUNK_SIZE]
                threshold_results = ee.List(chunk).getInfo()
                for idx, res in enumerate(threshold_results):
                    y, m = threshold_indices[i + idx]
                    threshold_cache[(y, m)] = res

        PIPELINE_PROGRESS.update({
            "stage": "Running mining excavation detection on legal mine zone...",
            "progress": 60
        })

        # --- STEP 3: RUN DETECTION (Optimized) ---
        mine_out = run_detection(
            mine_geom, valid_keys, monthly_cache, threshold_cache, 
            out_id="mine", generate_maps=True, chunk_size=CHUNK_SIZE
        )
        
        PIPELINE_PROGRESS.update({
            "stage": "Running compliance detection on No-Go zones...",
            "progress": 75
        })
        no_go_outs = []
        total_zones = len(no_go_geoms)
        
        if total_zones == 0:
            PIPELINE_PROGRESS["progress"] = 85
        else:
            for i, zg in enumerate(no_go_geoms):
                no_go_outs.append(
                    run_detection(
                        zg, valid_keys, monthly_cache, threshold_cache, 
                        out_id=f"no_go_zone_{i}", generate_maps=False, chunk_size=CHUNK_SIZE
                    )
                )
                PIPELINE_PROGRESS["progress"] = 75 + int(15 * (i + 1) / total_zones)

        PIPELINE_PROGRESS.update({
            "stage": "Compiling metrics, violation logs, and trend predictions...",
            "progress": 90
        })

        mine_ts = mine_out["timeseries"]
        mine_area_total = (
            mine_geom.area().divide(1e6).getInfo()
        )

        current_area = mine_ts[-1]["area_km2"] if mine_ts else 0.0
        current_pct = (current_area / mine_area_total) * 100 if mine_area_total > 0 else 0
        
        monthly_growth = compute_monthly_growth(mine_ts)
        current_month_growth = monthly_growth[-1]["growth_km2"] if monthly_growth else 0.0
        
        mine_predicted_next_area = predict_next_month(mine_ts)

        no_go_results = {}

        for i, (zg, out) in enumerate(zip(no_go_geoms, no_go_outs)):
            zone_ts = out["timeseries"]

            zone_area = zg.area().divide(1e6).getInfo()
            zone_current = zone_ts[-1]["area_km2"] if zone_ts else 0.0
            zone_pct = (zone_current / zone_area) * 100 if zone_area > 0 else 0.0

            alerts_log = []
            prev_area = 0.0

            for row in zone_ts:
                growth = row["area_km2"] - prev_area
                alert = classify_alert(row["area_km2"], zone_area)

                alerts_log.append({
                    "date": row["date"],
                    "area_km2": row["area_km2"],
                    "growth_km2": growth,
                    "alert": alert
                })

                prev_area = row["area_km2"]

            zone_predicted_next_area = predict_next_month(zone_ts)
            zone_predicted_next_alert = classify_alert(
                zone_predicted_next_area or 0.0,
                zone_area
            )

            no_go_results[f"no_go_zone_{i}"] = {
                "timeseries": zone_ts,
                "current_area_km2": zone_current,
                "percentage_mined": zone_pct,
                "alerts": alerts_log,
                "first_violation": first_violation(zone_ts),
                "monthly_growth": compute_monthly_growth(zone_ts),
                "predicted_next_area": zone_predicted_next_area,
                "predicted_next_alert": zone_predicted_next_alert,
                "analysis_start": out["analysis_start"],
                "analysis_end": out["analysis_end"]
            }

        mine_ts = sorted(mine_ts, key=lambda x: x["date"])
        print("Runtime:", time.time() - t0)
        
        PIPELINE_PROGRESS.update({
            "status": "done",
            "stage": "Analysis completed successfully",
            "progress": 100,
            "error": None
        })

        results = {
            "metadata": {
                "analysis_start": mine_out["analysis_start"],
                "analysis_end": mine_out["analysis_end"],
                "valid_months": valid_keys
            },
            "mine": {
                "timeseries": mine_ts,
                "current_area_km2": current_area,
                "percentage_mined": current_pct,
                "monthly_growth": monthly_growth,
                "current_month_growth": current_month_growth,
                "predicted_next_month_area": mine_predicted_next_area,
                "quantified_maps": mine_out["quantified_maps"]
            },
            "no_go_zones": no_go_results,
        }
        try:
            output_path = "output.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"✅ Output successfully saved to {os.path.abspath(output_path)}")
        except Exception as e:
            print(f"❌ Failed to save output.json: {e}")

        return results
    except Exception as e:
        PIPELINE_PROGRESS.update({
            "status": "error",
            "stage": "Failed",
            "error": str(e)
        })
        raise e