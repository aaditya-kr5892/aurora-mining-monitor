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

def geojson_to_ee(geojson_dict):
    return ee.Geometry(geojson_dict["features"][0]["geometry"])

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
    dimensions=512
):
    """
    Exports an EE Image (single or RGB) as PNG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    params = {
        "min": min,
        "max": max,
        "dimensions": dimensions,
        "region": geom.bounds().getInfo()["coordinates"],
        "format": "png"
    }

    if bands:
        params["bands"] = bands

    try:
        url = image.clip(geom).getThumbURL(params)
        r = requests.get(url)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Error exporting {out_path}: {e}")



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


def run_detection(geom, valid_keys, monthly_cache, threshold_cache, out_id="", generate_maps=True):
    """
    Optimized detection run.
    Uses pre-computed threshold_cache.
    Batches area calculations to avoid getInfo() inside the loop.
    """
    rows = []
    masks_raw = {}
    quantified_maps = {}

    prev_s2 = None
    prev_s1 = None
    prev_change = None
    cum_mask = None
    
    # Temporary list to store server-side area numbers for batch resolution
    area_promises = []

    for y, m in valid_keys:

        # Cache is unclipped, so we clip here for local analysis
        s2_idx = monthly_cache[(y, m)]["s2_idx"].clip(geom)
        s2_rgb = monthly_cache[(y, m)]["s2_rgb"].clip(geom)
        s1     = monthly_cache[(y, m)]["s1"].clip(geom)

        # No empty check needed here; valid_keys guarantees data exists.

        if prev_s2 is None or prev_s1 is None:
            prev_s2, prev_s1 = s2_idx, s1
            continue

        # ---------- CVA (Local Change Magnitude) ----------
        d_s2 = s2_idx.subtract(prev_s2).pow(2).reduce("sum").sqrt()
        d_s1 = s1.subtract(prev_s1).abs().reduce("sum")

        # ---------- OPTIMIZATION: Use Pre-Computed Thresholds ----------
        # Thresholds are global (derived from mine) and pre-calculated.
        # We wrap the cached raw number in an Image.constant for comparison.
        opt_val = threshold_cache[(y, m)]["opt"]
        sar_val = threshold_cache[(y, m)]["sar"]
        
        opt_thresh_img = ee.Image.constant(opt_val)
        sar_thresh_img = ee.Image.constant(sar_val)

        candidate = d_s2.gt(opt_thresh_img).Or(
            d_s1.gt(sar_thresh_img)
        )

        stable = candidate if prev_change is None else candidate.And(prev_change)
        cum_mask = stable if cum_mask is None else cum_mask.Or(stable)

        date_key = f"{y}-{m:02d}-01"

        if generate_maps:
            masks_raw[date_key] = cum_mask.rename("mine_mask")

            # ---------- VISUALIZATION LOGIC ----------
            ndvi = s2_idx.select("NDVI")
            viz_mask = cum_mask.And(ndvi.lte(0.3)).selfMask()
            viz_rgb = s2_rgb.updateMask(viz_mask)

            png_path = f"outputs/{out_id}/{date_key}.png"
            export_png(
                viz_rgb, geom, png_path,
                bands=["B4", "B3", "B2"], min=0, max=0.3
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

        # Optimization: Do NOT call getInfo() here. Store the EE object.
        # We default to 0 if area is null (handled in batch resolution usually, 
        # but safely assumed 0 if mask is empty).
        area_ee = ee.Algorithms.If(area, ee.Number(area).divide(1e6), 0.0)
        
        rows.append({
            "date": date_key,
            # Placeholder for value
            "area_km2": 0.0
        })
        area_promises.append(area_ee)

        prev_s2, prev_s1, prev_change = s2_idx, s1, candidate

    # ---------- RESOLVE BATCHED AREAS ----------
    # Single network call to resolve all areas for this zone
    if area_promises:
        resolved_areas = ee.List(area_promises).getInfo()
        for i, val in enumerate(resolved_areas):
            rows[i]["area_km2"] = val

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
    PIPELINE_PROGRESS["status"] = "running"
    PIPELINE_PROGRESS["progress"] = 10
    try:
        t0 = time.time()
        PIPELINE_PROGRESS["status"] = "running"
        PIPELINE_PROGRESS["progress"] = 0
        init_ee()
        PIPELINE_PROGRESS["progress"] = 10
        mine_geom = geojson_to_ee(mine_geojson)
        no_go_geoms = []
        if no_go_geojson_list:
            no_go_geoms = [geojson_to_ee(g) for g in no_go_geojson_list]
        
        PIPELINE_PROGRESS["progress"] = 20
        years  = range(2020, 2025)
        months = [1, 3, 5, 7, 9, 11]

        # OPTIMIZATION: Generate candidates list purely in Python
        # We will filter these efficiently in a batch later
        candidate_months = [(y, m) for y in years for m in months]
        
        PIPELINE_PROGRESS["progress"] = 30

        # --- STEP 1: POPULATE CACHE & CHECK VALIDITY (BATCHED) ---
        monthly_cache = {}
        valid_keys = []
        
        # We need to check which candidates actually have data. 
        # Instead of 1 getInfo per month, we batch the size check.
        # Create list of images to check
        check_imgs = []
        for y, m in candidate_months:
            # We use mine_geom to check availability (S2 is consistent regionally)
            check_imgs.append(get_monthly_s2_indices(y, m, mine_geom))
            
        # Single network call to check all band counts
        # This replaces get_valid_months loop
        band_counts = ee.List([img.bandNames().size() for img in check_imgs]).getInfo()
        
        for i, count in enumerate(band_counts):
            if count > 0:
                y, m = candidate_months[i]
                
                # Fetch full data objects (no cost until getInfo)
                s2_idx = check_imgs[i] # Already created
                s2_rgb = get_monthly_s2_rgb(y, m, mine_geom)
                s1     = get_monthly_s1(y, m, mine_geom)
                
                monthly_cache[(y, m)] = {
                    "s2_idx": s2_idx,
                    "s2_rgb": s2_rgb,
                    "s1": s1
                }
                valid_keys.append((y, m))

        # --- STEP 2: PRE-COMPUTE THRESHOLDS (BATCHED) ---
        # We calculate thresholds on the MINE geometry once and reuse them.
        threshold_cache = {}
        
        threshold_ops = [] # List to store server-side reduction objects
        threshold_indices = [] # Keep track of which (y,m) corresponds to which op

        prev_s2 = None
        prev_s1 = None

        for y, m in valid_keys:
            s2 = monthly_cache[(y, m)]["s2_idx"]
            s1 = monthly_cache[(y, m)]["s1"]

            if prev_s2 is None:
                prev_s2, prev_s1 = s2, s1
                continue
            
            # Change Magnitude on Mine Geometry
            d_s2 = s2.subtract(prev_s2).pow(2).reduce("sum").sqrt()
            d_s1 = s1.subtract(prev_s1).abs().reduce("sum")

            # Define Reducers (Server-side only)
            opt_reducer = d_s2.reduceRegion(
                ee.Reducer.percentile([85]), mine_geom, 10, maxPixels=1e13
            )
            sar_reducer = d_s1.reduceRegion(
                ee.Reducer.percentile([80]), mine_geom, 10, maxPixels=1e13
            )
            
            # Store operation for batch execution
            threshold_ops.append({
                "opt": opt_reducer.values().get(0),
                "sar": sar_reducer.values().get(0)
            })
            threshold_indices.append((y, m))

            prev_s2, prev_s1 = s2, s1
        
        # Execute all threshold calculations in ONE network call
        if threshold_ops:
            threshold_results = ee.List(threshold_ops).getInfo()
            
            # Unpack results into cache
            for idx, res in enumerate(threshold_results):
                y, m = threshold_indices[idx]
                threshold_cache[(y, m)] = res

        PIPELINE_PROGRESS["progress"] = 50

        # --- STEP 3: RUN DETECTION (Optimized) ---
        mine_out = run_detection(
            mine_geom, valid_keys, monthly_cache, threshold_cache, 
            out_id="mine", generate_maps=True
        )
        
        PIPELINE_PROGRESS["progress"] = 60
        no_go_outs = []
        total_zones = len(no_go_geoms)
        
        if total_zones == 0:
            PIPELINE_PROGRESS["progress"] = 80
        else:
            for i, zg in enumerate(no_go_geoms):
                no_go_outs.append(
                    run_detection(
                        zg, valid_keys, monthly_cache, threshold_cache, 
                        out_id=f"no_go_zone_{i}", generate_maps=False
                    )
                )
                PIPELINE_PROGRESS["progress"] = 60 + int(20 * (i + 1) / total_zones)

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
        PIPELINE_PROGRESS["progress"] = 90
        print("Runtime:", time.time() - t0)
        PIPELINE_PROGRESS["status"] = "done"
        PIPELINE_PROGRESS["progress"] = 100
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
        # -----------------------------------------------

        return results
    except Exception as e:
        PIPELINE_PROGRESS["status"] = "error"
        PIPELINE_PROGRESS["error"] = str(e)
        raise e