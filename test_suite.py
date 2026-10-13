import json
import os
import unittest

from pipeline import (
    extract_geometry, validate_geometry, validate_coordinates,
    geojson_to_ee, PIPELINE_PROGRESS, init_ee, run_pipeline
)

class TestGeoJSONCompatibility(unittest.TestCase):
    def setUp(self):
        with open("sample inputs/legal_boundary.geojson") as f:
            self.sample_legal = json.load(f)
        with open("sample inputs/no_go_zone.geojson") as f:
            self.sample_nogo = json.load(f)
        with open("area_legal.geojson") as f:
            self.area_legal = json.load(f)
        with open("area_nogo.geojson") as f:
            self.area_nogo = json.load(f)

    def test_sample_inputs_extraction(self):
        """Sample FeatureCollections with optional properties (area, perimeter) parse properly."""
        geom1 = extract_geometry(self.sample_legal)
        self.assertEqual(geom1["type"], "Polygon")
        self.assertIn("coordinates", geom1)
        validate_geometry(geom1)

        geom2 = extract_geometry(self.sample_nogo)
        self.assertEqual(geom2["type"], "Polygon")
        validate_geometry(geom2)

    def test_area_inputs_with_custom_metadata(self):
        """area_legal (empty properties) and area_nogo (custom @circle metadata) parse properly."""
        geom1 = extract_geometry(self.area_legal)
        self.assertEqual(geom1["type"], "Polygon")
        validate_geometry(geom1)

        geom2 = extract_geometry(self.area_nogo)
        self.assertEqual(geom2["type"], "Polygon")
        validate_geometry(geom2)

    def test_direct_feature_and_polygon_inputs(self):
        """Standard GeoJSON Feature and raw Polygon geometries are accepted."""
        raw_polygon = {
            "type": "Polygon",
            "coordinates": [[[80.0, 20.0], [80.1, 20.0], [80.1, 20.1], [80.0, 20.1], [80.0, 20.0]]]
        }
        geom = extract_geometry(raw_polygon)
        self.assertEqual(geom["type"], "Polygon")
        validate_geometry(geom)

        feature_input = {
            "type": "Feature",
            "properties": {"foo": "bar", "@circle": "dummy"},
            "geometry": raw_polygon
        }
        geom = extract_geometry(feature_input)
        self.assertEqual(geom["type"], "Polygon")
        validate_geometry(geom)

    def test_unclosed_polygon_rejected(self):
        """Polygons with unclosed rings are detected and rejected."""
        unclosed = {
            "type": "Polygon",
            "coordinates": [[[80.0, 20.0], [80.1, 20.0], [80.1, 20.1], [80.0, 20.1], [80.0, 20.05]]]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_geometry(unclosed)
        self.assertIn("not closed", str(ctx.exception))

    def test_invalid_types_rejected(self):
        """Non-polygonal geometries (LineString, Point) are rejected."""
        linestring = {
            "type": "LineString",
            "coordinates": [[80.0, 20.0], [80.1, 20.0]]
        }
        with self.assertRaises(ValueError) as ctx:
            extract_geometry(linestring)
        self.assertIn("Unsupported geometry type", str(ctx.exception))

    def test_coordinates_out_of_bounds(self):
        """Coordinates outside valid WGS84 range are rejected."""
        out_of_bounds = {
            "type": "Polygon",
            "coordinates": [[[195.0, 20.0], [80.1, 20.0], [80.1, 20.1], [80.0, 20.1], [195.0, 20.0]]]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_geometry(out_of_bounds)
        self.assertIn("out of bounds", str(ctx.exception))


from app import app, health_check, get_progress, get_results, run, PipelineRequest
from fastapi import BackgroundTasks, HTTPException

class TestAPIEndpoints(unittest.TestCase):
    def test_health_check(self):
        res = health_check()
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("message", res)

    def test_progress_endpoint(self):
        res = get_progress()
        self.assertIn("status", res)
        self.assertIn("progress", res)
        self.assertIn("stage", res)

    def test_concurrent_run_rejection(self):
        """When an analysis is marked running, subsequent run calls raise HTTPException 409 Conflict."""
        original_status = PIPELINE_PROGRESS.get("status")
        try:
            PIPELINE_PROGRESS["status"] = "running"
            with open("sample inputs/legal_boundary.geojson") as f:
                legal = json.load(f)
            req = PipelineRequest(mine_geojson=legal)
            tasks = BackgroundTasks()
            with self.assertRaises(HTTPException) as ctx:
                run(req, tasks)
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertIn("currently in progress", str(ctx.exception.detail))
        finally:
            PIPELINE_PROGRESS["status"] = original_status


if __name__ == "__main__":
    unittest.main()
