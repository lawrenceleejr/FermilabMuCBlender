# Data provenance

All files in this directory are a one-time snapshot fetched on **2026-07-02**
by `scripts/fetch_data.py`. CI consumes these committed files only — no
network access is needed to rebuild the scene.

Coordinates in all files are local ENU meters (x = east, y = north) around
Wilson Hall (41.83841 N, 88.26148 W), via a simple equirectangular
projection (error < ~10 m over the ~10 km scene).

## heightmap.npz

- Source: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
  (Mapzen/Tilezen "terrarium" tiles, `s3.amazonaws.com/elevation-tiles-prod`),
  zoom 13, bbox (41.79, -88.31) - (41.87, -88.20).
- Decoded as `(R*256 + G + B/256) - 32768` meters, then bilinearly resampled
  onto a regular 512x512 ENU grid.
- Contents: `z` (float32 elevation, m ASL), `x0/y0/dx/dy` (grid georef),
  `z_datum` (mean elevation; the scene places z=0 at this datum),
  `origin_latlon`.
- Attribution: terrain data from USGS 3DEP/NED, SRTM, and other sources
  composited by the Mapzen Joerd project.

## buildings.geojson

- Source: OpenStreetMap via the Overpass API
  (`way["building"]` / `relation["building"]` in the same bbox).
- (c) OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright).
- Properties per footprint: `height` (m; from OSM `height`, else
  `building:levels` x 3.2 m, else 5 m default), `name`, `highlight`
  (true for Wilson Hall, whose height is pinned to 76 m).
- Multipolygon relations keep outer rings only.

## imagery.jpg

- Source: [USGS The National Map — USGSImageryOnly service]
  (https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer)
  (NAIP-derived), tiles at zoom 16 (~1.8 m/px), fetched 2026-07-02.
- Public domain (USGS/USDA NAIP). Courtesy: "Imagery courtesy of USGS The
  National Map / NAIP".
- Mosaicked and bilinearly resampled onto the exact ENU extent of
  `heightmap.npz` (contract: image spans x0..x0+511*dx, y0..y0+511*dy;
  top row = north), so the scene builder UV-maps it with a plain planar
  unwrap.

## water.geojson

- Source: OpenStreetMap `natural=water` ways + multipolygon outer rings in
  the fetch bbox; (c) OpenStreetMap contributors, ODbL. Polygons smaller
  than 400 m^2 dropped.

## site_boundary.geojson

- Source: OSM way "Fermi National Accelerator Laboratory"
  (`landuse=institutional`), same license as above.
- Enclosed area of the snapshot polygon: ~27.5 km^2 (matches the ~6800-acre
  site).
