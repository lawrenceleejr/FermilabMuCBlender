# Third-party assets

Most external assets are **CC0 1.0** or US-government public domain; the OSM
layout is **ODbL** and needs attribution in published renders. None of it is
committed — fetch it with:

```sh
python3 tools/fetch_assets.py                            # textures, skies, star map
blender -b --python tools/make_sky_hdri.py -- --res 8k   # the night sky
python3 tools/fetch_geodata.py                           # the real site layout (OSM)
python3 tools/fetch_dem.py                               # real terrain (AWS terrain tiles)
```

| Asset | Source | Use |
|---|---|---|
| Grass004 (2K JPG) | ambientCG, https://ambientcg.com/a/Grass004 | prairie grass |
| Ground037 (2K JPG) | ambientCG, https://ambientcg.com/a/Ground037 | dry ground blended into grass |
| Concrete034 (2K JPG) | ambientCG, https://ambientcg.com/a/Concrete034 | Wilson Hall concrete |
| Concrete042A (1K JPG) | ambientCG, https://ambientcg.com/a/Concrete042A | service buildings |
| Asphalt031 (1K JPG) | ambientCG, https://ambientcg.com/a/Asphalt031 | roads |
| qwantani_dusk_2_puresky | Poly Haven, https://polyhaven.com/a/qwantani_dusk_2_puresky | sky / ambient light |
| kloppenheim_06_puresky | Poly Haven, https://polyhaven.com/a/kloppenheim_06_puresky | alternate sky |
| qwantani_night_puresky | Poly Haven, https://polyhaven.com/a/qwantani_night_puresky | alternate sky |
| Deep Star Maps 2020 (`starmap_2020_8k.exr`) | NASA/GSFC Scientific Visualization Studio, https://svs.gsfc.nasa.gov/4851 (public domain; Gaia DR2: ESA/Gaia/DPAC) | night sky: rotated into Fermilab's horizon frame by `tools/make_sky_hdri.py` to produce `assets/hdri/fermilab_night_sky.exr` |

| Fermilab site layout (`assets/geo/fermilab_site.json`) | OpenStreetMap via Overpass — boundary way 31974155, roads, water, woods, buildings | the real site plan, projected to metres about Wilson Hall. **ODbL**: credit "© OpenStreetMap contributors" wherever a render is published |
| Terrain (`assets/geo/dem.png`) | AWS Terrain Tiles (terrarium), open data, SRTM/NED derived | real elevation for the distance view. 60 km square, 131–398 m |

Everything else in the scene (terrain, Wilson Hall, rings, water, trees, lights,
the collider itself) is generated procedurally by the scripts in `scene/`.
