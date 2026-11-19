"""Bayerische PLZ aus PLZ-Polygonen (plz-5stellig.shp, WGS84) und VG250-Gemeinden (Bayern = SN_L 09)."""
import csv
import os
import shapefile
import shapely
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform, unary_union

IN = "/home/pedro/Linux-AntigravityProjects/SurroGrid-linux/GridExpand/1.grid_sampling/gridreadout/input_data"
HERE = os.path.dirname(os.path.abspath(__file__))

g = shapefile.Reader(f"{IN}/VG250_GEM.shp", encoding="utf-8")
fn = [f[0] for f in g.fields[1:]]
print("VG250 Felder:", fn)
i_sn = fn.index("SN_L"); i_gf = fn.index("GF")
by = []
for sr in g.iterShapeRecords():
    if str(sr.record[i_sn]) == "09" and sr.record[i_gf] in (4, "4"):
        by.append(shape(sr.shape.__geo_interface__))
bayern = shapely.make_valid(unary_union(by))
print("Bayern Flaeche km2:", bayern.area / 1e6)

p = shapefile.Reader(f"{IN}/plz-5stellig.shp", encoding="utf-8")
pf = [f[0] for f in p.fields[1:]]
print("PLZ Felder:", pf)
i_plz = pf.index("plz")
tr = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True).transform
prep = shapely.prepared.prep(bayern)
rows = []
for sr in p.iterShapeRecords():
    geom = shapely.make_valid(transform(tr, shape(sr.shape.__geo_interface__)))
    if not prep.intersects(geom):
        continue
    inter = geom.intersection(bayern).area
    rows.append((str(sr.record[i_plz]).zfill(5), inter / geom.area if geom.area else 0, geom.area / 1e6))
print("PLZ mit Bayern-Ueberlappung:", len(rows))
with open(os.path.join(HERE, "data", "plz_bayern_overlap.tsv"), "w") as f:
    f.write("plz\tanteil_in_bayern\tflaeche_km2\n")
    for plz, fr, a in sorted(rows):
        f.write(f"{plz}\t{fr:.4f}\t{a:.2f}\n")
print("ge 0.5:", sum(1 for r in rows if r[1] >= 0.5))

