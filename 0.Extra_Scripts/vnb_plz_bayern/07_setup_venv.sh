#!/bin/bash
# pip in venv bootstrappen und Geo-Pakete installieren
set -e
rm -rf /home/pedro/vnbenv
python3 -m venv --without-pip /home/pedro/vnbenv
curl -sS -m 300 -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
/home/pedro/vnbenv/bin/python /tmp/get-pip.py -q
/home/pedro/vnbenv/bin/pip install -q pyshp shapely pyproj
/home/pedro/vnbenv/bin/python -c "import shapefile, shapely; print('ok', shapely.__version__)"

