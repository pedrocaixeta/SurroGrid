#!/bin/bash
cd "$(dirname "$0")"
export UNITS=EinheitenSolar,EinheitenStromSpeicher,EinheitenWind,EinheitenBiomasse,EinheitenWasser,EinheitenVerbrennung,EinheitenStromVerbraucher
exec python3 -u 05_extract.py > data/extract.log 2>&1
