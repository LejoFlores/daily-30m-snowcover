"""Domain settings, chosen by env SNOW_DOMAIN (default dcew)."""
import os
DOMAINS = {
    "dcew": dict(X0=564480, Y1=4849500, X1=577020, Y0=4836000, LAT=43.72,
                 MODIS_RC=(1478, 1400), MODIS_SHAPE=(54, 120), VIIRS_RC=(1847, 1750), VIIRS_SHAPE=(67, 150),
                 REGIONS={1: "dry_creek", 2: "bogus"}),
    "rcew": dict(X0=510000, Y1=4797000, X1=526800, Y0=4766100, LAT=43.18,
                 MODIS_RC=(1591, 1075), MODIS_SHAPE=(89, 179), VIIRS_RC=(1988, 1344), VIIRS_SHAPE=(112, 224),
                 REGIONS={1: "reynolds_creek"}),
}
NAME = os.environ.get("SNOW_DOMAIN", "dcew")
D = DOMAINS[NAME]
X0, Y1, X1, Y0, LAT = D["X0"], D["Y1"], D["X1"], D["Y0"], D["LAT"]
RES = 30
NX, NY = (X1 - X0) // RES, (Y1 - Y0) // RES


def region_name(code):
    return D["REGIONS"].get(int(code), "")
