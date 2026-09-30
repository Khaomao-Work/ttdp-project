# -*- coding: utf-8 -*-
"""
geoapify_fetch.py
=================
ดึงข้อมูลจริงจาก Geoapify แล้วสร้างไฟล์ places.csv + matrix.csv
ให้ตรงกับรูปแบบที่ TTDPData.from_csv() ใน ttdp_model.py อ่านได้

ขั้นตอน
  1) geocode()       : ชื่อโรงแรม/ที่อยู่  -> พิกัด (lon, lat)
  2) search_places() : พิกัด + หมวดหมู่ + รัศมี -> รายชื่อสถานที่ (พร้อมพิกัด)
  3) route_matrix()  : พิกัดทุกจุด -> ระยะทาง (กม.) และเวลาเดินทาง (ชม.) ทุกคู่
  4) เขียน CSV

ตั้งค่า API key (อย่าเขียน key ลงในโค้ด):
  Windows (PowerShell):  $env:GEOAPIFY_API_KEY="xxxx"
  Mac/Linux:             export GEOAPIFY_API_KEY="xxxx"

ติดตั้งไลบรารี:  pip install requests

หมายเหตุ: Geoapify ใช้ลำดับพิกัดเป็น [lon, lat] (ลองจิจูดก่อนละติจูด)
"""

from __future__ import annotations
import csv
import os
from typing import Dict, List, Tuple

import requests

API_KEY = os.environ.get("GEOAPIFY_API_KEY", "")
BASE = "https://api.geoapify.com"
TIMEOUT = 30


def _check_key():
    if not API_KEY:
        raise RuntimeError("ยังไม่ได้ตั้งค่า GEOAPIFY_API_KEY")


# ---------------------------------------------------------------------------
# 1) Geocoding: ข้อความ -> พิกัด
# ---------------------------------------------------------------------------
def geocode(text: str, country: str = "th") -> Tuple[float, float]:
    _check_key()
    r = requests.get(
        f"{BASE}/v1/geocode/search",
        params={
            "text": text,
            "lang": "th",
            "filter": f"countrycode:{country}",
            "limit": 1,
            "format": "json",
            "apiKey": API_KEY,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"หาพิกัดของ '{text}' ไม่เจอ")
    return results[0]["lon"], results[0]["lat"]


# ---------------------------------------------------------------------------
# 2) Places API: หาสถานที่รอบ ๆ จุดศูนย์กลาง
# ---------------------------------------------------------------------------
def search_places(
    categories: str, lon: float, lat: float, radius_m: int = 15000, limit: int = 10
) -> List[Dict]:
    """categories เช่น 'tourism.attraction' หรือ 'catering.restaurant'
    (ดูรายการหมวดทั้งหมดในเอกสาร Geoapify Places API)"""
    _check_key()
    r = requests.get(
        f"{BASE}/v2/places",
        params={
            "categories": categories,
            "filter": f"circle:{lon},{lat},{radius_m}",
            "bias": f"proximity:{lon},{lat}",
            "limit": limit,
            "lang": "th",
            "apiKey": API_KEY,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    out = []
    for feat in r.json().get("features", []):
        p = feat["properties"]
        name = (p.get("name") or "").strip()
        if not name:          # ข้ามสถานที่ที่ไม่มีชื่อ
            continue
        out.append({"name": name, "lon": p["lon"], "lat": p["lat"]})
    return out


# ---------------------------------------------------------------------------
# 3) Route Matrix API: ระยะทาง/เวลา ทุกคู่ในคำขอเดียว
# ---------------------------------------------------------------------------
def route_matrix(coords: List[Tuple[float, float]], mode: str = "drive"):
    """coords = [(lon, lat), ...] คืนค่า matrix[i][j] = {'distance': เมตร, 'time': วินาที}"""
    _check_key()
    locs = [{"location": [lon, lat]} for lon, lat in coords]
    r = requests.post(
        f"{BASE}/v1/routematrix",
        params={"apiKey": API_KEY},
        json={"mode": mode, "sources": locs, "targets": locs},
        timeout=TIMEOUT * 2,
    )
    r.raise_for_status()
    return r.json()["sources_to_targets"]


# ---------------------------------------------------------------------------
# 4) รวมทั้งหมด -> เขียน places.csv / matrix.csv
# ---------------------------------------------------------------------------
def build_csvs(
    hotel_text: str,
    n_attractions: int = 8,
    n_restaurants: int = 4,
    radius_m: int = 15000,
    places_csv: str = "places.csv",
    matrix_csv: str = "matrix.csv",
):
    # 1) พิกัดโรงแรม
    h_lon, h_lat = geocode(hotel_text)

    # 2) สถานที่ท่องเที่ยว + ร้านอาหาร
    attractions = search_places("tourism.attraction", h_lon, h_lat, radius_m, n_attractions)
    restaurants = search_places("catering.restaurant", h_lon, h_lat, radius_m, n_restaurants)

    # รวมเป็นรายการเดียว (ชื่อต้องไม่ซ้ำกัน เพราะโมเดลใช้ชื่อเป็นคีย์)
    rows: List[Dict] = [
        {"name": "Hotel", "category": "hotel", "lon": h_lon, "lat": h_lat}
    ]
    seen = {"Hotel"}
    for cat, items in (("attraction", attractions), ("restaurant", restaurants)):
        for it in items:
            name = it["name"]
            if name in seen:
                continue
            seen.add(name)
            rows.append({"name": name, "category": cat, "lon": it["lon"], "lat": it["lat"]})

    # 3) matrix ระยะทาง/เวลา
    coords = [(r["lon"], r["lat"]) for r in rows]
    m = route_matrix(coords)

    # 4a) places.csv — ค่า default ไว้ก่อน ต้องแก้เองใน Excel:
    #     score, visit_minutes, entrance_fee, open/close, pref_start/end
    with open(places_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "category", "visit_minutes", "score", "entrance_fee",
                    "open_time", "close_time", "pref_start", "pref_end"])
        for r in rows:
            if r["category"] == "hotel":
                w.writerow([r["name"], "hotel", "", "", "", "", "", "", ""])
            else:
                w.writerow([r["name"], r["category"], 60, 5, 0, "", "", "", ""])

    # 4b) matrix.csv — แปลงหน่วย: เมตร->กม., วินาที->ชั่วโมง
    with open(matrix_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to", "distance_km", "time_hours", "toll", "parking"])
        for i, ri in enumerate(rows):
            for j, rj in enumerate(rows):
                if i == j:
                    continue
                cell = m[i][j]
                if not cell or cell.get("distance") is None or cell.get("time") is None:
                    print(f"เตือน: หาเส้นทาง {ri['name']} -> {rj['name']} ไม่ได้ (ข้ามคู่นี้)")
                    continue
                w.writerow([
                    ri["name"], rj["name"],
                    round(cell["distance"] / 1000.0, 3),
                    round(cell["time"] / 3600.0, 4),
                    0, 0,
                ])

    print(f"สร้าง {places_csv} ({len(rows)} แห่ง) และ {matrix_csv} เรียบร้อย")
    print("ขั้นต่อไป: เปิด places.csv แล้วกรอก score / เวลาเปิด-ปิด / ค่าเข้าชม ก่อนรันโมเดล")


if __name__ == "__main__":
    build_csvs(hotel_text="ใส่ชื่อโรงแรมหรือที่อยู่ของคุณ, จังหวัด")