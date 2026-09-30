# -*- coding: utf-8 -*-
"""
ttdp_test.py
============
ตัวอย่างข้อมูล (สถานที่, คะแนน, พิกัด ฯลฯ) สำหรับทดสอบ ttdp_model.py

วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ ttdp_model.py แล้วรันได้เลย (กด F5 ใน VS Code
หรือรัน `python ttdp_test.py` ใน terminal)
"""

from __future__ import annotations
import itertools

from ttdp_model import TTDPData, TTDPModel


# ---------------------------------------------------------------------------
# ข้อมูลตัวอย่าง — 1 วัน, โรงแรม + สถานที่ 5 แห่ง + ร้านอาหาร 1 แห่ง
# ---------------------------------------------------------------------------
hotel = "Hotel"

# พิกัด (x, y) ของแต่ละสถานที่ (สมมติขึ้น ใช้คำนวณระยะทาง)
coords = {
    "Hotel":      (0, 0),
    "Temple":     (1, 1),
    "Museum":     (2, 1),
    "Market":     (3, 0),
    "Restaurant": (2, -1),
    "Viewpoint":  (1, -1),
}

places = ["Temple", "Museum", "Market", "Restaurant", "Viewpoint"]
A = places
F = ["Restaurant"]      # ร้านอาหาร
M = ["Temple"]          # สถานที่บังคับต้องไป

# คะแนนความพึงพอใจของแต่ละสถานที่
score = {
    "Temple": 9, "Museum": 7, "Market": 6,
    "Restaurant": 5, "Viewpoint": 8,
}

# ระยะเวลาที่ใช้เยี่ยมชมแต่ละที่ (ชั่วโมง)
visit_duration = {
    "Temple": 1.0, "Museum": 1.5, "Market": 1.0,
    "Restaurant": 1.0, "Viewpoint": 0.5,
}

# ค่าเข้าชม (บาท)
entrance_fee = {
    "Temple": 50, "Museum": 100, "Market": 0,
    "Restaurant": 150, "Viewpoint": 0,
}

# ช่วงเวลาที่อยากเข้าชม (ชั่วโมง, ระบบ 24 ชม.)
pref_start = {"Temple": 8.0, "Museum": 8.0, "Market": 8.0, "Restaurant": 11.0, "Viewpoint": 8.0}
pref_end   = {"Temple": 18.0, "Museum": 18.0, "Market": 20.0, "Restaurant": 14.0, "Viewpoint": 18.0}

# เวลาเปิด-ปิดของสถานที่
open_time  = {"Temple": 8.0, "Museum": 9.0, "Market": 10.0, "Restaurant": 10.0, "Viewpoint": 6.0}
close_time = {"Temple": 18.0, "Museum": 17.0, "Market": 22.0, "Restaurant": 22.0, "Viewpoint": 20.0}


def _distance_km(a: str, b: str) -> float:
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 5  # scale ให้เป็นกิโลเมตร


# สร้าง matrix ระยะทาง (km) และเวลาเดินทาง (ชั่วโมง) ระหว่างทุกคู่สถานที่
nodes = [hotel] + places
travel_dist = {}
travel_time = {}
for i, j in itertools.permutations(nodes, 2):
    km = _distance_km(i, j)
    travel_dist[(i, j)] = round(km, 2)
    travel_time[(i, j)] = round(km / 20, 2)   # ความเร็วเฉลี่ย 20 กม./ชม.


data = TTDPData(
    A=A, F=F, M=M, K=[1], hotel=hotel,
    t=travel_time,
    dist=travel_dist, FC=8.0, P_fuel=32.0,   # พารามิเตอร์ค่าน้ำมัน
    v=visit_duration,
    s=score,
    c_visit=entrance_fee,
    B=1500,            # งบประมาณรวม (บาท)
    T=10.0,             # เวลาเที่ยวสูงสุดต่อวัน (ชั่วโมง)
    E=pref_start,
    L=pref_end,
    OP=open_time,
    CL=close_time,
    r={1: 1},           # closed route: กลับโรงแรม
    start_time=8.0,
)


if __name__ == "__main__":
    model = TTDPModel(data)
    result = model.solve()
    solution = model.decode(result)

    if not solution["success"]:
        print("ไม่พบคำตอบ:", solution["message"])
    else:
        print(f"คะแนนความพึงพอใจรวม (Z) = {solution['objective']:.1f}")
        print(f"ค่าใช้จ่ายรวม (รวมค่าน้ำมัน) = {solution['total_cost']:.2f} บาท")
        for k, day in solution["days"].items():
            print(f"\n-- วันที่ {k} --")
            print("เส้นทาง:", " -> ".join(day["route"]))
            print("เวลาไปถึงแต่ละจุด:", day["arrival_times"])
            print("คะแนนวันนี้:", day["score"])