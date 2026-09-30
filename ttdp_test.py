# -*- coding: utf-8 -*-
"""
ttdp_test.py
============
ตัวอย่างข้อมูลสมมติ (1 วัน) สำหรับลองรันโมเดล ttdp_model.py
รัน:  python ttdp_test.py
"""

from __future__ import annotations
import itertools

from ttdp_model import TTDPData, TTDPModel

# ---- 1) ข้อมูลสถานที่ (สมมติ) ------------------------------------------------
hotel = "Hotel"
coords = {                       # พิกัด (x, y) สมมติ ใช้คำนวณระยะทาง
    "Hotel": (0, 0), "Temple": (1, 1), "Museum": (2, 1),
    "Market": (3, 0), "Restaurant": (2, -1), "Viewpoint": (1, -1),
}
places = ["Temple", "Museum", "Market", "Restaurant", "Viewpoint"]
A = places
F = ["Restaurant"]              # ร้านอาหาร
M = ["Temple"]                  # สถานที่บังคับต้องไป

score = {"Temple": 9, "Museum": 7, "Market": 6, "Restaurant": 5, "Viewpoint": 8}
visit_duration = {"Temple": 1.0, "Museum": 1.5, "Market": 1.0, "Restaurant": 1.0, "Viewpoint": 0.5}
entrance_fee = {"Temple": 50, "Museum": 100, "Market": 0, "Restaurant": 150, "Viewpoint": 0}
pref_start = {"Temple": 8.0, "Museum": 8.0, "Market": 8.0, "Restaurant": 11.0, "Viewpoint": 8.0}
pref_end = {"Temple": 18.0, "Museum": 18.0, "Market": 20.0, "Restaurant": 14.0, "Viewpoint": 18.0}
open_time = {"Temple": 8.0, "Museum": 9.0, "Market": 10.0, "Restaurant": 10.0, "Viewpoint": 6.0}
close_time = {"Temple": 18.0, "Museum": 17.0, "Market": 22.0, "Restaurant": 22.0, "Viewpoint": 20.0}


def _distance_km(a: str, b: str) -> float:
    (x1, y1), (x2, y2) = coords[a], coords[b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 5   # 1 หน่วยพิกัด = 5 กม.


# ---- 2) ตารางระยะทาง/เวลาเดินทางทุกคู่ ------------------------------------------
travel_dist, travel_time = {}, {}
for i, j in itertools.permutations([hotel] + places, 2):
    km = _distance_km(i, j)
    travel_dist[(i, j)] = round(km, 2)
    travel_time[(i, j)] = round(km / 20, 2)                # ความเร็วเฉลี่ย 20 กม./ชม.

# ---- 3) รวมเป็นก้อนข้อมูลเดียว -------------------------------------------------
data = TTDPData(
    A=A, F=F, M=M, K=[1], hotel=hotel,
    t=travel_time, dist=travel_dist,
    FC=8.0, P_fuel=32.0,          # 8 ลิตรต่อ 100 กม., น้ำมันลิตรละ 32 บาท
    v=visit_duration, s=score, c_visit=entrance_fee,
    B=1500, T=10.0,
    E=pref_start, L=pref_end, OP=open_time, CL=close_time,
    r={1: 1},                     # 1 = กลับโรงแรม (เส้นทางปิด)
    start_time=8.0,
)


def show(solution: dict):
    """พิมพ์ผลลัพธ์ให้อ่านง่าย"""
    if not solution["success"]:
        print("ไม่พบคำตอบ:", solution["message"])
        return
    print(f"คะแนนความพึงพอใจรวม (Z) = {solution['objective']:.1f}")
    print(f"ค่าใช้จ่ายรวม (เข้าชม + น้ำมัน) = {solution['total_cost']:.2f} บาท")
    for k, day in solution["days"].items():
        print(f"\n-- วันที่ {k} --")
        print("เส้นทาง:", " -> ".join(day["route"]))
        print("เวลาไปถึงแต่ละจุด:", day["arrival_times"])
        print(f"กลับถึงเวลา {day['end_time']} (ใช้เวลาทั้งวัน {day['day_length']} ชม.)")
        print("คะแนนวันนี้:", day["score"])


if __name__ == "__main__":
    model = TTDPModel(data)
    show(model.decode(model.solve()))
