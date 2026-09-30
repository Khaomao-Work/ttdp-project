# -*- coding: utf-8 -*-
"""
check_fixes.py
==============
ชุดตรวจอัตโนมัติว่าการแก้ไขทำงานถูกต้อง  รัน:  python check_fixes.py
ทุกข้อต้องขึ้น PASS (ถ้ามี FAIL แปลว่ามีอะไรพัง)
"""

import itertools
import sys

from ttdp_model import TTDPData, TTDPModel

HOTEL = "Hotel"
COORDS = {"Hotel": (0, 0), "Temple": (1, 1), "Museum": (2, 1), "Market": (3, 0),
          "Restaurant": (2, -1), "Cafe": (0, 2), "Viewpoint": (1, -1)}


def make_data(K, r, restaurants=("Restaurant",), drop_pair=None, T=10.0):
    """สร้างข้อมูลทดสอบ ปรับจำนวนวัน/ร้านอาหาร/ตัดข้อมูลบางคู่ออกได้"""
    places = ["Temple", "Museum", "Market", "Viewpoint"] + list(restaurants)
    nodes = [HOTEL] + places
    t, dist = {}, {}
    for i, j in itertools.permutations(nodes, 2):
        (x1, y1), (x2, y2) = COORDS[i], COORDS[j]
        km = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 * 5
        dist[(i, j)] = km
        t[(i, j)] = km / 20
    if drop_pair:
        for key in (drop_pair, drop_pair[::-1]):
            t.pop(key, None)
    sc = {"Temple": 9, "Museum": 7, "Market": 6, "Viewpoint": 8, "Restaurant": 5, "Cafe": 4}
    return TTDPData(
        A=places, F=list(restaurants), M=["Temple"], K=K, hotel=HOTEL,
        t=t, dist=dist, FC=8.0, P_fuel=32.0,
        v={p: (0.5 if p == "Viewpoint" else 1.0) for p in places},
        s={p: sc[p] for p in places},
        c_visit={p: 0 for p in places}, B=5000, T=T,
        E={"Restaurant": 11.0, "Cafe": 11.0}, L={"Restaurant": 14.0, "Cafe": 14.0},
        OP={"Museum": 9.0}, CL={"Museum": 17.0},
        r=r, start_time=8.0,
    )


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), "-", name, detail)


# 1) เวลารอต้องถูกนับ: วันหนึ่งต้องไม่ยาวเกิน T  [FIX 11]
data = make_data([1], {1: 1})
sol = TTDPModel(data).decode(TTDPModel(data).solve())
check("วันยาวไม่เกิน T (นับเวลารอแล้ว)",
      sol["success"] and sol["days"][1]["day_length"] <= data.T + 1e-6,
      f"(ใช้ {sol['days'][1]['day_length']} ชม. จาก T={data.T})")

# 1b) ลอง T เข้มขึ้นก็ยังต้องไม่เกิน
data = make_data([1], {1: 1}, T=7.0)
m = TTDPModel(data)
sol = m.decode(m.solve())
check("T=7 ก็ยังไม่เกิน", sol["success"] and sol["days"][1]["day_length"] <= 7.0 + 1e-6,
      f"(ใช้ {sol['days'][1]['day_length'] if sol['success'] else '-'} ชม.)")

# 2) หลายวันต้องไม่เที่ยวที่เดิมซ้ำ  [FIX 12]
data = make_data([1, 2], {1: 1, 2: 1}, restaurants=("Restaurant", "Cafe"))
m = TTDPModel(data)
sol = m.decode(m.solve())
seen = []
if sol["success"]:
    for day in sol["days"].values():
        seen += [p for p in day["route"] if p in data.A]
check("2 วัน ไม่มีสถานที่ซ้ำ", sol["success"] and len(seen) == len(set(seen)), f"(เที่ยว {seen})")

# 3) ข้อมูลขาดคู่ ต้อง error  [FIX 14]
try:
    TTDPModel(make_data([1], {1: 1}, drop_pair=("Temple", "Market")))
    check("ขาดข้อมูลคู่ -> error", False)
except ValueError:
    check("ขาดข้อมูลคู่ -> error", True)

# 3b) ร้านอาหารน้อยกว่าจำนวนวัน ต้อง error  [FIX 12/validate]
try:
    TTDPModel(make_data([1, 2], {1: 1, 2: 1}))
    check("ร้านอาหารไม่พอต่อจำนวนวัน -> error", False)
except ValueError:
    check("ร้านอาหารไม่พอต่อจำนวนวัน -> error", True)

# 4) เส้นทางเปิด r=0 ต้องแก้ได้ และไม่กลับโรงแรม  [FIX 3, 15]
data = make_data([1], {1: 0})
m = TTDPModel(data)
sol = m.decode(m.solve())
ok = sol["success"] and sol["days"][1]["route"][-1] == "(สิ้นสุดเส้นทาง)" \
    and HOTEL not in sol["days"][1]["route"][1:]
check("เส้นทางเปิด (r=0)", ok, f"({' -> '.join(sol['days'][1]['route']) if sol['success'] else ''})")

# 5) สูตรน้ำมัน: 10 กม. ที่ 8 ลิตร/100 กม. ราคา 32 บาท = 25.6 บาท  [FIX 13]
d = make_data([1], {1: 1})
d.dist[("Hotel", "Temple")] = 10.0
check("ค่าน้ำมัน 10 กม. = 25.6 บาท", abs(d.fuel_cost("Hotel", "Temple") - 25.6) < 1e-9)

print("\nสรุป:", "ผ่านทั้งหมด" if all(results) else "มีข้อที่ไม่ผ่าน")
sys.exit(0 if all(results) else 1)
