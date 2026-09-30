# -*- coding: utf-8 -*-
"""
ttdp_model.py
=============
ตัวโมเดลหลักของโครงงาน: Tourist Trip Design Problem (TTDP) / Orienteering Problem
เขียนเป็น Binary Integer Program (BIP) แล้วให้ตัวแก้ปัญหา HiGHS (ผ่าน scipy.optimize.milp) หาคำตอบ

ไฟล์นี้มีแต่ "ชิ้นส่วน" ที่เอาไปใช้ซ้ำได้ ไม่พิมพ์อะไรออกจอเอง
วิธีใช้ (จากไฟล์อื่น):
    from ttdp_model import TTDPData, TTDPModel
    model  = TTDPModel(data)        # สร้างโมเดลจากข้อมูล
    result = model.solve()          # ให้ solver หาคำตอบ
    answer = model.decode(result)   # แปลงคำตอบเป็นเส้นทางที่อ่านง่าย

ประวัติการแก้ไข (FIX)
  FIX 1  เชื่อมเวลาไปถึง (a_ik) กับเส้นทาง (x_ijk) ด้วย Big-M
  FIX 2  เงื่อนไขช่วงเวลา (8)(9)(11) ผูกกับ y_ik (สถานที่ที่ไม่ไป ไม่ต้องเช็กเวลา)
  FIX 3  เส้นทางเปิด (r = 0) ใช้จุดปลายสมมติ DUMMY
  FIX 4  ไม่นับเวลามื้ออาหารซ้ำ (ตัด R ออกจาก (7))
  FIX 5  ใส่ขอบเขตให้ u_ik, a_ik
  FIX 6  งบประมาณ (6') รวมทั้งค่าเข้าชม + ค่าเดินทาง/น้ำมัน
  FIX 7  สูตรค่าน้ำมัน (12) อยู่ใน fuel_cost()
  FIX 8  from_csv() โหลดข้อมูลจริง (เช่นที่ได้จาก Geoapify)
  FIX 9  meal_window แบบตายตัวปิดไว้เป็นค่าเริ่มต้น
  ---- รอบแก้ล่าสุด ----
  FIX 11 [สำคัญ] เพิ่มตัวแปร a_end = เวลากลับถึงจุดสิ้นสุดของวัน แล้วบังคับ
         a_end - start_time <= T  -> "เวลารอ" ถูกนับรวมแล้ว (เดิมสมการ (7) ไม่นับเวลารอ
         ทำให้วันจริงอาจยาวเกิน T)
  FIX 12 [สำคัญ] เพิ่มสมการ sum_k y_ik <= 1 : สถานที่หนึ่งแห่งเที่ยวได้ไม่เกิน 1 ครั้งตลอดทริป
         (เดิมหลายวันจะเที่ยวที่เดิมซ้ำทุกวันและนับคะแนนซ้ำ)
  FIX 13 FC เปลี่ยนหน่วยเป็น "ลิตรต่อ 100 กม." และสูตรเป็น d * FC/100 * P_fuel
         (เดิมตีความเป็น 8 ลิตรต่อ 15 กม. ซึ่งแพงเกินจริงประมาณ 6 เท่า)
  FIX 14 ข้อมูลระยะเวลา/ระยะทางขาดคู่ใด -> error ทันที (เดิมมองเป็น 0 เงียบ ๆ
         ทำให้ solver "เดินทางฟรี" ระหว่างคู่ที่ไม่มีข้อมูล)
  FIX 15 เส้นทางเปิด (r = 0) ไม่สร้างเส้นเข้าโรงแรมอีก (สะอาดขึ้น ไม่เปลี่ยนคำตอบ)
  FIX 16 Big-M ของเวลาปรับให้เป็น 24 + v_i + t_ij ต่อเส้นทาง (ถูกต้องตามทฤษฎีกว่า 24 ตายตัว)
"""

from __future__ import annotations
import csv
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


# ค่าคงที่ ------------------------------------------------------------------
BIG_M_TIME = 24.0        # จำนวนชั่วโมงในหนึ่งวัน ใช้เป็นฐานของ Big-M ด้านเวลา
DUMMY = "__DUMMY_END__"  # จุดปลายสมมติ ใช้เมื่อเส้นทางเป็นแบบเปิด (ไม่กลับโรงแรม)


# ===========================================================================
# ส่วนที่ 1: เก็บข้อมูล (เซต + พารามิเตอร์ทั้งหมดของโมเดล)
# ===========================================================================
@dataclass
class TTDPData:
    """กล่องเก็บข้อมูลทั้งหมด: ชื่อตัวแปรในโค้ด <-> สัญลักษณ์ในเอกสาร"""

    # ---- เซต (Sets) ----
    A: List[str]                      # สถานที่ท่องเที่ยว + ร้านอาหาร
    F: List[str]                      # ร้านอาหาร (F เป็นส่วนหนึ่งของ A)
    M: List[str]                      # สถานที่บังคับต้องไป (M เป็นส่วนหนึ่งของ A)
    K: List[int]                      # วันของทริป เช่น [1, 2, 3]
    hotel: str                        # ชื่อโรงแรม (จุดเริ่ม/สิ้นสุดของทุกวัน)

    # ---- พารามิเตอร์การเดินทางและสถานที่ ----
    t: Dict[Tuple[str, str], float] = field(default_factory=dict)   # t_ij เวลาเดินทาง (ชม.)
    v: Dict[str, float] = field(default_factory=dict)               # v_i เวลาเยี่ยมชม (ชม.)
    s: Dict[str, float] = field(default_factory=dict)               # s_i คะแนนความพึงพอใจ
    c_visit: Dict[str, float] = field(default_factory=dict)         # c_i ค่าเข้าชม (บาท)
    B: float = 0.0                    # งบรวมทั้งทริป (บาท)
    T: float = 0.0                    # เวลาสูงสุดต่อวัน (ชม.) นับตั้งแต่ออกจนกลับถึง
    E: Dict[str, float] = field(default_factory=dict)               # E_i เริ่มช่วงที่อยากไป
    L: Dict[str, float] = field(default_factory=dict)               # L_i สิ้นสุดช่วงที่อยากไป
    OP: Dict[str, float] = field(default_factory=dict)              # OP_i เวลาเปิด
    CL: Dict[str, float] = field(default_factory=dict)              # CL_i เวลาปิด
    r: Dict[int, int] = field(default_factory=dict)                 # r ของแต่ละวัน: 1 = กลับโรงแรม, 0 = ไม่กลับ
    start_time: float = 8.0           # เวลาออกจากโรงแรมทุกวัน (ชม. ระบบ 24 ชม.)

    # ---- พารามิเตอร์ค่าน้ำมัน สมการ (12) ----
    dist: Dict[Tuple[str, str], float] = field(default_factory=dict)  # d_ij ระยะทาง (กม.)
    FC: float = 8.0                   # อัตราสิ้นเปลือง: ลิตรต่อ 100 กม.  [FIX 13]
    P_fuel: float = 32.0              # ราคาน้ำมัน บาทต่อลิตร
    toll: Dict[Tuple[str, str], float] = field(default_factory=dict)      # ค่าทางด่วน (ไม่บังคับ)
    parking: Dict[Tuple[str, str], float] = field(default_factory=dict)   # ค่าจอดรถ (ไม่บังคับ)
    c_travel_override: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # ^ ถ้าใส่ค่าใช้จ่ายเดินทางของคู่ไหนไว้ที่นี่ จะใช้ค่านี้แทนสูตรน้ำมัน

    # ---- ตัวเลือกช่วงเวลามื้ออาหารแบบตายตัว (สมการ 11) ----
    use_meal_window: bool = False
    meal_window: Tuple[float, float] = (11.5, 13.5)

    # ---------------------------------------------------------------- ตัวช่วย --
    @property
    def N(self) -> List[str]:
        """เซต N = โรงแรม + สถานที่ทั้งหมด"""
        return [self.hotel] + list(self.A)

    @staticmethod
    def _pair(table: Dict, i: str, j: str):
        """หาค่าของคู่ (i, j) ถ้าไม่มีลองหา (j, i) ถ้าไม่มีทั้งสองทิศ คืน None"""
        if (i, j) in table:
            return table[(i, j)]
        if (j, i) in table:
            return table[(j, i)]
        return None

    def travel_time(self, i: str, j: str) -> float:
        """t_ij: เวลาเดินทางจาก i ไป j (ไปจุด DUMMY = 0)  [FIX 14: ไม่มีข้อมูล -> error]"""
        if i == j or i == DUMMY or j == DUMMY:
            return 0.0
        val = self._pair(self.t, i, j)
        if val is None:
            raise KeyError(f"ไม่มีข้อมูลเวลาเดินทางของคู่ ({i}, {j})")
        return val

    def fuel_cost(self, i: str, j: str) -> float:
        """สมการ (12): ค่าน้ำมัน = d_ij * (FC/100) * P_fuel  [FIX 13]"""
        if i == j or i == DUMMY or j == DUMMY:
            return 0.0
        d_ij = self._pair(self.dist, i, j)
        if d_ij is None:
            raise KeyError(f"ไม่มีข้อมูลระยะทางของคู่ ({i}, {j})")
        return d_ij * (self.FC / 100.0) * self.P_fuel

    def travel_cost(self, i: str, j: str) -> float:
        """ค่าเดินทางรวมของคู่ (i, j) = ค่าน้ำมัน + ทางด่วน + ที่จอด (ถ้ามี)
        ถ้าใส่ค่า override ไว้ จะใช้ค่านั้นก่อนเสมอ"""
        if i == j or i == DUMMY or j == DUMMY:
            return 0.0
        override = self._pair(self.c_travel_override, i, j)
        if override is not None:
            return override
        toll_ij = self._pair(self.toll, i, j) or 0.0
        parking_ij = self._pair(self.parking, i, j) or 0.0
        return self.fuel_cost(i, j) + toll_ij + parking_ij

    # ------------------------------------------------------------ [FIX 14] --
    def validate(self) -> None:
        """ตรวจข้อมูลก่อนสร้างโมเดล พบปัญหาใดให้ error พร้อมบอกว่าผิดตรงไหน"""
        problems: List[str] = []

        if self.hotel in self.A:
            problems.append(f"ชื่อโรงแรม '{self.hotel}' ซ้ำกับสถานที่ใน A")
        if len(set(self.A)) != len(self.A):
            problems.append("ชื่อสถานที่ใน A ซ้ำกัน (โมเดลใช้ชื่อเป็นคีย์ ต้องไม่ซ้ำ)")
        for label, group in (("F", self.F), ("M", self.M)):
            bad = [p for p in group if p not in self.A]
            if bad:
                problems.append(f"{label} มีสถานที่ที่ไม่อยู่ใน A: {bad}")
        for k in self.K:
            if k not in self.r:
                problems.append(f"ยังไม่ได้กำหนด r สำหรับวันที่ {k}")
        # สมการ (10) บังคับกินร้านอาหาร 1 แห่ง/วัน และ FIX 12 ห้ามซ้ำ -> ต้องมีร้านอย่างน้อยเท่าจำนวนวัน
        if self.F and len(self.F) < len(self.K):
            problems.append(
                f"มีร้านอาหาร {len(self.F)} แห่ง แต่ทริป {len(self.K)} วัน "
                f"(ต้องกินร้านต่างกันวันละแห่ง) -> เพิ่มร้านอาหารหรือลดจำนวนวัน"
            )

        # ต้องมีเวลา (และระยะทาง ถ้าไม่ได้ override) ครบทุกคู่
        miss_t, miss_d = [], []
        for i, j in itertools.combinations(self.N, 2):
            if self._pair(self.t, i, j) is None:
                miss_t.append((i, j))
            if (self._pair(self.dist, i, j) is None
                    and self._pair(self.c_travel_override, i, j) is None):
                miss_d.append((i, j))
        if miss_t:
            problems.append(f"ขาดเวลาเดินทาง {len(miss_t)} คู่ เช่น {miss_t[:5]}")
        if miss_d:
            problems.append(f"ขาดระยะทาง {len(miss_d)} คู่ เช่น {miss_d[:5]}")

        if problems:
            raise ValueError("ข้อมูลไม่ครบ/ไม่ถูกต้อง:\n - " + "\n - ".join(problems))

    # ---------------------------------------------------------- [FIX 8] -----
    @classmethod
    def from_csv(
        cls,
        places_csv: str,
        matrix_csv: str,
        K: List[int],
        r: Dict[int, int],
        B: float,
        T: float,
        start_time: float = 8.0,
        FC: float = 8.0,
        P_fuel: float = 32.0,
        must_visit: Optional[List[str]] = None,
        use_meal_window: bool = False,
    ) -> "TTDPData":
        """สร้าง TTDPData จากไฟล์ CSV 2 ไฟล์ (เช่นที่ geoapify_fetch.py สร้างให้)

        places_csv (ต้องมีแถวหัวตาราง):
            name, category, visit_minutes, score, entrance_fee,
            open_time, close_time, pref_start, pref_end
          - category: hotel | attraction | restaurant  (ต้องมี hotel แถวเดียว)
          - เวลาเป็นชั่วโมงทศนิยม เช่น 11.5 = 11:30 เว้นว่าง = ไม่มีข้อจำกัด

        matrix_csv (ต้องมีแถวหัวตาราง):
            from, to, distance_km, time_hours, toll, parking
          - ต้องมีครบทุกคู่ (อย่างน้อยหนึ่งทิศ) ไม่ครบ = error  [FIX 14]
        """
        A, F, hotel_name = [], [], None
        v, s, c_visit, E, L, OP, CL = {}, {}, {}, {}, {}, {}, {}

        with open(places_csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                name = row["name"].strip()
                category = row["category"].strip().lower()

                def _num(key):
                    """อ่านช่องตัวเลข ช่องว่าง -> None"""
                    val = (row.get(key) or "").strip()
                    return float(val) if val != "" else None

                if category == "hotel":
                    if hotel_name is not None:
                        raise ValueError(
                            f"from_csv: พบโรงแรมมากกว่า 1 แถว ({hotel_name!r} และ {name!r})"
                        )
                    hotel_name = name
                    continue

                A.append(name)
                if category == "restaurant":
                    F.append(name)

                minutes = _num("visit_minutes")
                v[name] = (minutes / 60.0) if minutes else 0.0
                s[name] = _num("score") or 0.0
                c_visit[name] = _num("entrance_fee") or 0.0
                if _num("pref_start") is not None:
                    E[name] = _num("pref_start")
                if _num("pref_end") is not None:
                    L[name] = _num("pref_end")
                if _num("open_time") is not None:
                    OP[name] = _num("open_time")
                if _num("close_time") is not None:
                    CL[name] = _num("close_time")

        if hotel_name is None:
            raise ValueError("from_csv: places_csv ไม่มีแถวที่ category == 'hotel'")

        t, dist, toll, parking = {}, {}, {}, {}
        with open(matrix_csv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                i, j = row["from"].strip(), row["to"].strip()

                def _mnum(key):
                    val = (row.get(key) or "").strip()
                    return float(val) if val != "" else 0.0

                t[(i, j)] = _mnum("time_hours")
                dist[(i, j)] = _mnum("distance_km")
                toll[(i, j)] = _mnum("toll")
                parking[(i, j)] = _mnum("parking")

        return cls(
            A=A, F=F, M=list(must_visit or []), K=K, hotel=hotel_name,
            t=t, v=v, s=s, c_visit=c_visit, B=B, T=T, E=E, L=L, OP=OP, CL=CL,
            r=r, start_time=start_time,
            dist=dist, FC=FC, P_fuel=P_fuel, toll=toll, parking=parking,
            use_meal_window=use_meal_window,
        )


# ===========================================================================
# ส่วนที่ 2: ตัวช่วยสะสมสมการ (แปลงสมการเป็นตารางเมทริกซ์ให้ solver)
# ===========================================================================
class _RowBuilder:
    """เก็บสมการทีละแถวในรูป  lb <= (ผลรวมสัมประสิทธิ์ * ตัวแปร) <= ub"""

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self._rows: List[int] = []
        self._cols: List[int] = []
        self._data: List[float] = []
        self._lb: List[float] = []
        self._ub: List[float] = []
        self._nrow = 0

    def add(self, coeffs: Dict[int, float], lb: float, ub: float):
        """เพิ่มหนึ่งสมการ coeffs = {เลขตัวแปร: สัมประสิทธิ์}"""
        for col, val in coeffs.items():
            if val == 0.0:
                continue
            self._rows.append(self._nrow)
            self._cols.append(col)
            self._data.append(val)
        self._lb.append(lb)
        self._ub.append(ub)
        self._nrow += 1

    def add_eq(self, coeffs, value):      # สมการ "เท่ากับ"
        self.add(coeffs, value, value)

    def add_le(self, coeffs, upper):      # สมการ "น้อยกว่าหรือเท่ากับ"
        self.add(coeffs, -np.inf, upper)

    def add_ge(self, coeffs, lower):      # สมการ "มากกว่าหรือเท่ากับ"
        self.add(coeffs, lower, np.inf)

    def to_linear_constraint(self) -> LinearConstraint:
        A = coo_matrix(
            (self._data, (self._rows, self._cols)),
            shape=(self._nrow, self.n_vars),
        ).tocsr()
        return LinearConstraint(A, np.array(self._lb), np.array(self._ub))


# ===========================================================================
# ส่วนที่ 3: ตัวโมเดล (สร้างตัวแปร + สมการ + แก้ + แปลผล)
# ===========================================================================
class TTDPModel:
    """สร้างและแก้โมเดล BIP ของหนึ่งกรณีศึกษา"""

    def __init__(self, data: TTDPData, big_m_time: float = BIG_M_TIME):
        data.validate()                    # [FIX 14] ตรวจข้อมูลก่อนเสมอ
        self.d = data
        self.big_m = big_m_time
        self.var_index: Dict[Tuple, int] = {}   # ชื่อตัวแปร -> เลขคอลัมน์
        self.var_kind: List[str] = []           # 'x' | 'y' | 'u' | 'a' | 'aend'
        self.var_lb: List[float] = []
        self.var_ub: List[float] = []
        self.integrality: List[int] = []        # 1 = จำนวนเต็ม (ไบนารี), 0 = ต่อเนื่อง
        self._node_set_per_day: Dict[int, List[str]] = {}
        self._build_variables()

    # ------------------------------------------------------------ ตัวแปร ----
    def _new_var(self, key, kind: str, lb: float, ub: float, integer: bool):
        """ลงทะเบียนตัวแปรใหม่ คืนเลขคอลัมน์"""
        idx = len(self.var_kind)
        self.var_index[key] = idx
        self.var_kind.append(kind)
        self.var_lb.append(lb)
        self.var_ub.append(ub)
        self.integrality.append(1 if integer else 0)
        return idx

    def _build_variables(self):
        d = self.d
        for k in d.K:
            is_open = d.r[k] == 0
            nodes_k = [d.hotel] + list(d.A) + ([DUMMY] if is_open else [])
            self._node_set_per_day[k] = nodes_k

            # x_ijk: เดินทางจาก i ไป j ในวัน k หรือไม่ (0/1)
            for i, j in itertools.permutations(nodes_k, 2):
                if i == DUMMY:
                    continue                          # DUMMY ไม่มีเส้นทางออก
                if is_open and j == d.hotel:
                    continue                          # [FIX 15] เส้นทางเปิด: ไม่กลับเข้าโรงแรม
                if is_open and i == d.hotel and j == DUMMY:
                    continue                          # [FIX 15] โรงแรม -> DUMMY ไม่มีความหมาย
                self._new_var(("x", i, j, k), "x", 0, 1, True)

            # y_ik: เยี่ยมชมสถานที่ i ในวัน k หรือไม่ (0/1)
            for i in d.A:
                self._new_var(("y", i, k), "y", 0, 1, True)

            # u_ik: ลำดับการเดินทาง (ใช้กำจัด subtour แบบ MTZ) ค่า 1..|A|  [FIX 5]
            for i in d.A:
                self._new_var(("u", i, k), "u", 1, len(d.A), False)

            # a_ik: เวลาไปถึง (โรงแรมล็อกไว้ที่ start_time)  [FIX 5]
            for i in [d.hotel] + list(d.A):
                if i == d.hotel:
                    self._new_var(("a", i, k), "a", d.start_time, d.start_time, False)
                else:
                    self._new_var(("a", i, k), "a", 0.0, 24.0, False)

            # a_end_k: เวลากลับถึงจุดสิ้นสุดของวัน  [FIX 11]
            # เพดานบนคือ start_time + T ตรง ๆ = บังคับไม่ให้วันยาวเกิน T (รวมเวลารอ)
            self._new_var(("aend", k), "aend",
                          d.start_time, min(24.0, d.start_time + d.T), False)

    # ตัวช่วยเรียกหาเลขตัวแปร
    def x(self, i, j, k):
        return self.var_index.get(("x", i, j, k))   # อาจเป็น None ถ้าไม่มีเส้นนี้

    def y(self, i, k):
        return self.var_index[("y", i, k)]

    def u(self, i, k):
        return self.var_index[("u", i, k)]

    def a(self, i, k):
        return self.var_index[("a", i, k)]

    def aend(self, k):
        return self.var_index[("aend", k)]

    def _big_m_arc(self, i: str, j: str) -> float:
        """[FIX 16] Big-M ของเส้นทาง i->j = 24 + v_i + t_ij (ใหญ่พอให้สมการหลวมเมื่อไม่ได้ใช้เส้นนี้)"""
        return self.big_m + self.d.v.get(i, 0.0) + self.d.travel_time(i, j)

    # ------------------------------------------------------------- สมการ ----
    def build(self) -> Tuple[np.ndarray, LinearConstraint, Bounds, np.ndarray]:
        d = self.d
        n = len(self.var_kind)
        rb = _RowBuilder(n)

        # ---- ฟังก์ชันจุดประสงค์ (1): Maximize sum s_i*y_ik
        # solver ทำได้เฉพาะ minimize จึงใส่เครื่องหมายลบ
        c = np.zeros(n)
        for k in d.K:
            for i in d.A:
                c[self.y(i, k)] = -d.s.get(i, 0.0)

        for k in d.K:
            nodes_k = self._node_set_per_day[k]
            end_node_k = d.hotel if d.r[k] == 1 else DUMMY

            # (2)(3) Flow conservation: ถ้าไปสถานที่ i ต้องมีเส้นเข้า 1 เส้น และออก 1 เส้น
            for i in d.A:
                out_coeffs, in_coeffs = {}, {}
                for j in nodes_k:
                    if j == i:
                        continue
                    xij = self.x(i, j, k)
                    xji = self.x(j, i, k)
                    if xij is not None:
                        out_coeffs[xij] = out_coeffs.get(xij, 0) + 1
                    if xji is not None:
                        in_coeffs[xji] = in_coeffs.get(xji, 0) + 1
                out_coeffs[self.y(i, k)] = out_coeffs.get(self.y(i, k), 0) - 1
                in_coeffs[self.y(i, k)] = in_coeffs.get(self.y(i, k), 0) - 1
                rb.add_eq(out_coeffs, 0.0)   # (ออก) - y = 0
                rb.add_eq(in_coeffs, 0.0)    # (เข้า) - y = 0

            # (4) เริ่มต้น: ออกจากโรงแรมไปสถานที่แรกเพียง 1 เส้น
            start_coeffs = {}
            for j in d.A:
                xij = self.x(d.hotel, j, k)
                if xij is not None:
                    start_coeffs[xij] = start_coeffs.get(xij, 0) + 1
            rb.add_eq(start_coeffs, 1.0)

            # (5) สิ้นสุด [FIX 3]: เข้าจุดสิ้นสุดเพียง 1 เส้น (โรงแรมถ้าปิด / DUMMY ถ้าเปิด)
            end_coeffs = {}
            for i in d.A:
                xie = self.x(i, end_node_k, k)
                if xie is not None:
                    end_coeffs[xie] = end_coeffs.get(xie, 0) + 1
            rb.add_eq(end_coeffs, 1.0)

            # (7) เวลารวมต่อวัน: เดินทาง + เยี่ยมชม <= T  [FIX 4: ไม่บวก R ซ้ำ]
            # หมายเหตุ: สมการนี้ยังไม่นับเวลารอ -> ดู FIX 11 ด้านล่างที่ครอบคลุมกว่า
            time_coeffs = {}
            for i, j in itertools.permutations(nodes_k, 2):
                xij = self.x(i, j, k)
                if xij is None:
                    continue
                tt = d.travel_time(i, j)
                if tt:
                    time_coeffs[xij] = time_coeffs.get(xij, 0) + tt
            for i in d.A:
                yik = self.y(i, k)
                time_coeffs[yik] = time_coeffs.get(yik, 0) + d.v.get(i, 0.0)
            rb.add_le(time_coeffs, d.T)

            # [FIX 1][FIX 16] เชื่อมเวลาไปถึงกับเส้นทาง:
            #   ถ้าเดินทาง i -> j  แล้ว  a_j >= a_i + v_i + t_ij
            #   (Big-M ทำให้สมการ "หลวม" เมื่อไม่ได้ใช้เส้นทางนี้)
            real_nodes = [d.hotel] + list(d.A)
            for i, j in itertools.permutations(real_nodes, 2):
                if j == d.hotel:
                    continue                     # เวลากลับโรงแรมจัดการใน FIX 11
                xij = self.x(i, j, k)
                if xij is None:
                    continue
                m_ij = self._big_m_arc(i, j)
                rhs = d.v.get(i, 0.0) + d.travel_time(i, j) - m_ij
                rb.add_ge({self.a(j, k): 1, self.a(i, k): -1, xij: -m_ij}, rhs)

            # [FIX 11] เวลากลับถึงจุดสิ้นสุด: a_end >= a_i + v_i + t_i,end  เมื่อ i คือสถานที่สุดท้าย
            # ขอบบนของ a_end = start_time + T ถูกใส่ไว้ตอนสร้างตัวแปรแล้ว
            for i in d.A:
                xie = self.x(i, end_node_k, k)
                if xie is None:
                    continue
                m_ie = self._big_m_arc(i, end_node_k)
                rhs = d.v.get(i, 0.0) + d.travel_time(i, end_node_k) - m_ie
                rb.add_ge({self.aend(k): 1, self.a(i, k): -1, xie: -m_ie}, rhs)

            # (8) ช่วงเวลาที่อยากเข้าชม E_i <= a_ik <= L_i  [FIX 2: ผูกกับ y_ik]
            for i in d.A:
                yik, aik = self.y(i, k), self.a(i, k)
                Ei, Li = d.E.get(i), d.L.get(i)
                if Ei is not None:
                    rb.add_ge({aik: 1, yik: -self.big_m}, Ei - self.big_m)
                if Li is not None:
                    rb.add_le({aik: 1, yik: self.big_m}, Li + self.big_m)

            # (9) เวลาเปิด-ปิด OP_i <= a_ik <= CL_i - v_i  [FIX 2]
            for i in d.A:
                yik, aik = self.y(i, k), self.a(i, k)
                OPi, CLi = d.OP.get(i), d.CL.get(i)
                if OPi is not None:
                    rb.add_ge({aik: 1, yik: -self.big_m}, OPi - self.big_m)
                if CLi is not None:
                    limit = CLi - d.v.get(i, 0.0)
                    rb.add_le({aik: 1, yik: self.big_m}, limit + self.big_m)

            # (11) ช่วงเวลามื้ออาหารแบบตายตัว [FIX 9] ปิดไว้เป็นค่าเริ่มต้น
            # ใช้เฉพาะร้านที่ไม่มี E_i/L_i ของตัวเอง
            if d.use_meal_window:
                meal_lo, meal_hi = d.meal_window
                for i in d.F:
                    if i in d.E or i in d.L:
                        continue
                    yik, aik = self.y(i, k), self.a(i, k)
                    rb.add_ge({aik: 1, yik: -self.big_m}, meal_lo - self.big_m)
                    rb.add_le({aik: 1, yik: self.big_m}, meal_hi + self.big_m)

            # (10) กินร้านอาหารวันละ 1 แห่งพอดี
            meal_coeffs = {self.y(i, k): 1 for i in d.F}
            if meal_coeffs:
                rb.add_eq(meal_coeffs, 1.0)

            # (13) MTZ กำจัด subtour: ถ้าไป i -> j แล้ว u_j >= u_i + 1
            nA = len(d.A)
            for i, j in itertools.permutations(d.A, 2):
                xij = self.x(i, j, k)
                if xij is None:
                    continue
                rb.add_le({self.u(i, k): 1, self.u(j, k): -1, xij: nA}, nA - 1)

        # (12) สถานที่บังคับ: ต้องไปพอดี 1 ครั้งตลอดทริป
        for i in d.M:
            rb.add_eq({self.y(i, k): 1 for k in d.K}, 1.0)

        # [FIX 12] สถานที่ใดก็ตามเที่ยวได้ไม่เกิน 1 ครั้งตลอดทริป (กันซ้ำข้ามวัน)
        for i in d.A:
            rb.add_le({self.y(i, k): 1 for k in d.K}, 1.0)

        # (6') งบประมาณ [FIX 6]: ค่าเข้าชม + ค่าเดินทาง/น้ำมัน <= B
        budget_coeffs = {}
        for k in d.K:
            nodes_k = self._node_set_per_day[k]
            for i in d.A:
                yik = self.y(i, k)
                budget_coeffs[yik] = budget_coeffs.get(yik, 0) + d.c_visit.get(i, 0.0)
            for i, j in itertools.permutations(nodes_k, 2):
                xij = self.x(i, j, k)
                if xij is None:
                    continue
                tc = d.travel_cost(i, j)
                if tc:
                    budget_coeffs[xij] = budget_coeffs.get(xij, 0) + tc
        rb.add_le(budget_coeffs, d.B)

        constraints = rb.to_linear_constraint()
        bounds = Bounds(np.array(self.var_lb), np.array(self.var_ub))
        integrality = np.array(self.integrality)
        return c, constraints, bounds, integrality

    # ------------------------------------------------------------- แก้โจทย์ --
    def solve(self, **milp_kwargs):
        """ส่งโมเดลให้ HiGHS แก้ (ใส่ options เพิ่มได้ เช่น options={"time_limit": 60})"""
        c, constraints, bounds, integrality = self.build()
        return milp(
            c,
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
            **milp_kwargs,
        )

    def total_cost(self, res) -> float:
        """ค่าใช้จ่ายรวมจริงของคำตอบ = ค่าเข้าชม + ค่าเดินทาง ตาม (6')"""
        d = self.d
        x_sol = res.x
        total = 0.0
        for k in d.K:
            nodes_k = self._node_set_per_day[k]
            for i in d.A:
                if round(x_sol[self.y(i, k)]) == 1:
                    total += d.c_visit.get(i, 0.0)
            for i, j in itertools.permutations(nodes_k, 2):
                xij = self.x(i, j, k)
                if xij is not None and round(x_sol[xij]) == 1:
                    total += d.travel_cost(i, j)
        return total

    def decode(self, res) -> Dict:
        """แปลงคำตอบตัวเลขของ solver เป็นเส้นทางแต่ละวันที่อ่านง่าย"""
        if not res.success:
            return {"success": False, "message": res.message}

        x_sol = res.x
        d = self.d
        itinerary = {}
        total_score = 0.0
        for k in d.K:
            nodes_k = self._node_set_per_day[k]
            end_node_k = d.hotel if d.r[k] == 1 else DUMMY

            # เดินตามเส้นที่ x = 1 เริ่มจากโรงแรมจนถึงจุดสิ้นสุด
            route = [d.hotel]
            current = d.hotel
            for _ in range(len(nodes_k) + 2):        # กันวนไม่รู้จบ
                nxt = None
                for j in nodes_k:
                    if j == current:
                        continue
                    xij = self.x(current, j, k)
                    if xij is not None and round(x_sol[xij]) == 1:
                        nxt = j
                        break
                if nxt is None:
                    break
                route.append(nxt if nxt != DUMMY else "(สิ้นสุดเส้นทาง)")
                if nxt == end_node_k:
                    break
                current = nxt

            day_score = sum(
                d.s.get(i, 0.0) for i in d.A if round(x_sol[self.y(i, k)]) == 1
            )
            total_score += day_score

            arrivals = {d.hotel: round(float(x_sol[self.a(d.hotel, k)]), 2)}
            for i in d.A:
                if round(x_sol[self.y(i, k)]) == 1:
                    arrivals[i] = round(float(x_sol[self.a(i, k)]), 2)

            end_time = round(float(x_sol[self.aend(k)]), 2)     # [FIX 11]
            itinerary[k] = {
                "route": route,
                "arrival_times": arrivals,
                "end_time": end_time,
                "day_length": round(end_time - d.start_time, 2),
                "score": day_score,
            }
        return {
            "success": True,
            "objective": -res.fun,          # solver minimize (-Z) จึงกลับเครื่องหมาย
            "total_score": total_score,
            "total_cost": self.total_cost(res),
            "days": itinerary,
        }