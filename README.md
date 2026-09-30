# TTDP – Tourist Trip Design Problem (BIP)

โมเดลจัดเส้นทางท่องเที่ยวที่ให้คะแนนความพึงพอใจสูงสุด ภายใต้งบประมาณ เวลา เวลาเปิด-ปิด มื้ออาหาร
และสถานที่บังคับ (Binary Integer Program แก้ด้วย HiGHS ผ่าน `scipy.optimize.milp`)

## ไฟล์ในโปรเจกต์
| ไฟล์ | หน้าที่ |
|---|---|
| `ttdp_model.py` | ตัวโมเดลหลัก (ข้อมูล, ตัวแปร, สมการ, แก้, แปลผล) |
| `ttdp_test.py` | ตัวอย่างข้อมูลสมมติ 1 วัน ลองรันได้ทันที |
| `check_fixes.py` | ชุดตรวจอัตโนมัติ ต้องขึ้น PASS ทุกข้อ |
| `geoapify_fetch.py` | ดึงพิกัด/ระยะทางจริงจาก Geoapify -> `places.csv`, `matrix.csv` |
| `requirements.txt` | รายชื่อไลบรารีที่ต้องติดตั้ง |

## วิธีรัน
```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate       Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python ttdp_test.py      # ลองรันตัวอย่าง
python check_fixes.py    # ตรวจว่าโมเดลถูกต้อง
```

## ใช้ข้อมูลจริงจาก Geoapify
```bash
# ตั้ง API key (อย่าเขียนลงโค้ด / อย่า commit)
# Windows PowerShell:  $env:GEOAPIFY_API_KEY="xxxx"
# Mac/Linux:           export GEOAPIFY_API_KEY="xxxx"
python geoapify_fetch.py   # แก้ชื่อโรงแรมท้ายไฟล์ก่อน
```
จากนั้นเปิด `places.csv` กรอก score / ค่าเข้าชม / เวลาเปิด-ปิด แล้วโหลดด้วย `TTDPData.from_csv(...)`

## หน่วยที่ต้องระวัง
- เวลาทั้งหมดเป็น **ชั่วโมงทศนิยม** (11.5 = 11:30)
- `FC` = อัตราสิ้นเปลือง **ลิตรต่อ 100 กม.**, `P_fuel` = บาทต่อลิตร
- พิกัด Geoapify เป็น **[lon, lat]**
