import tkinter as tk
from tkinter import Label, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import pytesseract
import numpy as np
from rapidfuzz import fuzz, process

# === CONFIG ===
LANGUAGE = "tha"
CONFIDENCE_THRESHOLD = 0.5

# === MOCK DATA ===
thai_provinces = ["กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"]
car_list = ["กข1234กรุงเทพมหานคร", "1กข2345เชียงใหม่", "2กข3456ขอนแก่น"]

# === UTILS ===
def match_province(input_text, threshold=30):
    best_match = process.extractOne(input_text, thai_provinces, scorer=fuzz.ratio)
    if best_match and best_match[1] >= threshold:
        return best_match[0]
    return ""

def mock_ocr_processing(image_path):
    """จำลองการประมวลผล OCR"""
    # ในการใช้งานจริงจะใช้ YOLO และ OCR
    # ที่นี่เราจำลองผลลัพธ์
    mock_results = [
        ("กข1234", "กรุงเทพมหานคร"),
        ("1กข2345", "เชียงใหม่"),
        ("2กข3456", "ขอนแก่น"),
        ("3กข4567", "ภูเก็ต")
    ]
    import random
    return random.choice(mock_results)

# === MAIN WORKFLOW ===
def process_image():
    file_path = filedialog.askopenfilename(
        title="เลือกรูปภาพป้ายทะเบียน",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        return
    
    try:
        # แสดงภาพที่เลือก
        image = Image.open(file_path)
        image = image.resize((640, 480))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        # จำลองการประมวลผล
        code, province = mock_ocr_processing(file_path)
        plate_txt = code.replace(" ", "") + province.replace(" ", "")
        
        print(f"📸 Plate: {plate_txt}")
        
        # เช็ค whitelist
        if plate_txt.strip() in car_list:
            status_value.config(text="✅ อนุญาตให้ผ่าน", fg='green')
            messagebox.showinfo("ผลลัพธ์", "✅ อนุญาตให้ผ่าน")
        else:
            status_value.config(text="❌ ไม่พบในระบบ", fg='red')
            messagebox.showwarning("ผลลัพธ์", "❌ ไม่พบในระบบ")
        
        # อัปเดต GUI
        plate_value.config(text=code.strip())
        province_value.config(text=province.strip())
        
    except Exception as e:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถประมวลผลภาพได้: {str(e)}")

# === GUI INIT ===
root = tk.Tk()
root.configure(bg='black')
root.title("License Plate Recognition - Demo")
root.geometry("900x600")

# ภาพหลัก
default_img = Image.new("RGB", (640, 480), color='black')
photo = ImageTk.PhotoImage(default_img)
image_label = Label(root, image=photo)
image_label.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

# Label GUI
Label(root, text="เลขทะเบียน", fg='skyblue', bg='black', font=("Arial", 20, "bold")).grid(row=0, column=1, sticky='w', padx=10)
plate_value = Label(root, text="---", fg='white', bg='grey', font=("Arial", 18), width=20)
plate_value.grid(row=1, column=1, sticky='w', padx=10, pady=5)

Label(root, text="จังหวัด", fg='skyblue', bg='black', font=("Arial", 20, "bold")).grid(row=2, column=1, sticky='w', padx=10)
province_value = Label(root, text="---", fg='white', bg='grey', font=("Arial", 18), width=20)
province_value.grid(row=3, column=1, sticky='w', padx=10, pady=5)

Label(root, text="สถานะ", fg='red', bg='black', font=("Arial", 20, "bold")).grid(row=4, column=1, sticky='w', padx=10)
status_value = Label(root, text="---", fg='white', bg='grey', font=("Arial", 18), width=20)
status_value.grid(row=5, column=1, sticky='w', padx=10, pady=5)

# ปุ่มเลือกภาพ
process_button = tk.Button(root, text="เลือกภาพป้ายทะเบียน", command=process_image, 
                          bg='blue', fg='white', font=("Arial", 16, "bold"), width=20)
process_button.grid(row=6, column=1, padx=10, pady=20)

print("📦 Demo system ready!")
root.mainloop()