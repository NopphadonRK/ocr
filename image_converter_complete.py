import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import os

# --- Style Configuration ---
BG_COLOR = "#f0f0f0"
FRAME_BG_COLOR = "#ffffff"
HEADER_COLOR = "#2c3e50"
BUTTON_BG = "#4a90e2"
BUTTON_FG = "white"
TEXT_COLOR = "#333333"

FONT_MAIN_TITLE = ("Helvetica", 20, "bold")
FONT_SUB_TITLE = ("Helvetica", 12, "bold")
FONT_NORMAL = ("Helvetica", 10)

# Global variables
original_cv_image = None
license_plate_model = None
code_prov_model = None

def load_models():
    """Load YOLO models if available"""
    global license_plate_model, code_prov_model
    try:
        if os.path.exists("untitled folder/license_plate.pt"):
            license_plate_model = YOLO("untitled folder/license_plate.pt")
            print("License plate model loaded")
        elif os.path.exists("PR-LPR-RPI/LicensePlate.pt"):
            license_plate_model = YOLO("PR-LPR-RPI/LicensePlate.pt")
            print("License plate model loaded from PR-LPR-RPI")
            
        if os.path.exists("untitled folder/code_prov.pt"):
            code_prov_model = YOLO("untitled folder/code_prov.pt")
            print("Code province model loaded")
        elif os.path.exists("PR-LPR-RPI/CodeProv.pt"):
            code_prov_model = YOLO("PR-LPR-RPI/CodeProv.pt")
            print("Code province model loaded from PR-LPR-RPI")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def upload_image():
    """Upload and display image"""
    global original_cv_image
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        return
    
    try:
        original_cv_image = cv2.imread(file_path)
        if original_cv_image is None:
            messagebox.showerror("Error", "Cannot load image")
            return
            
        # Display original image
        display_img = cv2.cvtColor(original_cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        pil_img = pil_img.resize((400, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        
        original_label.config(image=photo)
        original_label.image = photo
        
        filename_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
        process_button.config(state="normal")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error loading image: {str(e)}")

def process_image():
    """Process image with YOLO detection and OCR"""
    global original_cv_image
    
    if original_cv_image is None:
        messagebox.showwarning("Warning", "Please upload an image first")
        return
    
    try:
        # Clear previous results
        result_text.delete(1.0, tk.END)
        
        # Mock processing if models not available
        if license_plate_model is None:
            result_text.insert(tk.END, "Demo Mode - Models not loaded\n")
            result_text.insert(tk.END, "Simulating license plate detection...\n")
            result_text.insert(tk.END, "Mock Result: กข1234 กรุงเทพมหานคร\n")
            return
        
        # Real YOLO processing
        result_text.insert(tk.END, "Processing with YOLO models...\n")
        
        # Step 1: Detect license plate
        results = license_plate_model(original_cv_image, verbose=False)
        
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            result_text.insert(tk.END, "No license plate detected\n")
            return
        
        # Get first detection
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        result_text.insert(tk.END, f"License plate detected (confidence: {confidence:.2f})\n")
        
        # Crop license plate
        cropped_plate = original_cv_image[y1:y2, x1:x2]
        
        # Display cropped plate
        display_cropped = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
        pil_cropped = Image.fromarray(display_cropped)
        pil_cropped = pil_cropped.resize((300, 100), Image.LANCZOS)
        photo_cropped = ImageTk.PhotoImage(pil_cropped)
        
        cropped_label.config(image=photo_cropped)
        cropped_label.image = photo_cropped
        
        # Step 2: Detect code and province parts
        if code_prov_model is not None:
            code_results = code_prov_model(cropped_plate, verbose=False)
            
            if code_results and code_results[0].boxes:
                result_text.insert(tk.END, f"Found {len(code_results[0].boxes)} text regions\n")
                
                # Process each detected region
                for i, box in enumerate(code_results[0].boxes):
                    x1_c, y1_c, x2_c, y2_c = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Crop text region
                    text_region = cropped_plate[y1_c:y2_c, x1_c:x2_c]
                    
                    # OCR on text region
                    try:
                        text = pytesseract.image_to_string(text_region, lang='tha', config='--psm 7').strip()
                        region_type = "Code" if cls_id == 0 else "Province"
                        result_text.insert(tk.END, f"{region_type}: {text} (conf: {conf:.2f})\n")
                    except:
                        result_text.insert(tk.END, f"OCR failed for region {i}\n")
            else:
                result_text.insert(tk.END, "No text regions detected\n")
        else:
            # Fallback OCR on entire plate
            try:
                text = pytesseract.image_to_string(cropped_plate, lang='tha').strip()
                result_text.insert(tk.END, f"OCR Result: {text}\n")
            except:
                result_text.insert(tk.END, "OCR processing failed\n")
                
    except Exception as e:
        messagebox.showerror("Error", f"Processing error: {str(e)}")

# Create main window
root = tk.Tk()
root.title("License Plate Recognition")
root.configure(bg=BG_COLOR)
root.geometry("800x700")

# Title
title_label = tk.Label(root, text="License Plate Recognition System", 
                      font=FONT_MAIN_TITLE, bg=BG_COLOR, fg=HEADER_COLOR)
title_label.pack(pady=10)

# Upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image,
                         bg=BUTTON_BG, fg=BUTTON_FG, font=FONT_SUB_TITLE,
                         width=15, height=2)
upload_button.pack(pady=10)

# Filename label
filename_label = tk.Label(root, text="No image selected", 
                         font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR)
filename_label.pack(pady=5)

# Original image frame
original_frame = tk.Frame(root, bg=FRAME_BG_COLOR, bd=2, relief="solid")
original_frame.pack(pady=10)

tk.Label(original_frame, text="Original Image", font=FONT_SUB_TITLE, 
         bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)

original_label = tk.Label(original_frame, bg=FRAME_BG_COLOR, 
                         text="No image", width=50, height=15)
original_label.pack(padx=10, pady=10)

# Process button
process_button = tk.Button(root, text="Process Image", command=process_image,
                          bg="#28a745", fg="white", font=FONT_SUB_TITLE,
                          width=15, height=2, state="disabled")
process_button.pack(pady=10)

# Cropped plate frame
cropped_frame = tk.Frame(root, bg=FRAME_BG_COLOR, bd=2, relief="solid")
cropped_frame.pack(pady=10)

tk.Label(cropped_frame, text="Detected License Plate", font=FONT_SUB_TITLE,
         bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)

cropped_label = tk.Label(cropped_frame, bg=FRAME_BG_COLOR,
                        text="No detection", width=40, height=5)
cropped_label.pack(padx=10, pady=10)

# Results frame
results_frame = tk.Frame(root, bg=FRAME_BG_COLOR, bd=2, relief="solid")
results_frame.pack(pady=10, fill="both", expand=True)

tk.Label(results_frame, text="Processing Results", font=FONT_SUB_TITLE,
         bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)

result_text = tk.Text(results_frame, height=8, width=80, font=FONT_NORMAL)
result_text.pack(padx=10, pady=10, fill="both", expand=True)

# Load models on startup
print("Loading models...")
load_models()
print("Application ready!")

root.mainloop()