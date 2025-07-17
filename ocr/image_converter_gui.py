import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# --- Style Configuration ---
BG_COLOR = "#f0f0f0"
FRAME_BG_COLOR = "#ffffff"
HEADER_COLOR = "#2c3e50"
BUTTON_BG = "#4a90e2"
BUTTON_FG = "white"
BUTTON_ACTIVE = "#3675b5"
ACCENT_COLOR = "#28a745"
ACCENT_TEXT_COLOR = "white"
TEXT_COLOR = "#333333"

FONT_MAIN_TITLE = ("Helvetica", 20, "bold")
FONT_SUB_TITLE = ("Helvetica", 12, "bold")
FONT_NORMAL = ("Helvetica", 10)
FONT_RESULT = ("Helvetica", 16, "bold")


# Global variables
original_cv_image = None
original_image_tk = None
license_plate_model = None
code_prov_model = None
thai_provinces_list = []

# --- Model and Data Loading Functions ---
def load_models():
    """Loads YOLOv8 models and province list."""
    global license_plate_model, code_prov_model
    try:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Uncomment and set your Tesseract path if needed
        license_plate_model = YOLO("license_plate.pt")
        code_prov_model = YOLO("code_prov.pt")
        print("YOLOv8 models loaded successfully.")
        filename_label.config(text="Models loaded successfully.", fg=TEXT_COLOR)
        load_provinces()
    except Exception as e:
        print(f"Error loading models: {e}")
        filename_label.config(text=f"Error loading models: {e}", fg="red")

def load_provinces():
    """Loads the list of Thai provinces from 'thai_provinces.txt'."""
    global thai_provinces_list
    try:
        with open("thai_provinces.txt", "r", encoding="utf-8") as f:
            thai_provinces_list = [line.strip() for line in f.readlines()]
        print("Thai provinces list loaded successfully.")
    except FileNotFoundError:
        print("thai_provinces.txt not found. Please create this file with Thai province names.")
        filename_label.config(text="Error: thai_provinces.txt not found.", fg="red")

def remove_thai_vowels(text):
    """
    Removes Thai vowels and tonal marks from a string.
    """
    vowels_and_tones = "ะาิีึืุูเแโใไำ" + "่้๊็" + "์"
    for char in vowels_and_tones:
        text = text.replace(char, "")
    return text

# --- Helper function for perspective transformation ---
def order_points(pts):
    """
    Orders a list of 4 points in top-left, top-right, bottom-right, bottom-left order.
    Used for perspective transformation.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Applies a four-point perspective transform to an image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Main function for YOLO + OCR with step-by-step display ---
def run_yolo_detection_and_ocr():
    """
    Uses YOLOv8 models for detection and OCR, displaying results at each step.
    Includes skew correction for the license plate.
    """
    global original_cv_image
    
    if original_cv_image is None:
        filename_label.config(text="Please upload an image first.", fg="red")
        return
    
    if license_plate_model is None or code_prov_model is None:
        filename_label.config(text="Loading models...", fg=TEXT_COLOR)
        root.update_idletasks()
        load_models()
        if license_plate_model is None or code_prov_model is None:
            filename_label.config(text="Error: Models not loaded. Check console for details.", fg="red")
            return
            
    # Clear previously displayed images
    for attr in ['detected_plate_label', 'cropped_plate_label', 'gray_label', 'thresh_label', 'contour_label', 'corner_label', 'warped_label', 'sharpened_label', 'unsharp_label', 'laplacian_label', 'detected_chars_label']:
        if hasattr(root, attr):
            getattr(root, attr).config(image='')
    result_label.config(text="")

    max_width = 400
    
    # 1. Detection: YOLO
    filename_label.config(text="Step 1/11: Detecting license plate...", fg=TEXT_COLOR)
    root.update_idletasks()
    results_plate = license_plate_model(original_cv_image, verbose=False)
    
    if not results_plate or not results_plate[0].boxes or len(results_plate[0].boxes) == 0:
        result_label.config(text="No license plate found in the image.", fg="red")
        filename_label.config(text="Detection complete: No plate found.", fg=TEXT_COLOR)
        return

    detected_plate_cv = original_cv_image.copy()
    box_plate = results_plate[0].boxes[0]
    x1, y1, x2, y2 = map(int, box_plate.xyxy[0])
    cv2.rectangle(detected_plate_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)

    detected_plate_pil = Image.fromarray(cv2.cvtColor(detected_plate_cv, cv2.COLOR_BGR2RGB))
    detected_plate_pil = detected_plate_pil.resize((max_width, int(detected_plate_pil.height * max_width / detected_plate_pil.width)), Image.LANCZOS)
    detected_plate_tk = ImageTk.PhotoImage(detected_plate_pil)

    if not hasattr(root, 'detected_plate_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="1. Plate Detected", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.detected_plate_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.detected_plate_label.pack(padx=5, pady=5)
    root.detected_plate_label.config(image=detected_plate_tk)
    root.detected_plate_label.image = detected_plate_tk
    canvas.xview_moveto(0)

    # 2. Cropping ป้ายทะเบียน
    filename_label.config(text="Step 2/11: Cropping plate...", fg=TEXT_COLOR)
    root.update_idletasks()
    initial_cropped_plate = original_cv_image[y1:y2, x1:x2]

    cropped_pil = Image.fromarray(cv2.cvtColor(initial_cropped_plate, cv2.COLOR_BGR2RGB))
    cropped_pil = cropped_pil.resize((max_width, int(cropped_pil.height * max_width / cropped_pil.width)), Image.LANCZOS)
    cropped_tk = ImageTk.PhotoImage(cropped_pil)
    
    if not hasattr(root, 'cropped_plate_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="2. Cropped Plate", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.cropped_plate_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.cropped_plate_label.pack(padx=5, pady=5)
    root.cropped_plate_label.config(image=cropped_tk)
    root.cropped_plate_label.image = cropped_tk
    canvas.xview_moveto(0.2)
    
    # 3. Grayscale
    filename_label.config(text="Step 3/11: Converting to Grayscale...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    gray = cv2.cvtColor(initial_cropped_plate, cv2.COLOR_BGR2GRAY)
    
    gray_pil = Image.fromarray(gray)
    gray_pil = gray_pil.resize((max_width, int(gray_pil.height * max_width / gray_pil.width)), Image.LANCZOS)
    gray_tk = ImageTk.PhotoImage(gray_pil)

    if not hasattr(root, 'gray_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="3. Grayscale", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.gray_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.gray_label.pack(padx=5, pady=5)
    root.gray_label.config(image=gray_tk)
    root.gray_label.image = gray_tk
    canvas.xview_moveto(0.3)
    
    # 4. Threshold
    filename_label.config(text="Step 4/11: Thresholding...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh_pil = Image.fromarray(thresh)
    thresh_pil = thresh_pil.resize((max_width, int(thresh_pil.height * max_width / thresh_pil.width)), Image.LANCZOS)
    thresh_tk = ImageTk.PhotoImage(thresh_pil)

    if not hasattr(root, 'thresh_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="4. Threshold", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.thresh_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.thresh_label.pack(padx=5, pady=5)
    root.thresh_label.config(image=thresh_tk)
    root.thresh_label.image = thresh_tk
    canvas.xview_moveto(0.4)
    
    # 5. Contour Detection
    filename_label.config(text="Step 5/11: Contour Detection...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    # Find largest white contour only
    white_thresh = cv2.bitwise_not(thresh)
    white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(white_contours, key=cv2.contourArea) if white_contours else None
    contour_img = initial_cropped_plate.copy()
    if largest_contour is not None:
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)
    
    contour_pil = Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    contour_pil = contour_pil.resize((max_width, int(contour_pil.height * max_width / contour_pil.width)), Image.LANCZOS)
    contour_tk = ImageTk.PhotoImage(contour_pil)
    
    if not hasattr(root, 'contour_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="5. Contour Detection", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.contour_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.contour_label.pack(padx=5, pady=5)
    root.contour_label.config(image=contour_tk)
    root.contour_label.image = contour_tk
    canvas.xview_moveto(0.45)
    
    # 6. Rectangle Corner Detection
    filename_label.config(text="Step 6/11: Rectangle Corner Detection...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    corner_img = initial_cropped_plate.copy()
    if largest_contour is not None:
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        cv2.drawContours(corner_img, [largest_contour], -1, (0, 255, 0), 2)
        if len(approx) == 4:
            for point in approx:
                cv2.circle(corner_img, tuple(point[0]), 8, (0, 0, 255), -1)

    corner_pil = Image.fromarray(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
    corner_pil = corner_pil.resize((max_width, int(corner_pil.height * max_width / corner_pil.width)), Image.LANCZOS)
    corner_tk = ImageTk.PhotoImage(corner_pil)

    if not hasattr(root, 'corner_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="6. Rectangle Corners", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.corner_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.corner_label.pack(padx=5, pady=5)
    root.corner_label.config(image=corner_tk)
    root.corner_label.image = corner_tk
    canvas.xview_moveto(0.55)
    
    # 7. Warp Perspective
    filename_label.config(text="Step 7/11: Warp Perspective...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    warped_cv = initial_cropped_plate.copy()
    if largest_contour is not None:
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(approx) == 4:
            # Thai license plate ratio 15:34
            plate_width = 340
            plate_height = 150
            dst_points = np.array([[0, 0], [plate_width, 0], [plate_width, plate_height], [0, plate_height]], dtype="float32")
            src_points = approx.reshape(4, 2).astype("float32")
            # Order points properly
            src_points = order_points(src_points)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_cv = cv2.warpPerspective(initial_cropped_plate, M, (plate_width, plate_height))

    warped_pil = Image.fromarray(cv2.cvtColor(warped_cv, cv2.COLOR_BGR2RGB))
    warped_pil = warped_pil.resize((max_width, int(warped_pil.height * max_width / warped_pil.width)), Image.LANCZOS)
    warped_tk = ImageTk.PhotoImage(warped_pil)

    if not hasattr(root, 'warped_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="7. Warped Perspective", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.warped_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.warped_label.pack(padx=5, pady=5)
    root.warped_label.config(image=warped_tk)
    root.warped_label.image = warped_tk
    canvas.xview_moveto(0.65)
    
    # 8. Sharpening
    filename_label.config(text="Step 8/11: Sharpening...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    warped_gray = cv2.cvtColor(warped_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(warped_gray, -1, kernel)
    
    sharpened_pil = Image.fromarray(sharpened)
    sharpened_pil = sharpened_pil.resize((max_width, int(sharpened_pil.height * max_width / sharpened_pil.width)), Image.LANCZOS)
    sharpened_tk = ImageTk.PhotoImage(sharpened_pil)
    
    if not hasattr(root, 'sharpened_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="8. Sharpened", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.sharpened_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.sharpened_label.pack(padx=5, pady=5)
    root.sharpened_label.config(image=sharpened_tk)
    root.sharpened_label.image = sharpened_tk
    canvas.xview_moveto(0.7)
    
    # 9. Unsharp Masking
    filename_label.config(text="Step 9/11: Unsharp Masking...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    blurred = cv2.GaussianBlur(warped_gray, (9, 9), 10.0)
    unsharp = cv2.addWeighted(warped_gray, 1.5, blurred, -0.5, 0)
    
    unsharp_pil = Image.fromarray(unsharp)
    unsharp_pil = unsharp_pil.resize((max_width, int(unsharp_pil.height * max_width / unsharp_pil.width)), Image.LANCZOS)
    unsharp_tk = ImageTk.PhotoImage(unsharp_pil)
    
    if not hasattr(root, 'unsharp_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="9. Unsharp Mask", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.unsharp_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.unsharp_label.pack(padx=5, pady=5)
    root.unsharp_label.config(image=unsharp_tk)
    root.unsharp_label.image = unsharp_tk
    canvas.xview_moveto(0.75)
    
    # 10. Laplacian Filter
    filename_label.config(text="Step 10/11: Laplacian Filter...", fg=TEXT_COLOR)
    root.update_idletasks()
    
    laplacian = cv2.Laplacian(warped_gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    laplacian_pil = Image.fromarray(laplacian)
    laplacian_pil = laplacian_pil.resize((max_width, int(laplacian_pil.height * max_width / laplacian_pil.width)), Image.LANCZOS)
    laplacian_tk = ImageTk.PhotoImage(laplacian_pil)
    
    if not hasattr(root, 'laplacian_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="10. Laplacian", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.laplacian_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.laplacian_label.pack(padx=5, pady=5)
    root.laplacian_label.config(image=laplacian_tk)
    root.laplacian_label.image = laplacian_tk
    canvas.xview_moveto(0.8)
    
    # 11. OCR: Tesseract
    filename_label.config(text="Step 11/11: OCR and cleaning text...", fg=TEXT_COLOR)
    root.update_idletasks()
    results_chars = code_prov_model(warped_cv, verbose=False)
    
    if not results_chars or not results_chars[0].boxes or len(results_chars[0].boxes) == 0:
        result_label.config(text="No characters or province found on the license plate.", fg="red")
        filename_label.config(text="Detection complete: No characters/province found.", fg=TEXT_COLOR)
        return
    
    detected_chars_cv = warped_cv.copy()
    sorted_boxes = sorted(results_chars[0].boxes, key=lambda b: b.xyxy[0][0])
    extracted_text_chars_only = ""
    extracted_text_full = ""
    
    for box in sorted_boxes:
        x_char, y_char, w_char, h_char = map(int, box.xyxy[0])
        cv2.rectangle(detected_chars_cv, (x_char, y_char), (w_char, h_char), (0, 0, 255), 2)
        
        cropped_char = warped_cv[y_char:h_char, x_char:w_char]
        cropped_char_gray = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
        
        class_name = code_prov_model.names[int(box.cls)]
        
        if class_name == 'จังหวัด':
                # Skip OCR for now
            found_province = "จังหวัด"
            extracted_text_full += found_province
            extracted_text_chars_only += remove_thai_vowels(found_province)
            
        else:
            # Skip OCR for now
            text = "ตัวอักษร"
            extracted_text_full += text
            extracted_text_chars_only += remove_thai_vowels(text)
    
    detected_chars_pil = Image.fromarray(cv2.cvtColor(detected_chars_cv, cv2.COLOR_BGR2RGB))
    detected_chars_pil = detected_chars_pil.resize((max_width, int(detected_chars_pil.height * max_width / detected_chars_pil.width)), Image.LANCZOS)
    detected_chars_tk = ImageTk.PhotoImage(detected_chars_pil)
    
    if not hasattr(root, 'detected_chars_label'):
        frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame, text="11. Chars Detected", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR).pack(pady=5)
        root.detected_chars_label = tk.Label(frame, bg=FRAME_BG_COLOR)
        root.detected_chars_label.pack(padx=5, pady=5)
    root.detected_chars_label.config(image=detected_chars_tk)
    root.detected_chars_label.image = detected_chars_tk
    canvas.xview_moveto(0.85)

    # Display both full and chars-only text
    result_label.config(text=f"Detected License Plate: {extracted_text_full}\n(Chars Only: {extracted_text_chars_only})", fg=ACCENT_COLOR)
    filename_label.config(text="Detection and OCR complete.", fg=TEXT_COLOR)
    print(f"Extracted Text (Full): {extracted_text_full}")
    # print(f"Extracted Text (Chars Only): {extracted_text_chars_only}")


# --- Image Upload Function ---
def upload_image():
    """
    Opens a file dialog for the user to select an image file and processes it.
    """
    global original_cv_image, original_image_tk

    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        try:
            original_cv_image = cv2.imread(file_path)
            if original_cv_image is None:
                filename_label.config(text="Could not read this image file. Please ensure it's a valid image.", fg="red")
                return

            original_pil_image = Image.fromarray(cv2.cvtColor(original_cv_image, cv2.COLOR_BGR2RGB))
            max_width = 400
            original_pil_image = original_pil_image.resize((max_width, int(original_pil_image.height * max_width / original_pil_image.width)), Image.LANCZOS)
            original_image_tk = ImageTk.PhotoImage(original_pil_image)
            original_label.config(image=original_image_tk)
            original_label.image = original_image_tk

            filename_label.config(text="Selected file: " + file_path.split("/")[-1], fg=TEXT_COLOR)
            result_label.config(text="", fg=ACCENT_COLOR)
            
            for attr in ['detected_plate_label', 'cropped_plate_label', 'gray_label', 'thresh_label', 'contour_label', 'corner_label', 'warped_label', 'sharpened_label', 'unsharp_label', 'laplacian_label', 'detected_chars_label']:
                if hasattr(root, attr):
                    getattr(root, attr).config(image='')
            canvas.xview_moveto(0)

        except Exception as e:
            filename_label.config(text=f"Error processing image: {e}", fg="red")
    else:
        filename_label.config(text="No file selected.", fg=TEXT_COLOR)

# --- GUI Construction ---
root = tk.Tk()
root.title("YOLO License Plate Detector (Custom Workflow)")
root.geometry("1400x700") 
root.resizable(width=False, height=False)
root.configure(bg=BG_COLOR)

main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

header_frame = tk.Frame(main_frame, bg=HEADER_COLOR)
header_frame.pack(fill=tk.X, pady=(0, 20))
title_label = tk.Label(header_frame, text="License Plate Detection & OCR", font=FONT_MAIN_TITLE, bg=HEADER_COLOR, fg="white", pady=10)
title_label.pack()

top_frame = tk.Frame(main_frame, bg=BG_COLOR)
top_frame.pack(pady=10)

upload_button = tk.Button(top_frame, text="Upload Image", command=upload_image, font=FONT_NORMAL, bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_ACTIVE, padx=10, pady=5, bd=0)
upload_button.pack(side=tk.LEFT, padx=10)

yolo_button = tk.Button(top_frame, text="Run Detection & OCR", command=run_yolo_detection_and_ocr, font=FONT_NORMAL, bg=ACCENT_COLOR, fg=ACCENT_TEXT_COLOR, activebackground=BUTTON_ACTIVE, padx=10, pady=5, bd=0)
yolo_button.pack(side=tk.LEFT, padx=20)

filename_label = tk.Label(top_frame, text="No file selected", font=FONT_NORMAL, bg=BG_COLOR, fg=TEXT_COLOR)
filename_label.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(main_frame, text="", font=FONT_RESULT, bg=BG_COLOR, fg=ACCENT_COLOR)
result_label.pack(pady=10)

canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0, bg=BG_COLOR)
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

h_scrollbar = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
canvas.configure(xscrollcommand=h_scrollbar.set)

image_frame = tk.Frame(canvas, bg=BG_COLOR)
canvas.create_window((0, 0), window=image_frame, anchor="nw")

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

image_frame.bind("<Configure>", on_frame_configure)

original_frame = tk.Frame(image_frame, bg=FRAME_BG_COLOR, bd=2, relief="solid")
original_frame.pack(side=tk.LEFT, padx=10, pady=10)
before_label = tk.Label(original_frame, text="Original Image", font=FONT_SUB_TITLE, bg=FRAME_BG_COLOR, fg=HEADER_COLOR)
before_label.pack(pady=5)
original_label = tk.Label(original_frame, bg=FRAME_BG_COLOR)
original_label.pack(padx=5, pady=5)

load_models()

root.mainloop()