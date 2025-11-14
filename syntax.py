from PIL import Image
import fitz
import os
from pathlib import Path

# --- CONFIG ---
template_path = "2024 SAT Template (3).pdf"
output_path = "output_with_all_questions.pdf"
image_folder = "question_images"
fixed_image_width = 230

# --- Define slots for page 1 and page 2+ ---
page1_slots = [
    fitz.Rect(50, 350, 280, 560),   # top-left on page 1
    fitz.Rect(320, 350, 570, 560),  # top-right on page 1
]

page2plus_slots = [
    fitz.Rect(50, 130, 280, 340),   # top-left
    fitz.Rect(50, 370, 280, 580),   # bottom-left
    fitz.Rect(320, 130, 550, 340),  # top-right
    fitz.Rect(320, 370, 550, 580),  # bottom-right
]

# --- STEP 1: Load template and image list ---
doc = fitz.open(template_path)
image_paths = sorted(Path(image_folder).glob("*.png"))

if len(image_paths) < 2:
    raise ValueError("You need at least 2 images in the folder.")

# --- STEP 2: Insert 2 images on page 1 ---
page1 = doc[0]
for idx in range(2):
    img_path = image_paths[idx]
    img = Image.open(img_path)
    img_w, img_h = img.size
    slot = page1_slots[idx]

    scale = fixed_image_width / img_w
    new_w = fixed_image_width
    new_h = img_h * scale

    # Top-left anchored placement
    x0 = slot.x0
    y0 = slot.y0
    x1 = x0 + new_w
    y1 = y0 + new_h
    fit_rect = fitz.Rect(x0, y0, x1, y1)

    page1.insert_image(fit_rect, filename=str(img_path))

# --- STEP 3: Insert remaining images from page 2+ with overlap detection ---
remaining_images = image_paths[2:]
img_idx = 0
current_page_num = 1
placed_rects = {}  # Keeps track of images placed on each page

# --- Overlap helper ---
def overlaps(r1, r2):
    return not (r1.x1 <= r2.x0 or r1.x0 >= r2.x1 or r1.y1 <= r2.y0 or r1.y0 >= r2.y1)

def next_available_page(doc, start_at=1):
    while start_at >= len(doc):
        doc.insert_pdf(fitz.open(template_path), start_at=1)
    return doc[start_at]

while img_idx < len(remaining_images):
    page = next_available_page(doc, current_page_num)
    if current_page_num not in placed_rects:
        placed_rects[current_page_num] = []

    placed = False
    for slot in page2plus_slots:
        img_path = remaining_images[img_idx]
        img = Image.open(img_path)
        img_w, img_h = img.size

        scale = fixed_image_width / img_w
        new_w = fixed_image_width
        new_h = img_h * scale

        # Top-left anchored placement
        x0 = slot.x0
        y0 = slot.y0
        x1 = x0 + new_w
        y1 = y0 + new_h
        fit_rect = fitz.Rect(x0, y0, x1, y1)

        # Check for overlap
        if any(overlaps(fit_rect, r) for r in placed_rects[current_page_num]):
            continue  # try next slot

        # No overlap – insert image
        page.insert_image(fit_rect, filename=str(img_path))
        placed_rects[current_page_num].append(fit_rect)
        img_idx += 1
        placed = True
        break  # go to next image

    if not placed:
        current_page_num += 1  # try next page

# --- STEP 4: Save final output ---
doc.save(output_path)
print(f"Done: {len(image_paths)} images inserted with top-left alignment and overlap detection.")

