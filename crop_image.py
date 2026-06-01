import os
import cv2

# --- CONFIGURATION ---
# Path where your unzipped dataset is
BASE_PATH = r"C:\Users\user\Desktop\Climbing Hold CNN\Full Dataset\Final_Dataset"
OUTPUT_PATH = r"C:\Users\user\Desktop\Climbing Hold CNN\Clean_Classification_Dataset"

# Map the YOLO integer ID to the actual text names
CLASS_MAP = {
    0: "Jug",
    1: "Crimp",
    2: "Pinch",
    3: "Slope",
    4: "Pocket",
    5: "Volume"
}

# Create output subdirectories
for name in CLASS_MAP.values():
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(OUTPUT_PATH, split, name), exist_ok=True)

def crop_and_save(split):
    img_dir = os.path.join(BASE_PATH, split, "images")   # Check if images are in an 'images' subfolder
    label_dir = os.path.join(BASE_PATH, split, "labels") # Check if labels are in a 'labels' subfolder
    
    # Fallback if everything is just dumped into 'train', 'valid', 'test' directly
    if not os.path.exists(img_dir):
        img_dir = os.path.join(BASE_PATH, split)
        label_dir = os.path.join(BASE_PATH, split)

    counter = 0
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            label_path = os.path.join(label_dir, file)
            # Find matching image (checking for common extensions)
            img_name = file.replace('.txt', '.jpg')
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, file.replace('.txt', '.jpeg'))
                
            if not os.path.exists(img_path):
                continue # Skip if image missing
                
            # Load the full wall image
            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w, _ = image.shape
            
            # Read YOLO coordinates
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])
                
                # Convert normalized YOLO coordinates back to pixel bounding boxes
                x1 = int((x_center - bbox_w / 2) * w)
                y1 = int((y_center - bbox_h / 2) * h)
                x2 = int((x_center + bbox_w / 2) * w)
                y2 = int((y_center + bbox_h / 2) * h)
                
                # Boundary safety check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Crop out the hold
                cropped_hold = image[y1:y2, x1:x2]
                
                if cropped_hold.size > 0:
                    class_name = CLASS_MAP.get(class_id, "Unknown")
                    save_name = f"hold_{counter}.jpg"
                    save_path = os.path.join(OUTPUT_PATH, split, class_name, save_name)
                    cv2.imwrite(save_path, cropped_hold)
                    counter += 1

# Process all three splits
for split in ['train', 'valid', 'test']:
    print(f"Cropping holds from {split} set...")
    crop_and_save(split)

print(f"Done! Clean classification dataset saved to: {OUTPUT_PATH}")