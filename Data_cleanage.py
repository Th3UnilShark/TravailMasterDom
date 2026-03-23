import os
from pathlib import Path
import shutil

# --- CONFIGURATION ---
source_dir = Path("stab_data_unsorted")
output_dir = Path("stab_data")

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

def fix_image_labels():
    count_processed = 0
    
    # Iterate through all jpg files
    for file_path in source_dir.glob("*.jpg"):
        filename = file_path.stem  # Get name without .jpg extension
        parts = filename.split("_") # Split by underscore
        
        # Expected structure: [Smooth, Type1, Type2, Type3, Number]
        if len(parts) != 5:
            continue
            
        label, t1, t2, t3, num_str = parts
        num_int = int(num_str)
        
        # LOGIC:
        # 0001 - 0009: Keep 'Smooth', Keep number
        if 1 <= num_int <= 9:
            new_label = "Smooth"
            new_num = num_str # Keep "0001" format
            
        # 0010 - 0018: Change to 'Serrated', Reset number to 0001-0009
        elif 10 <= num_int <= 18:
            new_label = "Serrated"
            # Adjust number: 10 becomes 1, 11 becomes 2, etc.
            new_num = f"{num_int - 9:04d}" 
            
        else:
            # Skip files that don't fall into your 1-18 range
            continue

        # Construct new filename
        new_filename = f"{new_label}_{t1}_{t2}_{t3}_{new_num}.jpg"
        destination = output_dir / new_filename
        
        # Copy the file to the new folder
        shutil.copy2(file_path, destination)
        count_processed += 1
        print(f"Mapped: {filename} -> {new_filename}")

    print(f"\nFinished! Processed {count_processed} images into: {output_dir}")

if __name__ == "__main__":
    fix_image_labels()