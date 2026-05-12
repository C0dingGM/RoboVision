#!/usr/bin/env python3
"""
Convert HEIC images to JPG for labeling.
"""
from pathlib import Path
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(heic_path, output_dir):
    """Convert HEIC image to JPG."""
    try:
        heif_file = pillow_heif.read_heif(str(heic_path))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        
        output_path = output_dir / f"{Path(heic_path).stem}.jpg"
        image.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"✗ Error converting {heic_path.name}: {e}")
        return False

def main():
    # Setup paths
    source_dir = Path("yolo-3:15/chess_images")
    output_dir = Path("dataset/images/unlabeled")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🖼️  Converting HEIC images to JPG...\n")
    
    # Convert HEIC images
    heic_files = list(source_dir.glob("*.HEIC")) + list(source_dir.glob("*.heic"))
    converted = 0
    
    for heic_file in heic_files:
        if convert_heic_to_jpg(heic_file, output_dir):
            converted += 1
            if converted % 20 == 0:
                print(f"✓ Converted {converted}/{len(heic_files)} images...")
    
    print(f"\n✅ Complete! Converted {converted} HEIC images to JPG")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"\n💡 Next step: Run 'labelImg {output_dir.absolute()}' to start labeling")

if __name__ == "__main__":
    main()
