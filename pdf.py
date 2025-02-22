import os
import sys
from PIL import Image

def main():
    # Determine input folder and output file from command-line arguments (or use defaults)
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "input"
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else "output.pdf"

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Gather all image files with common extensions
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
    image_files = [os.path.join(input_folder, f)
                   for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]

    image_files.sort()  # Sort the files alphabetically

    if not image_files:
        print("No images found in the input folder.")
        sys.exit(1)

    images = []
    for file in image_files:
        try:
            img = Image.open(file)
            # Ensure image is in RGB mode (PDF requires RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error opening {file}: {e}")

    if not images:
        print("No valid images were found to create the PDF.")
        sys.exit(1)

    # Save the images to a single PDF file.
    try:
        # The first image is used as the base, and the rest are appended
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"PDF created successfully: '{output_pdf}'")
    except Exception as e:
        print(f"Error saving PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
