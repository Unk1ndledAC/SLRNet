import fitz
import os


def pdf_to_images(pdf_path, output_folder, zoom=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)

    print(f"Converting {pdf_path}, total {len(doc)} pages...")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        output_path = os.path.join(f"{output_folder}.png")
        pix.save(output_path)

    doc.close()
    print("Conversion completed!")


if __name__ == "__main__":
    pdf_to_images("SOTA-Table.pdf", "SOTA-Table", zoom=10)
