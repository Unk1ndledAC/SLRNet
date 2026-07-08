import fitz


def get_pdf_page_size(pdf_path, page_num=0):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    rect = page.rect
    width = rect.width
    height = rect.height
    doc.close()
    return width, height


if __name__ == "__main__":
    pdf_path = "visual_comparison.pdf"
    width, height = get_pdf_page_size(pdf_path)
    print(f"PDF page size: {width} x {height}")
