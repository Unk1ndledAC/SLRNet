import fitz


def crop_by_margins(input_path, output_path, margins, page_num=0):
    doc = fitz.open(input_path)
    page = doc[page_num]

    original_rect = page.rect

    left_margin = margins[0]
    right_margin = margins[1]
    top_margin = margins[2]
    bottom_margin = margins[3]

    new_x0 = original_rect.x0 + left_margin
    new_y0 = original_rect.y0 + top_margin
    new_x1 = original_rect.x1 - right_margin
    new_y1 = original_rect.y1 - bottom_margin

    if new_x0 >= new_x1 or new_y0 >= new_y1:
        print("Error: crop margins too large, resulting crop region is invalid!")
        doc.close()
        return

    crop_rect = (new_x0, new_y0, new_x1, new_y1)
    page.set_cropbox(crop_rect)

    doc.save(output_path)
    doc.close()
    print(f"Crop completed! Saved to: {output_path}")
    print(f"Crop region coordinates: {crop_rect}")


if __name__ == "__main__":
    input_pdf = "SOTA-Table.pdf"
    output_pdf = "SOTA-Table crop.pdf"
    my_margins = (41, 41, 195, 187)
    crop_by_margins(input_pdf, output_pdf, my_margins)
