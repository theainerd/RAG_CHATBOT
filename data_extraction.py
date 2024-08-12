import os
from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(pdf_path, output_dir, max_chars=4000, new_after_n_chars=3800, combine_text_under_n_chars=2000):
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=max_chars,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
        image_output_dir_path=output_dir,
    )

def categorize_elements_by_type(elements):
    tables = []
    texts = []
    for element in elements:
        element_type = str(type(element))
        if "CompositeElement" in element_type:
            texts.append(str(element))
    return texts


