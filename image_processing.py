import base64
import io
from PIL import Image

def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def split_image_text_types(docs):
    images = []
    text = []
    for doc in docs:
        doc_content = doc.page_content
        if is_base64(doc_content):
            images.append(resize_base64_image(doc_content, size=(250, 250)))
        else:
            text.append(doc_content)
    return {"images": images, "texts": text}
