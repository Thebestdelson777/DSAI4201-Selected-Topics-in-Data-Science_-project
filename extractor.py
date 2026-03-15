import fitz  # PyMuPDF
from docx import Document

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_txt(file_path)
    else:
        return ""

def extract_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()

def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()