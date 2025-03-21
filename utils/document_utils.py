import os
import json
import re
import logging
from markdown import markdown
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.document_loaders import TextLoader
from typing import List, Dict, Any
import traceback

def strip_markdown(text: str) -> str:
    """Simplify markdown text to plain text"""
    html = markdown(text)
    return re.sub(r'<.*?>', '', html)

def extract_json_strings(data: Any) -> str:
    """Recursively extract all strings from JSON data"""
    if isinstance(data, dict):
        return " ".join(extract_json_strings(v) for v in data.values())
    if isinstance(data, list):
        return " ".join(extract_json_strings(v) for v in data)
    return str(data) if isinstance(data, str) else ""

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using PyPDF2"""
    try:
        reader = PdfReader(file_path)
        pdf_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text.append(text)
        return "\n".join(pdf_text)
    except Exception as e:
        logging.error(f"Error extracting PDF text from {file_path}: {e}")
        return ""

def extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX file using python-docx"""
    try:
        doc = DocxDocument(file_path)
        doc_text = []
        for para in doc.paragraphs:
            doc_text.append(para.text)
        return "\n".join(doc_text)
    except Exception as e:
        logging.error(f"Error extracting DOCX text from {file_path}: {e}")
        return ""

def load_and_clean(file_path: str) -> str:
    """Load and clean file content based on file type"""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".md":
            # Markdown files - convert to plain text
            with open(file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            return strip_markdown(markdown_content)
        elif ext == ".json":
            # JSON files - extract strings from the JSON object
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return extract_json_strings(data)
        elif ext == ".pdf":
            # PDF files - extract text using PyPDF2
            return extract_pdf_text(file_path)
        elif ext == ".docx":
            # DOCX files - extract text using python-docx
            return extract_docx_text(file_path)
        elif ext == ".txt":
            # For text files, just load directly
            loader = TextLoader(file_path)
            return loader.load()[0].page_content
        else:
            logging.warning(f"Unsupported file type: {file_path}")
            return ""
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return ""