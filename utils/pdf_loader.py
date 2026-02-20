import PyPDF2
from typing import Optional

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text from a PDF file using PyPDF2.
    Returns the extracted text or None if no readable text is found.
    """
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Clean extracted text (basic)
        cleaned_text = " ".join(text.split())
        
        return cleaned_text if cleaned_text.strip() else None
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None
