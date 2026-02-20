from typing import Optional

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """
    Extracts text from a TXT file.
    Returns the extracted text or None if no readable text is found.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading TXT file {file_path}: {e}")
        return None
