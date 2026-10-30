from . import  txt_loader, pdf_loader, csv_loader, docx_loader, excel_loader

LOADERS = {
    ".pdf":  pdf_loader.load,
    ".txt":  txt_loader.load,
    ".csv":  csv_loader.load,
    ".docx": docx_loader.load,
    ".xlsx": excel_loader.load,
    ".xls":  excel_loader.load,
}

def load_file(path: str) -> str:
    ext = "." + path.rsplit(".", 1)[-1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)