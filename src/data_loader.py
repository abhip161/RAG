from pathlib import Path 
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader


def load_all_documents(data_dir: str) -> List[Any]:
    """Load all documents from the specified directory. and convert to Langchain document"""
    # use project root data folder
    data_path = Path(data_dir).resolve() 
    print(f"[DEBUG] Data path: {data_path}") 
    documents = [] 

    ## pdf files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] found {len(pdf_files)} pdf files:{[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading pdf: {pdf_file}") 
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} pages from {pdf_file}")
            documents.extend(loaded) 
        except Exception as e:
            print(f"[ERROR] Failed to load pdf {pdf_file}: {e}")

    ## txt files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] found {len(txt_files)} txt files:{[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading txt: {txt_file}") 
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} pages from {txt_file}")
            documents.extend(loaded) 
        except Exception as e:
            print(f"[ERROR] Failed to load txt {txt_file}: {e}")
    return documents
            