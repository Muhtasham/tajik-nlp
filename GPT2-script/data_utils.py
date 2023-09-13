import argparse
from pathlib import Path
import docx2txt
import fitz  # PyMuPDF
import logging
from time import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(filename=os.path.join(script_dir, 'file_conversion.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_docx(docx_file: Path) -> str:
    try:
        logging.info(f"Processing DOCX file: {docx_file.name}")
        return docx2txt.process(str(docx_file))
    except Exception as e:
        logging.error(f"Could not process DOCX file {docx_file.name} due to: {str(e)}")
        return None

def read_pdf(pdf_file: Path) -> str:
    try:
        logging.info(f"Processing PDF file: {pdf_file.name}")
        doc = fitz.open(pdf_file)
        text = [page.get_text() for page in doc]
        return "\n".join(filter(None, text))
    except Exception as e:
        logging.error(f"Could not process PDF file {pdf_file.name} due to: {str(e)}")
        return None

def main(args: argparse.Namespace) -> None:
    start_time = time()
    logging.info(f"Starting the conversion of DOCX and PDF files to text files.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []

    files_in_dir = list(Path(args.input_dir).rglob('*'))
    logging.info(f"Found {len(files_in_dir)} files in the directory.")
    
    processed_files_counter = 0

    for filepath in files_in_dir:
        if filepath.is_file():
            processed_file_text = None
            if filepath.suffix == '.docx':
                processed_file_text = read_docx(filepath)
            elif filepath.suffix == '.pdf':
                processed_file_text = read_pdf(filepath)
            else:
                logging.warning(f"Skipping unsupported file type: {filepath.name}")

            if processed_file_text:
                texts.append(processed_file_text)
                processed_files_counter += 1

    all_texts = "\n".join(filter(None, texts))
    
    output_file_path = output_dir / "combined_texts.txt"
    with open(output_file_path, "w", encoding='utf-8') as out_file:
        out_file.write(all_texts)

    file_size = output_file_path.stat().st_size / (1024 * 1024)  # Size in MB
    logging.info(f"The combined text file has been created with a size of {file_size:.2f} MB")
    logging.info(f"Processed {processed_files_counter} files out of {len(files_in_dir)} files found in the directory.")

    end_time = time()
    elapsed_time = end_time - start_time
    logging.info(f"The script took {elapsed_time:.2f} seconds to complete.")
    
    print(f"The combined text file has been created with a size of {file_size:.2f} MB")
    print(f"Processed {processed_files_counter} files out of {len(files_in_dir)} files found in the directory.")
    print(f"The script took {elapsed_time:.2f} seconds to complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DOCX and PDF files to text files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing DOCX and PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the text files")
    args = parser.parse_args()

    main(args)
