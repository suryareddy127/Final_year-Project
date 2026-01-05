import os
from mainfile.components.document_loader import load_pdf_files, split_documents
from mainfile.components.vectorstore import save_vector_store
from mainfile.config.config import DB_FAISS_PATH
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger = get_logger(__name__)

def processandstore_pdf():
    try:
        logger.info("Starting process to load, split, and store PDF documents.")
        
        documents = load_pdf_files()

        text_chunks = split_documents(documents)

        save_vector_store(text_chunks)

        logger.info("Process completed successfully.")

    except Exception as e:
        error_message = CustomException(f"Error in processing and storing PDF documents: {e}")
        logger.error(str(error_message))

if __name__ == "__main__":
    processandstore_pdf()