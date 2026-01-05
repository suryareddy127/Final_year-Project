import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

from mainfile.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger=get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The specified data path {DATA_PATH} does not exist.")
        logger.info(f"Loading PDF files from directory: {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            raise CustomException("No PDF files found in the directory.")
        else:
            logger.info(f"Loaded {len(documents)} documents from PDF files.")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        return []
    
def split_documents(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for splitting.")
        logger.info("Splitting documents into smaller chunks.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP )
        text_chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Splited documents into {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return []