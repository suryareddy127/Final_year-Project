from langchain_huggingface import HuggingFaceEmbeddings

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger=get_logger(__name__)

def get_huggingface_embeddings():
    try:
        logger.info("Initializing HuggingFace Embeddings.")
       
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        logger.info("HuggingFace Embeddings initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing HuggingFace Embeddings: {e}")
        return []