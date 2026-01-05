import os
from langchain_community.vectorstores import FAISS

from mainfile.components.embeddings import get_huggingface_embeddings

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException
from mainfile.config.config import DB_FAISS_PATH

logger=get_logger(__name__)

def loader_faiss_vectorstore():
    try:
        logger.info("Loading FAISS vector store from disk.")
        
        embedding_model = get_huggingface_embeddings()

        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"FAISS database found at {DB_FAISS_PATH}. Loading...")
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            logger.warning(f"FAISS database not found at {DB_FAISS_PATH}. Returning None.")
    except Exception as e:
        error_message = CustomException(f"Error loading FAISS vector store: {e}")
        logger.error(str(error_message))

#new vector store save function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        
        logger.info("Generating your new vectorstore")

        embedding_model = get_huggingface_embeddings()

        db = FAISS.from_documents(text_chunks,embedding_model)

        logger.info("Saving vectorstoree")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vectostore saved sucesfulyy...")

        return db
    
    except Exception as e:
        error_message = CustomException("Failed to craete new vectorstore " , e)
        logger.error(str(error_message))