from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from mainfile.components.llm import load_llm
from mainfile.components.vectorstore import loader_faiss_vectorstore

from mainfile.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException  

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """You are a medical healthcare AI assistant that provides accurate, educational, and context-aware health information using Retrieval-Augmented Generation (RAG).
You must not diagnose or prescribe treatments.
All responses should be based on verified medical sources and clearly state that the information is for awareness only.
Encourage consulting healthcare professionals when necessary.
Use the following context to answer the question at the end.

Context:{context}

Question: {question}

Answer:"""

def set_custom_prompt():
    return ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

def format_docs(docs):
    """Format documents for prompt (handles empty case)"""
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = loader_faiss_vectorstore()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded")

        
        qa_chain = (
            {"context": db.as_retriever(search_kwargs={'k': 3}) | format_docs, 
             "question": RunnablePassthrough()}
            | set_custom_prompt()
            | llm
            | StrOutputParser()
        )

        logger.info("Successfully created the QA chain")
        return qa_chain

    except Exception as e:
       
        error_message = CustomException("Failed to make a QA chain", str(e))
        logger.error(str(error_message))
        return None