import os
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, 
    UnstructuredMarkdownLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def load_and_split(folder_path, chunk_size, chunk_overlap):
    """扫描文件夹，读取多格式文档并切分"""
    all_docs = []
    loaders = {
        ".pdf": PyPDFLoader, 
        ".docx": UnstructuredWordDocumentLoader, 
        ".md": UnstructuredMarkdownLoader, 
        ".txt": lambda path: TextLoader(path, encoding="utf-8") # 👈 改成这样
    }
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []

    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[-1].lower()
        if ext in loaders:
            loader = loaders[ext](os.path.join(folder_path, file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file
            all_docs.extend(docs)
            
    if not all_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(all_docs)

def get_db(domain_key, domains_config, embedding_model, chunk_size, chunk_overlap):
    """获取或创建 FAISS 数据库"""
    folder, index_path = domains_config[domain_key]
    
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    
    # 否则重建
    split_docs = load_and_split(folder, chunk_size, chunk_overlap)
    if not split_docs:
        return None
    
    db = FAISS.from_documents(split_docs, embedding_model)
    db.save_local(index_path)
    return db