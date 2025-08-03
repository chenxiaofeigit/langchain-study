import os
import logging
import time
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 配置日志系统
def setup_logger():
    """配置并返回日志记录器"""
    # 创建日志目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"flower_rag_{timestamp}.log")
    
    # 创建日志记录器
    logger = logging.getLogger("FlowerRAG")
    logger.setLevel(logging.DEBUG)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 应用格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 获取日志记录器
logger = setup_logger()

class FlowerRAGSystem:
    def __init__(self, data_dir='./OneFlower', api_key=None, 
                 chunk_size=200, chunk_overlap=10, search_k=3,
                 hf_home="/home/fiona/data/hf_cache"):
        """
        初始化鲜花知识RAG系统
        
        参数:
        data_dir: 存放文档的目录路径
        api_key: DeepSeek API密钥
        chunk_size: 文本分割大小
        chunk_overlap: 文本分割重叠大小
        search_k: 检索相关文档数量
        hf_home: HuggingFace缓存目录
        """
        logger.info("初始化 FlowerRAG 系统开始")
        logger.debug(f"参数: data_dir={data_dir}, chunk_size={chunk_size}, "
                     f"chunk_overlap={chunk_overlap}, search_k={search_k}, "
                     f"hf_home={hf_home}")
        
        # 设置 HuggingFace 缓存目录
        if hf_home:
            os.environ["HF_HOME"] = hf_home
            logger.info(f"设置 HF_HOME 环境变量: {hf_home}")
        
        self.data_dir = data_dir
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_k = search_k
        self.hf_home = hf_home
        
        if not self.api_key:
            error_msg = "DeepSeek API密钥未提供，请设置环境变量DEEPSEEK_API_KEY或传入api_key参数"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 初始化组件
        self.documents = None
        self.splits = None
        self.vectorstore = None
        self.qa_chain = None
        
        # 自动构建RAG管道
        try:
            self._build_rag_pipeline()
            logger.info("FlowerRAG 系统初始化成功")
        except Exception as e:
            logger.exception("FlowerRAG 系统初始化失败")
            raise
    
    def _load_documents(self):
        """加载指定目录中的所有文档"""
        logger.info(f"开始加载文档，目录: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            error_msg = f"文档目录不存在: {self.data_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        documents = []
        file_count = 0
        supported_files = []
        
        for file in os.listdir(self.data_dir):    
            file_path = os.path.join(self.data_dir, file)    
            if file.endswith('.pdf'):        
                loader = PyPDFLoader(file_path)
                try:
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(file)
                    file_count += 1
                    logger.debug(f"成功加载 PDF 文件: {file}, 文档数: {len(docs)}")
                except Exception as e:
                    logger.error(f"加载 PDF 文件 {file} 失败: {str(e)}")
            elif file.endswith('.docx'):         
                loader = Docx2txtLoader(file_path)
                try:
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(file)
                    file_count += 1
                    logger.debug(f"成功加载 DOCX 文件: {file}, 文档数: {len(docs)}")
                except Exception as e:
                    logger.error(f"加载 DOCX 文件 {file} 失败: {str(e)}")
            elif file.endswith('.txt'):       
                loader = TextLoader(file_path)
                try:
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(file)
                    file_count += 1
                    logger.debug(f"成功加载 TXT 文件: {file}, 文档数: {len(docs)}")
                except Exception as e:
                    logger.error(f"加载 TXT 文件 {file} 失败: {str(e)}")
            else:
                logger.warning(f"跳过不支持的文件类型: {file}")
        
        if not documents:
            error_msg = f"在 {self.data_dir} 目录下未找到任何可加载的文档"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"成功加载 {file_count} 个文件，共 {len(documents)} 个文档片段")
        logger.debug(f"加载的文件列表: {', '.join(supported_files)}")
        return documents
    
    def _split_documents(self, docs):
        """分割文档为适合处理的块"""
        logger.info("开始分割文档")
        logger.debug(f"分割参数: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        splits = splitter.split_documents(docs)
        logger.info(f"文档分割完成，生成 {len(splits)} 个文本块")
        
        # 记录前5个块的样本
        if logger.isEnabledFor(logging.DEBUG):
            for i, chunk in enumerate(splits[:5]):
                logger.debug(f"文本块 #{i+1} (长度: {len(chunk.page_content)}): "
                             f"{chunk.page_content[:100]}...")
        
        return splits
    
    def _create_vector_store(self, splits):
        """创建向量存储"""
        logger.info("开始创建向量存储")
        
        try:
            # 使用本地嵌入模型
            logger.debug(f"初始化嵌入模型: BAAI/bge-m3 (使用缓存: {self.hf_home})")
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",  # 推荐的中文模型
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 在内存中创建向量库
            logger.debug("创建 Qdrant 向量存储")
            vectorstore = Qdrant.from_documents(
                documents=splits,
                embedding=embeddings,
                location=":memory:",
                collection_name="flower_knowledge"
            )
            
            logger.info("向量存储创建成功")
            return vectorstore
        except Exception as e:
            logger.exception("创建向量存储失败")
            raise
    
    def _build_qa_chain(self, vectorstore):
        """构建问答链"""
        logger.info("开始构建问答链")
        
        try:
            # 初始化DeepSeek模型
            logger.debug("初始化 DeepSeek 聊天模型")
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                openai_api_base="https://api.deepseek.com/v1",
                model_name="deepseek-chat",
                temperature=0.3
            )

            # 定义提示模板
            prompt = ChatPromptTemplate.from_template(
                """你是一名“易速鲜花”客服助手，请严格根据以下上下文回答问题：
                ---
                上下文：{context}
                ---
                问题：{question}
                答案："""
            )

            # 构建检索器
            logger.debug(f"创建检索器，k={self.search_k}")
            retriever = vectorstore.as_retriever(search_kwargs={"k": self.search_k})
            
            # 组合链
            qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()} 
                | prompt 
                | llm 
            )
            
            logger.info("问答链构建成功")
            return qa_chain
        except Exception as e:
            logger.exception("构建问答链失败")
            raise
    
    def _build_rag_pipeline(self):
        """构建完整的RAG管道"""
        logger.info("开始构建 RAG 管道")
        start_time = time.time()
        
        self.documents = self._load_documents()
        self.splits = self._split_documents(self.documents)
        self.vectorstore = self._create_vector_store(self.splits)
        self.qa_chain = self._build_qa_chain(self.vectorstore)
        
        elapsed = time.time() - start_time
        logger.info(f"RAG 管道构建完成，耗时 {elapsed:.2f} 秒")
    
    def query(self, question):
        """
        向RAG系统提问
        
        参数:
        question: 问题字符串
        
        返回:
        包含答案的LangChain响应对象
        """
        if not question or not question.strip():
            logger.warning("收到空问题")
            return {"error": "问题不能为空"}
            
        logger.info(f"收到问题: '{question}'")
        start_time = time.time()
        
        if not self.qa_chain:
            error_msg = "问答链未初始化，请先调用_build_rag_pipeline()"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.debug("开始处理问题...")
            response = self.qa_chain.invoke(question)
            elapsed = time.time() - start_time
            
            logger.info(f"问题处理完成，耗时 {elapsed:.2f} 秒")
            logger.debug(f"完整响应: {response}")
            
            # 记录响应内容摘要
            response_content = response.content
            if len(response_content) > 200:
                logger.debug(f"响应内容: {response_content[:200]}...")
            else:
                logger.debug(f"响应内容: {response_content}")
            
            return response
        except Exception as e:
            logger.exception(f"处理问题失败: '{question}'")
            return {"error": str(e)}
    
    def reload_documents(self):
        """重新加载文档并更新RAG系统"""
        logger.info("开始重新加载文档")
        start_time = time.time()
        
        try:
            self._build_rag_pipeline()
            elapsed = time.time() - start_time
            logger.info(f"文档重新加载完成，耗时 {elapsed:.2f} 秒")
            return True
        except Exception as e:
            logger.exception("文档重新加载失败")
            return False
    
    def get_document_count(self):
        """获取处理的文档数量"""
        count = len(self.documents) if self.documents else 0
        logger.debug(f"获取文档数量: {count}")
        return count
    
    def get_chunk_count(self):
        """获取生成的文本块数量"""
        count = len(self.splits) if self.splits else 0
        logger.debug(f"获取文本块数量: {count}")
        return count


# 示例用法
if __name__ == "__main__":
    try:
        # 初始化RAG系统
        logger.info("开始示例测试")
        rag_system = FlowerRAGSystem(
            data_dir="./OneFlower",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            chunk_size=300,
            search_k=3,
            hf_home="/home/fiona/data/hf_cache"  # 指定 HuggingFace 缓存目录
        )
        
        logger.info(f"已加载 {rag_system.get_document_count()} 个文档")
        logger.info(f"生成 {rag_system.get_chunk_count()} 个文本块")
        
        # 提问示例
        questions = [
            "玫瑰花有哪些常见品种？",
            "如何养护兰花？",
            "送女朋友生日花束推荐",
            "郁金香的花语是什么？",
            "菊花适合在什么季节种植？"
        ]
        
        for question in questions:
            logger.info(f"\n提问: {question}")
            start_time = time.time()
            response = rag_system.query(question)
            elapsed = time.time() - start_time
            
            if hasattr(response, "content"):
                logger.info(f"回答: {response.content}")
            else:
                logger.error(f"无效响应: {response}")
            
            logger.info(f"处理时间: {elapsed:.2f}秒")
            logger.info("-" * 60)
        
        logger.info("示例测试完成")
    except Exception as e:
        logger.exception("示例测试失败")