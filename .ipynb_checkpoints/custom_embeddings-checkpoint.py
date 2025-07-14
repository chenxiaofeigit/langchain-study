from langchain_core.embeddings import Embeddings
import requests
from typing import List, Union
import numpy as np

class DeepSeekEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": "deepseek-embedding"}
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class HybridEmbeddings:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.local_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        
    def embed_query(self, text: str) -> List[float]:
        # 先尝试API
        if self.api_key:
            try:
                emb = self._get_api_embedding(text)
                if emb: return emb
            except:
                pass
        # 回退到本地模型
        return self.local_model.embed_query(text)
    
    def _get_api_embedding(self, text: str) -> Union[List[float], None]:
        # 实现你的API调用逻辑
        pass