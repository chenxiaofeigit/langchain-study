"""
高级鲜花花语查询系统 (LangChain 1.0+ 兼容版)

使用 LangChain 和 DeepSeek 模型查询不同季节鲜花花语，采用最新 Runnable API。
"""

import os
from typing import List, Optional
from dataclasses import dataclass
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FlowerQuery:
    """鲜花查询数据类"""
    flower: str
    season: str

class FlowerMeaningAnalyzer:
    """鲜花花语分析器 (兼容LangChain 1.0+)"""
    
    def __init__(self):
        self.chain = self._initialize_chain()
        
    def _initialize_chain(self):
        """初始化处理链"""
        try:
            # 验证API密钥
            if "DEEPSEEK_API_KEY" not in os.environ:
                raise ValueError("DEEPSEEK_API_KEY 未在环境变量中设置")
            
            # 创建提示模板 (使用ChatPromptTemplate)
            prompt = ChatPromptTemplate.from_template(
                "请告诉我{flower}在{season}的花语是什么？并简要解释原因。"
            )
            
            """初始化并返回ChatOpenAI模型实例"""
            api_key = os.getenv("DEEPSEEK_API_KEY")
            # 初始化DeepSeek模型
            llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                temperature=0.3
            )
            
            # 使用新的Runnable接口
            return (
                RunnablePassthrough() 
                | prompt 
                | llm 
                | self._format_output
            )
            
        except Exception as e:
            logger.error(f"初始化处理链失败: {e}")
            raise

    def _format_output(self, response):
        """格式化输出"""
        return {"text": response}

    def query_single(self, flower: str, season: str) -> dict:
        """查询单个鲜花花语"""
        return self.chain.invoke({
            'flower': flower,
            'season': season
        })

    def batch_query(self, queries: List[FlowerQuery]) -> list:
        """批量查询花语"""
        return self.chain.batch([
            {'flower': q.flower, 'season': q.season} 
            for q in queries
        ])

def main():
    try:
        logger.info("启动鲜花花语查询系统 (LangChain 1.0+)")
        
        # 初始化分析器
        analyzer = FlowerMeaningAnalyzer()
        
        # 示例查询
        sample_queries = [
            FlowerQuery(flower="玫瑰", season="夏季"),
            FlowerQuery(flower="百合", season="春季"),
            FlowerQuery(flower="郁金香", season="秋季")
        ]
        
        # 演示查询方式
        logger.info("\n=== 单条查询 ===")
        single_result = analyzer.query_single(flower="向日葵", season="夏季")
        logger.info(f"单条查询结果: {single_result['text']}")
        
        logger.info("\n=== 批量查询 ===")
        batch_results = analyzer.batch_query(sample_queries)
        for i, result in enumerate(batch_results, 1):
            logger.info(f"{i}. {result['text']}")
            
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
    finally:
        logger.info("查询结束")

if __name__ == "__main__":
    main()