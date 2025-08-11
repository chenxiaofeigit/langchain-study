"""
鲜花内容生成系统 (LangChain 1.0+ 修正版)
使用 DeepSeek 模型生成鲜花介绍、评论和社交媒体文案
"""
import os
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI

class FlowerContentGenerator:
    """鲜花内容生成器 (兼容LangChain 1.0+)"""
    
    def __init__(self):

        # 初始化 OpenAI 模型（用于前两步）
        self.chatglm_llm = ChatOpenAI(
            model="glm-4",  # 或 "glm-3-turbo"
            openai_api_base="https://open.bigmodel.cn/api/paas/v4",
            api_key=os.getenv("CHATGLM_API_KEY"),
            temperature=0.3,
        )

        # 初始化qwen-plus模型
        self.qwen_llm = ChatOpenAI(
            model_name="qwen-plus",  # 或 qwen-plus/qwen-max
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", 
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.3
        )


        # 初始化DeepSeek模型
        self.deepseek_llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3
        )

        self.full_chain = self._build_chain()
    
    def _build_chain(self) -> RunnableSequence:
        """构建处理链"""
        # 第一步：生成鲜花介绍
        introduction_prompt = ChatPromptTemplate.from_template(
            """你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
            花名: {name}
            颜色: {color}
            植物学家: 这是关于上述花的介绍:"""
                    )
        introduction_chain = introduction_prompt | self.chatglm_llm
        
        # 第二步：生成鲜花评论
        review_prompt = ChatPromptTemplate.from_template(
            """你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
            鲜花介绍:
            {introduction}
            花评人对上述花的评论:"""
                    )
        review_chain = review_prompt | self.qwen_llm
        
        # 第三步：生成社交媒体文案
        social_prompt = ChatPromptTemplate.from_template(
            """你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
            鲜花介绍:
            {introduction}
            花评人对上述花的评论:
            {review}
            社交媒体帖子:"""
                    )
        social_chain = social_prompt | self.deepseek_llm
        
        # 组合完整链条
        return (
            RunnablePassthrough()
            | {
                "introduction": introduction_chain,
                "name": RunnablePassthrough(),
                "color": RunnablePassthrough()
            }
            | {
                "review": review_chain,
                "introduction": RunnablePassthrough(),
                "name": RunnablePassthrough(),
                "color": RunnablePassthrough()
            }
            | {
                "social_post_text": social_chain,
                "introduction": RunnablePassthrough(),
                "review": RunnablePassthrough()
            }
        )
    
    def generate_content(self, name: str, color: str) -> Dict[str, str]:
        """生成完整鲜花内容"""
        # 注意这里直接调用链本身，而不是字典
        return self.full_chain.invoke({
            "name": name,
            "color": color
        })

def main():
    try:
        # 初始化生成器
        generator = FlowerContentGenerator()
        
        # 生成内容
        result = generator.generate_content(
            name="玫瑰",
            color="黑色"
        )
        
        # 打印结果
        print("\n=== 鲜花介绍 ===")
        print(result["introduction"])
        
        print("\n=== 鲜花评论 ===")
        print(result["review"])
        
        print("\n=== 社交媒体文案 ===")
        print(result["social_post_text"])
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()