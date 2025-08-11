import os
from langchain import PromptTemplate, OpenAI, LLMChain


def configure_environment():
    """配置环境变量"""
    os.environ["OPENAI_API_KEY"] = "Your Key"  # 替换为你的实际API密钥


def create_llm_chain() -> LLMChain:
    """创建并返回LLMChain实例"""
    template = "{flower}的花语是?"
    llm = OpenAI(temperature=0)  # 确定性输出
    
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(template))
    

def query_flower_meaning(llm_chain: LLMChain, flower_name: str) -> dict:
    """查询指定鲜花的花语"""
    return llm_chain(flower_name)


def main():
    try:
        # 初始化配置
        configure_environment()
        
        # 创建LLM链
        llm_chain = create_llm_chain()
        
        # 查询花语
        result = query_flower_meaning(llm_chain, "玫瑰")
        
        # 输出结果
        print(f"查询结果: {result['text']}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()