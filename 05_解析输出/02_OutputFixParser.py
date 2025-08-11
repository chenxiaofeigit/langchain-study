"""
鲜花文案生成器 - 使用ChatOpenAI和OutputFixingParser版本

该脚本使用LangChain和OpenAI API为不同鲜花生成营销文案，
使用OutputFixingParser自动修复格式问题，
并将结果存储在DataFrame中。
"""

import os
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class FlowerDescription(BaseModel):
    """鲜花描述数据结构"""
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")


def initialize_environment() -> None:
    """初始化环境变量"""
    load_dotenv()  # 加载.env文件中的环境变量



def initialize_model() -> ChatOpenAI:
    """初始化并返回ChatOpenAI模型实例"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    chat = ChatOpenAI(
        model_name="qwen-plus",  # 或 qwen-plus/qwen-max
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", 
        openai_api_key=api_key,
        temperature=0.7
    )

      
    """初始化并返回ChatOpenAI模型实例"""
    model = ChatOpenAI(
         model="glm-4",  # 或 "glm-3-turbo"
         openai_api_base="https://open.bigmodel.cn/api/paas/v4", 
         openai_api_key=api_key,
    ) # 注意参数名改为model

    return chat


def initialize_parser(model: ChatOpenAI) -> OutputFixingParser:
    """初始化并返回OutputFixingParser"""
    original_parser = PydanticOutputParser(pydantic_object=FlowerDescription)
    return OutputFixingParser.from_llm(parser=original_parser, llm=model)


def generate_flower_descriptions(
    model: ChatOpenAI,
    parser: OutputFixingParser,
    flowers: List[str],
    prices: List[str]
) -> List[Dict]:
    """为给定的鲜花列表生成描述"""
    # 获取输出格式指示
    format_instructions = parser.parser.get_format_instructions()

    # 创建提示模板
    prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
请严格按照以下格式返回结果：
{format_instructions}"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["flower", "price"],
        partial_variables={"format_instructions": format_instructions}
    )

    results = []
    for flower, price in zip(flowers, prices):
        # 准备输入
        formatted_prompt = prompt.format(flower=flower, price=price)
        
        # 获取模型输出
        output = model.invoke(formatted_prompt)
        
        # 使用OutputFixingParser解析输出
        parsed_output = parser.parse(output.content)
        results.append(parsed_output.model_dump())

    return results


def main():
    """主函数"""
    try:
        # 初始化
        initialize_environment()
        model = initialize_model()
        parser = initialize_parser(model)

        # 数据准备
        flowers = ["玫瑰", "百合", "康乃馨"]
        prices = ["50", "30", "20"]

        # 生成描述
        descriptions = generate_flower_descriptions(model, parser, flowers, prices)

        # 创建DataFrame并输出结果
        df = pd.DataFrame(descriptions)
        print("输出的数据：", df.to_dict(orient="records"))
        
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()