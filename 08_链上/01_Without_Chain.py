"""
OpenAI 花语查询工具

该脚本使用LangChain查询指定鲜花的花语信息。
"""

import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


def main():
    # 设置OpenAI API密钥
    os.environ["OPENAI_API_KEY"] = 'Your Key'  # 替换为您的实际API密钥
    
    try:
        # ---- 第一步 创建提示 ----
        # 原始字符串模板
        template = "{flower}的花语是?"
        
        # 创建LangChain提示模板
        prompt_template = PromptTemplate.from_template(template)
        
        # 根据模板创建具体提示
        prompt = prompt_template.format(flower="玫瑰")
        
        # 打印提示内容
        print("生成的提示:", prompt)

        # ---- 第二步 创建并调用模型 ----
        # 创建模型实例 (设置temperature=0以获得确定性结果)
        model = OpenAI(temperature=0)
        
        # 传入提示调用模型
        result = model(prompt)
        
        # 打印模型返回结果
        print("查询结果:", result)
        
    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()