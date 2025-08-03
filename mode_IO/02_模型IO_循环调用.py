import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def main():
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
    
    # 创建提示模板
    template = """
    您是一位专业的鲜花店文案撰写员。
    对于售价为 {price} 元的 {flower_name}，您能提供一个吸引人的简短描述吗？
    """
    prompt = PromptTemplate.from_template(template)
    
    # 打印提示模板结构
    print(f"提示模板内容:\n{prompt.template}\n")
    
    # 初始化DeepSeek模型
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    flowers = ["百合", "康乃馨"]
    prices = ["30", "20"]

    # 生成多种花的文案
    for flower, price in zip(flowers, prices):
        # 使用提示模板生成输入
        input_prompt = prompt.format(flower_name=flower, price=price)

        # 得到模型的输出
        output = model.invoke(input_prompt)

        # 打印输出内容
        print(output)

if __name__ == "__main__":
    main()