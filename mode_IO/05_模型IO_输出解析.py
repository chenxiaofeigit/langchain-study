import os
from dotenv import load_dotenv
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

def main():
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
    
    # 定义响应模式
    response_schemas = [
        ResponseSchema(name="description", description="鲜花的描述文案"),
        ResponseSchema(name="reason", description="为什么要这样写这个文案")
    ]
    
    # 创建输出解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    # 创建提示模板
    prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name}，您能提供一个吸引人的简短描述吗？
{format_instructions}"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["flower_name", "price"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # 打印提示模板结构
    print("=" * 50)
    print("提示模板内容:")
    print(prompt.template)
    print("=" * 50)
    print(f"格式指示:\n{format_instructions}")
    print("=" * 50)

    # 初始化DeepSeek模型
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )

    # 数据准备
    flowers = ["百合", "康乃馨"]
    prices = ["30", "20"]
    
    # 创建结果列表
    results = []

    for flower, price in zip(flowers, prices):
        print(f"\n处理花种: {flower}, 价格: {price}元")
        
        # 准备输入
        input_prompt = prompt.format(flower_name=flower, price=price)
        
        # 获取模型的输出
        output = model.invoke(input_prompt)
        
        # 解析模型的输出
        try:
            parsed_output = output_parser.parse(output.content)
            print(f"解析成功: {parsed_output}")
            
            # 添加额外字段
            parsed_output["flower"] = flower
            parsed_output["price"] = price
            results.append(parsed_output)
            
        except Exception as e:
            print(f"解析失败: {str(e)}")
            print(f"原始输出: {output.content}")

    # 创建DataFrame
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("最终结果:")
        print(df)
        
        # 保存到CSV
        df.to_csv("flowers_with_descriptions.csv", index=False)
        print("结果已保存到 flowers_with_descriptions.csv")
    else:
        print("没有有效结果可保存")

if __name__ == "__main__":
    main()