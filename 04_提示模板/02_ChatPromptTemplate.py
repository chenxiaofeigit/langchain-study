import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_openai import ChatOpenAI

def generate_company_name():
    """
    为特定产品公司生成名称建议
    
    该函数:
    1. 创建系统提示和用户提示模板
    2. 组合成完整的聊天提示
    3. 调用OpenAI模型生成公司名称
    4. 返回并打印结果
    """
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
    
    # 模板构建 - 系统角色提示
    system_template = "你是一位专业顾问，负责为专注于{product}的公司起名。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    # 模板构建 - 用户输入提示
    human_template = "公司主打产品是{product_detail}，请提供3个有创意的公司名称。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # 组合提示模板
    prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    # 格式化提示消息
    prompt = prompt_template.format_prompt(
        product="鲜花装饰",
        product_detail="创新的鲜花设计"
    ).to_messages()
    
    # 打印生成的提示
    print("=" * 60)
    print("生成的提示消息:")
    for message in prompt:
        print(f"[{message.type}]: {message.content}")
    print("=" * 60)
    
 
    # 初始化DeepSeek模型
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    # 调用模型生成结果
    response = model.invoke(prompt)
    
    # 返回并打印结果
    print("生成的名称建议:")
    print(response.content)
    print("=" * 60)
    
    return response.content

if __name__ == "__main__":
    # 执行公司名称生成函数
    company_names = generate_company_name()
    
    # 保存结果到文件
    with open("company_name_suggestions.txt", "w", encoding="utf-8") as f:
        f.write(company_names)
    
    print("名称建议已保存到 company_name_suggestions.txt")