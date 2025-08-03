import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI  # 使用新版LangChain导入方式

def generate_flower_recommendation(user_input: str):
    """
    根据用户输入生成鲜花推荐
    
    参数:
        user_input: 用户的花卉偏好描述
        
    返回:
        AI生成的鲜花推荐结果
    """
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
    
    
    # 初始化DeepSeek模型
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    # 角色设定模板
    role_template = "你是一个专业的花店电商AI助手，你的目标是帮助客户根据他们的喜好做出明智的花卉选择"
    
    # 思维链(COT)模板
    cot_template = """
## 推理过程指南
我将按照以下步骤思考：
1. 理解客户的具体需求
2. 分析花卉的颜色、花语和象征意义
3. 结合客户偏好给出个性化推荐
4. 解释推荐理由

## 参考案例
案例1:
  客户：我想找一种象征爱情的花。
  推荐：红玫瑰
  理由：红玫瑰是爱情的经典象征，红色代表热情和浓烈的感情，完美表达爱意。

案例2:
  客户：我想要一些独特和奇特的花。
  推荐：兰花
  理由：兰花外形独特、颜色鲜艳，象征奢华和独特之美，满足对独特性的追求。
"""
    
    # 创建系统消息提示模板
    system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
    system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)
    
    # 创建用户消息提示模板
    human_template = "客户需求: {human_input}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # 组合聊天提示
    chat_prompt = ChatPromptTemplate.from_messages([
        system_prompt_role,
        system_prompt_cot,
        human_prompt
    ])
    
    # 格式化提示
    prompt = chat_prompt.format_prompt(human_input=user_input).to_messages()
    
    # 打印生成的提示（调试用）
    print("=" * 60)
    print("生成的提示消息:")
    for message in prompt:
        print(f"[{message.type}]: {message.content}")
    print("=" * 60)
    
    # 调用模型生成响应
    response = llm.invoke(prompt)
    
    return response.content

if __name__ == "__main__":
    # 示例用户输入
    user_requests = [
        "我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?",
        "我需要为婚礼准备花卉，想要优雅白色系的花",
        "朋友刚升职，想送有成功寓意的花"
    ]
    
    for i, request in enumerate(user_requests, 1):
        print(f"\n{'=' * 30} 请求 #{i} {'=' * 30}")
        print(f"👤 用户: {request}")
        
        # 生成推荐
        recommendation = generate_flower_recommendation(request)
        
        print(f"\n🤖 AI推荐:")
        print(recommendation)
        print("=" * 60)
        
        # 保存结果到文件
        with open(f"flower_recommendation_{i}.txt", "w", encoding="utf-8") as f:
            f.write(recommendation)
        print(f"结果已保存到 flower_recommendation_{i}.txt")