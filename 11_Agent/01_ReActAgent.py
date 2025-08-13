import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI  # 使用正确的导入
from langchain.chains import LLMMathChain

def main():
    # 设置API密钥
    os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
    
    # 初始化大语言模型 - 使用正确的导入
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    # 创建工具集
    search = SerpAPIWrapper()
    
    # 创建数学链（不需要LLMMathChain）
    def math_calculator(query: str) -> str:
        """用于数学计算，特别是百分比计算"""
        try:
            # 从查询中提取数字和操作
            if "加价" in query or "加" in query:
                parts = query.replace("%", "").split()
                base = float(parts[0])
                percentage = float(parts[2])
                result = base * (1 + percentage/100)
                return f"{result:.2f}"
            else:
                # 简单计算逻辑
                return str(eval(query))
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="使用搜索引擎获取当前市场信息"
        ),
        Tool(
            name="Calculator",
            func=math_calculator,
            description="用于数学计算，特别是百分比计算"
        )
    ]
    
    # 创建代理提示模板
    prompt = hub.pull("hwchase17/react")
    
    # 创建代理
    agent = create_react_agent(llm, tools, prompt)
    
    # 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 执行代理查询
    result = agent_executor.invoke({
        "input": "目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？"
    })
    
    print("\n最终结果:")
    print(result["output"])

if __name__ == "__main__":
    # 检查并安装缺失的依赖
    try:
        import numexpr
    except ImportError:
        print("安装缺失的依赖: numexpr")
        import subprocess
        subprocess.run(["pip", "install", "numexpr"])
    
    main()