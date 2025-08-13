import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    # 初始化语言模型
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    # 创建带记忆的链 (LCEL规范)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=300,
        return_messages=True
    )
    
    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的聊天助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # 创建完整的对话链
    chain = (
        RunnablePassthrough.assign(
            history=lambda _: memory.load_memory_variables({})["history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 封装对话函数
    def converse(input_text):
        response = chain.invoke({"input": input_text})
        # 保存上下文到记忆
        memory.save_context({"input": input_text}, {"output": response})
        return response
    
    # 第一天的对话
    # 回合1
    result = converse("我姐姐明天要过生日，我需要一束生日花束。")
    print(result)
    
    # 回合2
    result = converse("她喜欢粉色玫瑰，颜色是粉色的。")
    print(result)
    
    # 第二天的对话
    # 回合3
    result = converse("我又来了，还记得我昨天为什么要来买花吗？")
    print(result)

if __name__ == "__main__":
    main()