import warnings
import os
from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 忽略警告
warnings.filterwarnings('ignore')


# 定义提示模板
FLOWER_CARE_TEMPLATE = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

FLOWER_DECO_TEMPLATE = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

DEFAULT_TEMPLATE = """
你是一个专业助手，请回答下面的问题:
{input}
"""

# 初始化语言模型
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.3
)

# 构建目标链 (使用LCEL语法)
flower_care_chain = (
    PromptTemplate.from_template(FLOWER_CARE_TEMPLATE)
    | llm
    | StrOutputParser()
)

flower_deco_chain = (
    PromptTemplate.from_template(FLOWER_DECO_TEMPLATE)
    | llm
    | StrOutputParser()
)

default_chain = (
    PromptTemplate.from_template(DEFAULT_TEMPLATE)
    | llm
    | StrOutputParser()
)

# 构建路由分类器
ROUTER_TEMPLATE = """给定一个用户问题，将其分类到最合适的回答类别。\
可选择的类别包括:
- 鲜花护理: 适合回答关于鲜花护理的问题
- 鲜花装饰: 适合回答关于鲜花装饰的问题
- 默认: 适合回答其他类型的问题

请严格按以下格式回答:
<category>

示例:
问题: 如何为玫瑰浇水？
回答: 鲜花护理

问题: 如何为婚礼场地装饰花朵？
回答: 鲜花装饰

问题: 什么是光合作用？
回答: 默认

现在开始:
问题: {input}
回答: """


#整个router_chain的作用是：
#   - 接收一个包含"input"键的字典（在运行时，这个input就是用户的问题）。
#   - 根据路由模板构建提示，询问LLM应该将问题分类到哪个类别（返回的字符串是中文类别名）。
#   - 将LLM返回的中文类别名映射成内部使用的键（英文），如果映射不到，则转为None。
#   - 然后检查这个键是否是有效的（在["flower_care", "flower_decoration"]中），如果是则保留，否则就返回"default"。
router_chain = (
    PromptTemplate.from_template(ROUTER_TEMPLATE)
    | llm
    | StrOutputParser()
    | {"flower_care": "鲜花护理", "flower_decoration": "鲜花装饰"}.get
    | RunnableLambda(lambda x: x if x in ["flower_care", "flower_decoration"] else "default")
)

# 构建主执行链
branch = RunnableBranch(
    (lambda x: x["category"] == "flower_care", flower_care_chain),
    (lambda x: x["category"] == "flower_decoration", flower_deco_chain),
    default_chain
)

full_chain = RunnableParallel(
    {"input": itemgetter("input"), "category": router_chain}
) | {
    "input": lambda x: x["input"],
    "category": lambda x: x["category"],
    "answer": branch
}

# 封装执行函数
def run_chain(question):
    result = full_chain.invoke({"input": question})
    return result["answer"]

# 测试
if __name__ == "__main__":
    # 测试1
    print("测试1结果:", run_chain("如何为玫瑰浇水？"), end="\n\n")
    # 测试2
    print("测试2结果:", run_chain("如何为婚礼场地装饰花朵？"), end="\n\n")
    # 测试3
    print("测试3结果:", run_chain("如何区分阿拉比卡咖啡豆和罗布斯塔咖啡豆？"), end="\n\n")