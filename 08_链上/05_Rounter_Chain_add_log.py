import warnings
import os
from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
def log_chain_selection(chain_name):
    def wrapper(input_text):
        logger.info(f"正在执行 [{chain_name}] 链处理问题: {input_text}")
        return input_text
    return RunnableLambda(wrapper)

flower_care_chain = (
    log_chain_selection("鲜花护理")
    | PromptTemplate.from_template(FLOWER_CARE_TEMPLATE)
    | llm
    | StrOutputParser()
)

flower_deco_chain = (
    log_chain_selection("鲜花装饰")
    | PromptTemplate.from_template(FLOWER_DECO_TEMPLATE)
    | llm
    | StrOutputParser()
)

default_chain = (
    log_chain_selection("默认")
    | PromptTemplate.from_template(DEFAULT_TEMPLATE)
    | llm
    | StrOutputParser()
)

# 构建路由分类器
# 1. 修改路由模板，明确要求纯文本输出
ROUTER_TEMPLATE = """给定一个用户问题，将其分类到最合适的回答类别。
可选择的类别包括:
- 鲜花护理: 适合回答关于鲜花护理的问题
- 鲜花装饰: 适合回答关于鲜花装饰的问题
- 默认: 适合回答其他类型的问题

请只输出类别名称（不要包含任何XML标签）:
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

def log_router_input(input_text):
    logger.info(f"路由判断 - 原始问题: {input_text}")
    return input_text

def log_router_output(output_text):
    logger.info(f"路由判断 - 原始分类: {output_text}")
    return output_text

def log_mapped_category(mapped_category):
    if mapped_category:
        logger.info(f"路由映射 - 映射后的分类: {mapped_category}")
    else:
        logger.warning(f"路由映射 - 未识别分类, 使用默认链")
    return mapped_category

def log_final_category(final_category):
    logger.info(f"路由决策 - 最终选择: {final_category}链")
    return final_category

# 2. 添加输入提取步骤
def extract_input(data):
    """从结构化输入中提取问题文本"""
    if isinstance(data, dict) and "input" in data:
        return data["input"]
    return data

# 3. 修正映射方向
category_map = {
    "鲜花护理": "flower_care",
    "鲜花装饰": "flower_decoration"
}

# 4. 添加输出清洗步骤
def clean_llm_output(output):
    """清洗LLM输出，移除XML标签和多余空格"""
    cleaned = output.replace("<category>", "").replace("</category>", "").strip()
    # 提取第一行内容（防止LLM输出多余解释）
    return cleaned.split("\n")[0].strip()

# 更新路由链
router_chain = (
    RunnableLambda(extract_input)  # 提取问题文本
    | RunnableLambda(log_router_input)  # 记录原始问题
    | PromptTemplate.from_template(ROUTER_TEMPLATE)
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_llm_output)  # 清洗LLM输出
    | RunnableLambda(log_router_output)  # 记录清洗后的分类
    | RunnableLambda(lambda x: category_map.get(x, "default"))  # 中文→英文映射
    | RunnableLambda(log_mapped_category)  # 记录映射结果
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
    logger.info(f"\n{'='*50}")
    logger.info(f"开始处理问题: {question}")

    result = full_chain.invoke({"input": question})
    
    logger.info(f"处理完成: {question}")
    logger.info(f"{'='*50}\n")
    return result["answer"]

# 测试
if __name__ == "__main__":
    # 测试1
    print("测试1结果:", run_chain("如何为玫瑰浇水？"), end="\n\n")
    # 测试2
    print("测试2结果:", run_chain("如何为婚礼场地装饰花朵？"), end="\n\n")
    # 测试3
    print("测试3结果:", run_chain("如何区分阿拉比卡咖啡豆和罗布斯塔咖啡豆？"), end="\n\n")