import logging
import os
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from models.website_function import WebsiteFunction
from pydantic import BaseModel, Field
from typing import List
from langchain_core.runnables import RunnableLambda
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
)

from playwright.async_api import async_playwright

async def create_playwright_toolkit():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    return PlayWrightBrowserToolkit(async_browser=browser)

logger = logging.getLogger("WebsiteTester")

class FunctionIdentifier:
    """使用LLM和Playwright工具包识别网站核心功能点"""
 
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3
        )
        
        # 包装模型定义移到外部
        self.parser = JsonOutputParser(pydantic_object=self.WebsiteFunctionList)
        self.toolkit = None
        
        self.format_instructions = self.parser.get_format_instructions()

        self.prompt_template = ChatPromptTemplate.from_template(
            "你是一个专业的网站测试工程师。分析以下网站内容，识别主要功能点:\n\n"
            "网站URL: {url}\n"
            "HTML结构摘要:\n{html_summary}\n\n"
            "要求:\n"
            "1. 识别5-8个核心功能点\n"
            "2. 每个功能点包含名称、描述、关键元素选择器和测试步骤\n"
            "3. 元素选择器使用CSS选择器格式\n"
            "4. 测试步骤应具体可操作\n"
            "5. 为每个功能点赋予优先级(1-最高, 3-最低)\n\n"
            "返回格式: {format_instructions}"
        )
        
        self.chain = (
            RunnablePassthrough.assign(
                html_summary=lambda x: self._summarize_html(x["html"]),
                format_instructions=lambda _: self.format_instructions
            )
            | self.prompt_template
            | self.llm
            | self.parser
            | RunnableLambda(self._extract_functions)  # 使用RunnableLambda包装
        )
        logger.info(f"Initialized FunctionIdentifier with model: deepseek and PlayWrightBrowserToolkit")

    # 包装模型定义
    class WebsiteFunctionList(BaseModel):
        functions: List[WebsiteFunction] = Field(description="功能点列表")

    # 修复的提取方法
    def _extract_functions(self, parsed_output):
        """从解析输出中提取功能点列表并转换为对象"""
        try:
            # 1. 尝试提取功能点列表
            if isinstance(parsed_output, dict) and 'functions' in parsed_output:
                function_list = parsed_output['functions']
            elif hasattr(parsed_output, 'functions'):
                function_list = parsed_output.functions
            else:
                logger.error(f"无法提取功能点列表，解析输出格式无效: {parsed_output}")
                return []
            
            # 2. 将字典转换为WebsiteFunction对象
            functions = []
            for func_dict in function_list:
                if isinstance(func_dict, WebsiteFunction):
                    functions.append(func_dict)
                elif isinstance(func_dict, dict):
                    # 确保字典包含所有必需字段
                    if all(key in func_dict for key in ['name', 'description', 'selector', 'test_steps', 'priority']):
                        functions.append(WebsiteFunction(**func_dict))
                    else:
                        logger.warning(f"功能点字典缺少必需字段: {func_dict}")
                else:
                    logger.error(f"未知的功能点格式: {type(func_dict)}")
            
            return functions
        except Exception as e:
            logger.error(f"提取功能点时出错: {str(e)}")
            return []
        
    async def setup_toolkit(self):
        """异步设置Playwright工具包"""
        if self.toolkit is None:
            self.toolkit = await create_playwright_toolkit()
            logger.info("Playwright工具包已初始化")

    def _summarize_html(self, html: str, max_elements: int = 50) -> str:
        """创建结构化的HTML摘要"""
        try:
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            
            summary_lines = []
            elements = soup.find_all(
                ["h1", "h2", "h3", "h4", "button", "form", "a", "input"]
            )
            
            for i, element in enumerate(elements[:max_elements]):
                if element.name.startswith("h"):
                    summary_lines.append(
                        f"[标题{element.name}] {element.text.strip()}"
                    )
                elif element.name == "button":
                    summary_lines.append(f"[按钮] {element.text.strip()}")
                elif element.name == "form":
                    inputs = [
                        i.get("name") or i.get("id") or i.get("type")
                        for i in element.find_all("input")
                    ]
                    summary_lines.append(
                        f"[表单] 字段: {', '.join(filter(None, inputs))}"
                    )
                elif element.name == "a" and "href" in element.attrs:
                    summary_lines.append(
                        f"[链接] {element.text.strip()} -> {element['href']}"
                    )
                elif element.name == "input":
                    input_type = element.get("type", "text")
                    placeholder = element.get("placeholder", "")
                    summary_lines.append(
                        f"[输入框] type={input_type}, placeholder='{placeholder}'"
                    )
            return "\n".join(summary_lines)
        except Exception as e:
            logger.error(f"HTML摘要生成失败: {str(e)}")
            return "无法生成HTML摘要"

    async def fetch_page_content(self, url: str) -> dict:
        """使用Playwright工具包获取页面内容和结构"""
        logger.info(f"开始获取页面内容: {url}")
        try:
            # 确保工具包已初始化
            if self.toolkit is None:
                await self.setup_toolkit()
                
            # 找到导航工具
            navigate_tool = next((t for t in self.toolkit.get_tools() if t.name == "navigate_browser"), None)
            if not navigate_tool:
                raise ValueError("找不到导航工具")
                
            # 执行导航
            result = await navigate_tool.arun(url=url)
            html = result["page_content"]
            title = result["page_title"]
            
            return {"url": url, "html": html, "title": title}
        except Exception as e:
            logger.error(f"获取页面内容失败: {str(e)}")
            return {
                "url": url,
                "html": "<html><body>无法获取页面内容</body></html>",
                "title": "获取失败",
            }

    async def identify_functions(self, url: str) -> list:
        """识别网站功能点"""
        logger.info(f"开始识别功能点: {url}")
        try:
            # 确保工具包已初始化
            if self.toolkit is None:
                await self.setup_toolkit()
                
            page_data = await self.fetch_page_content(url)
            result = await self.chain.ainvoke(page_data)
            
            # 结果应该是WebsiteFunction对象列表
            if not result:
                logger.error("未识别到任何功能点")
                return []
            
            # 确保是列表类型
            if not isinstance(result, list):
                result = [result]
            
            # 验证结果类型
            valid_functions = []
            for func in result:
                if isinstance(func, WebsiteFunction):
                    valid_functions.append(func)
                else:
                    logger.warning(f"无效的功能点类型: {type(func)}")
            
            logger.info(f"成功识别 {len(valid_functions)} 个功能点")
            
            # 按优先级排序
            return sorted(valid_functions, key=lambda x: x.priority)
        except Exception as e:
            logger.error(f"功能点识别失败: {str(e)}", exc_info=True)
            return []