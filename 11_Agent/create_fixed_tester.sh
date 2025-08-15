#!/bin/bash

# 创建项目目录结构
mkdir -p website_tester
cd website_tester
mkdir -p {models,modules,core}

# 创建配置文件
cat > config.py << 'EOL'
import logging
import os

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("website_testing.log")],
    )
    return logging.getLogger("WebsiteTester")

# 确保目录存在
def create_directories():
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("browser_context", exist_ok=True)  # 浏览器上下文缓存目录
EOL

# 创建模型文件
cat > models/__init__.py << 'EOL'
# 空文件，用于包初始化
EOL

cat > models/website_function.py << 'EOL'
from pydantic import BaseModel, Field
from typing import List

class WebsiteFunction(BaseModel):
    """网站功能点描述"""
    name: str = Field(description="功能名称")
    description: str = Field(description="功能详细描述")
    elements: List[str] = Field(description="实现此功能的关键页面元素选择器列表(CSS选择器)")
    test_steps: List[str] = Field(description="测试此功能的操作步骤列表(自然语言描述)")
    priority: int = Field(description="功能优先级(1-最高, 3-最低)", default=2)
EOL

cat > models/test_result.py << 'EOL'
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

class TestStepResult(BaseModel):
    """测试步骤结果详情"""
    step_description: str
    status: str  # 'success' or 'failed'
    error: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FunctionTestResult(BaseModel):
    """功能测试结果详情"""
    function_name: str
    overall_status: str  # 'passed' or 'failed'
    step_results: List[TestStepResult] = Field(default_factory=list)
    screenshot: str = Field(default="")
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = Field(default="")
    error_summary: str = Field(default="")
EOL

cat > models/test_report.py << 'EOL'
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from .test_result import FunctionTestResult

class TestReport(BaseModel):
    """测试报告结构"""
    website_url: str
    test_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_functions: int
    passed: int
    failed: int
    execution_time: float = Field(default=0.0)  # 秒
    results: List[FunctionTestResult]
    summary: str
EOL

# 创建模块文件
cat > modules/__init__.py << 'EOL'
# 空文件，用于包初始化
EOL

cat > modules/function_identifier.py << 'EOL'
import logging
import asyncio
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.playwright.toolkit import PlayWrightBrowserToolkit
from models.website_function import WebsiteFunction

logger = logging.getLogger("WebsiteTester")

class FunctionIdentifier:
    """使用LLM和Playwright工具包识别网站核心功能点"""
    
    def __init__(self, model_name: str = "gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=WebsiteFunction)
        self.toolkit = PlayWrightBrowserToolkit()
        
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
                html_summary=lambda x: self._summarize_html(x["html"])
            )
            | self.prompt_template
            | self.llm
            | self.parser
        )
        logger.info(f"Initialized FunctionIdentifier with model: {model_name} and PlayWrightBrowserToolkit")

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
            # 使用Playwright工具包获取页面
            result = await self.toolkit.async_run_tool("navigate", {"url": url})
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
            page_data = await self.fetch_page_content(url)
            result = await self.chain.ainvoke(page_data)
            
            if not isinstance(result, list):
                result = [result]
            
            logger.info(f"成功识别 {len(result)} 个功能点")
            for func in result:
                logger.debug(f"功能点: {func.name} (优先级 {func.priority})")
            
            return sorted(result, key=lambda x: x.priority)
        except Exception as e:
            logger.error(f"功能点识别失败: {str(e)}")
            return []
EOL

cat > modules/playwright_tester.py << 'EOL'
import os
import logging
import asyncio
from datetime import datetime
from typing import Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.async_api import BrowserContext, Page, async_playwright, expect
from langchain.agents.playwright.toolkit import PlayWrightBrowserToolkit
from models.website_function import WebsiteFunction
from models.test_result import TestStepResult, FunctionTestResult

logger = logging.getLogger("WebsiteTester")

class PlaywrightTester:
    """使用Playwright和LangChain工具包执行自动化测试"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.current_function = None
        self.toolkit = PlayWrightBrowserToolkit()
        logger.info("PlaywrightTester 初始化 (集成LangChain工具包)")

    async def setup(self, user_data_dir: str = "browser_context"):
        """设置Playwright环境并启用上下文缓存"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_data_dir=user_data_dir,  # 启用浏览器上下文缓存
                record_video_dir="videos",
                ignore_https_errors=True,
            )
            self.page = await self.context.new_page()
            logger.info(f"Playwright 环境设置完成 (上下文缓存: {user_data_dir})")
        except Exception as e:
            logger.error(f"Playwright 初始化失败: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def navigate_to_page(self, url: str) -> bool:
        """导航到指定URL（带重试机制）"""
        logger.info(f"导航到: {url} (带重试机制)")
        try:
            await self.page.goto(url, timeout=45000)  # 延长超时时间
            await self.page.wait_for_selector("body", state="attached")
            return True
        except Exception as e:
            logger.warning(f"页面导航失败，尝试重试: {str(e)}")
            raise

    async def execute_step(self, step_description: str) -> TestStepResult:
        """执行单个测试步骤（使用智能等待和工具包）"""
        logger.debug(f"执行步骤: {step_description}")
        try:
            # 首先尝试使用工具包执行步骤
            try:
                # 解析步骤描述为工具调用
                tool_name, tool_args = await self.parse_step_for_tool(step_description)
                
                # 使用Playwright工具包执行操作
                await self.toolkit.async_run_tool(tool_name, tool_args)
                return TestStepResult(step_description=step_description, status="success")
            except Exception as tool_error:
                logger.debug(f"工具包执行失败，回退到原生执行: {str(tool_error)}")
                # 工具包执行失败时回退到原生执行
                return await self.execute_native_step(step_description)
                
        except Exception as e:
            error_msg = f"步骤失败: {step_description} | 错误: {str(e)}"
            logger.error(error_msg)
            return TestStepResult(
                step_description=step_description,
                status="failed",
                error=str(e),
            )

    async def parse_step_for_tool(self, step_description: str) -> tuple:
        """将自然语言步骤解析为工具包调用"""
        step = step_description.lower()
        
        if "点击" in step or "选择" in step:
            return "click", {"selector": "button, a"}
        elif "输入" in step or "填写" in step:
            return "fill", {"selector": "input, textarea", "text": "自动化测试"}
        elif "导航" in step or "访问" in step or "转到" in step:
            return "navigate", {"url": "/"}
        elif "下拉" in step:
            return "select", {"selector": "select", "value": "1"}
        elif "等待" in step or "确保" in step:
            return "wait", {"selector": "body", "state": "visible"}
        else:
            # 默认使用检查元素存在
            return "check", {"selector": "body"}

    async def execute_native_step(self, step_description: str) -> TestStepResult:
        """原生执行步骤（工具包失败时的回退方案）"""
        parsed_step = await self.parse_step(step_description)
        
        if parsed_step["action"] == "click":
            await self.page.click(parsed_step["selector"], timeout=10000)
        elif parsed_step["action"] == "fill":
            await self.page.fill(parsed_step["selector"], parsed_step["value"], timeout=10000)
        elif parsed_step["action"] == "navigate":
            await self.page.goto(parsed_step["value"], wait_until="networkidle")
        elif parsed_step["action"] == "wait":
            locator = self.page.locator(parsed_step["selector"])
            await expect(locator).to_be_visible(timeout=15000)  # 智能等待
        elif parsed_step["action"] == "select":
            await self.page.select_option(parsed_step["selector"], parsed_step["value"])
        else:
            element = await self.page.query_selector(parsed_step["selector"])
            if not element:
                raise Exception(f"元素未找到: {parsed_step['selector']}")
        
        return TestStepResult(step_description=step_description, status="success")

    async def parse_step(self, step_description: str) -> dict:
        """解析自然语言步骤为可执行操作（原生）"""
        step = step_description.lower()
        default_selector = "body"
        
        if "点击" in step or "选择" in step:
            return {"action": "click", "selector": "button, a"}
        elif "输入" in step or "填写" in step:
            return {"action": "fill", "selector": "input, textarea", "value": "自动化测试"}
        elif "导航" in step or "访问" in step or "转到" in step:
            if self.current_function and self.current_function.elements:
                for selector in self.current_function.elements:
                    if selector.startswith("a["):
                        return {"action": "click", "selector": selector}
            return {"action": "navigate", "value": "/"}
        elif "选择" in step or "下拉" in step:
            return {"action": "select", "selector": "select", "value": "1"}
        elif "等待" in step or "确保" in step:
            if self.current_function and self.current_function.elements:
                return {"action": "wait", "selector": self.current_function.elements[0]}
            return {"action": "wait", "selector": default_selector}
        else:
            return {"action": "check", "selector": default_selector}

    async def monitor_resources(self):
        """监控浏览器资源使用情况"""
        try:
            async with self.context.new_cdp_session(self.page) as session:
                await session.send("Performance.enable")
                metrics = await session.send("Performance.getMetrics")
                logger.info(f"资源监控: JS堆使用 {metrics['JSHeapUsedSize']/1024:.1f}KB, 文档数: {metrics['Documents']}")
        except Exception as e:
            logger.warning(f"资源监控失败: {str(e)}")

    async def test_function(
        self, function: WebsiteFunction, url: str
    ) -> FunctionTestResult:
        """测试单个功能点（集成资源监控）"""
        self.current_function = function
        logger.info(
            f"开始测试功能: {function.name} (优先级 {function.priority})"
        )
        result = FunctionTestResult(
            function_name=function.name,
            overall_status="passed"
        )

        if not await self.navigate_to_page(url):
            result.overall_status = "failed"
            result.error_summary = "无法导航到起始页面"
            return result

        for step in function.test_steps:
            # 执行步骤前监控资源
            await self.monitor_resources()
            
            step_result = await self.execute_step(step)
            result.step_results.append(step_result)
            
            if step_result.status == "failed":
                result.overall_status = "failed"
                result.error_summary = step_result.error
                break

        try:
            screenshot_name = f"{function.name.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.png"
            screenshot_path = os.path.join("screenshots", screenshot_name)
            await self.page.screenshot(path=screenshot_path, full_page=True)
            result.screenshot = screenshot_path
            logger.info(f"已保存截图: {screenshot_path}")
        except Exception as e:
            logger.error(f"截图保存失败: {str(e)}")
            result.screenshot = "截图失败"

        result.end_time = datetime.now().isoformat()
        status_icon = "✅" if result.overall_status == "passed" else "❌"
        logger.info(
            f"测试完成: {status_icon} {function.name} - 状态: {result.overall_status}"
        )
        return result

    async def close(self):
        """清理资源"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Playwright 资源已释放")
        except Exception as e:
            logger.error(f"资源释放失败: {str(e)}")
EOL

cat > modules/report_generator.py << 'EOL'
import json
import os
from datetime import datetime
from typing import List
from models.test_report import TestReport
from models.test_result import FunctionTestResult

class ReportGenerator:
    """生成测试报告"""
    
    @staticmethod
    def generate_report(
        function_results: List[FunctionTestResult], url: str, start_time: float
    ) -> TestReport:
        """生成测试报告结构"""
        passed = sum(1 for r in function_results if r.overall_status == "passed")
        failed = len(function_results) - passed
        exec_time = datetime.now().timestamp() - start_time

        summary_lines = [
            f"网站测试报告: {url}",
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试耗时: {exec_time:.2f}秒",
            f"测试功能点: {len(function_results)}",
            f"通过: {passed} | 失败: {failed}",
            "\n功能点详情:",
        ]

        for result in function_results:
            status_icon = "✅" if result.overall_status == "passed" else "❌"
            summary_lines.append(f" {status_icon} {result.function_name}")
            if result.error_summary:
                summary_lines.append(f" 错误: {result.error_summary}")

        summary = "\n".join(summary_lines)

        return TestReport(
            website_url=url,
            total_functions=len(function_results),
            passed=passed,
            failed=failed,
            execution_time=exec_time,
            results=function_results,
            summary=summary,
        )

    @staticmethod
    def save_report(report: TestReport, format: str = "all"):
        """保存报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"report_{timestamp}"

        if format in ["json", "all"]:
            json_path = os.path.join("reports", f"{base_filename}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report.dict(), f, ensure_ascii=False, indent=2)

        if format in ["text", "all"]:
            text_path = os.path.join("reports", f"{base_filename}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("# 网站自动化测试报告\n\n")
                f.write(f"## 摘要\n{report.summary}\n\n")
                f.write("## 详细结果\n")
                for result in report.results:
                    status_icon = "✅" if result.overall_status == "passed" else "❌"
                    f.write(f"### {status_icon} {result.function_name}\n")
                    f.write(f"- 状态: {result.overall_status}\n")
                    f.write(f"- 开始时间: {result.start_time}\n")
                    f.write(f"- 结束时间: {result.end_time}\n")
                    if result.screenshot:
                        f.write(f"- 截图: {result.screenshot}\n")
                    if result.error_summary:
                        f.write(f"- 错误摘要: {result.error_summary}\n")
                    f.write("\n### 步骤详情:\n")
                    for step in result.step_results:
                        step_icon = "✓" if step.status == "success" else "✗"
                        f.write(f"- {step_icon} {step.step_description}\n")
                        if step.error:
                            f.write(f" 错误: {step.error}\n")
                    f.write("\n")

        if format in ["html", "all"]:
            html_path = os.path.join("reports", f"{base_filename}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(
                    f"""<!DOCTYPE html>
<html>
<head>
    <title>网站测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .function {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .steps {{ margin-left: 20px; }}
        .step {{ margin-bottom: 5px; }}
        .success {{ color: #4CAF50; }}
        .failed-step {{ color: #f44336; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin-top: 10px; }}
        .metrics {{ background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 15px; }}
    </style>
</head>
<body>
    <h1>网站自动化测试报告</h1>
    <div class="summary">
        <h2>摘要</h2>
        <pre>{report.summary}</pre>
    </div>
    <h2>详细结果</h2>
"""
                )
                for result in report.results:
                    status_class = "passed" if result.overall_status == "passed" else "failed"
                    f.write(
                        f"""
    <div class="function {status_class}">
        <h3>{result.function_name} <span class="{status_class}">({result.overall_status})</span></h3>
        <p><strong>开始时间:</strong> {result.start_time}</p>
        <p><strong>结束时间:</strong> {result.end_time}</p>
"""
                    )
                    if result.error_summary:
                        f.write(
                            f'        <p><strong>错误摘要:</strong> {result.error_summary}</p>\n'
                        )
                    if result.screenshot and os.path.exists(result.screenshot):
                        f.write(
                            f'        <p><strong>截图:</strong><br><img src="{result.screenshot}"></p>\n'
                        )
                    f.write('        <div class="steps"><h4>步骤详情:</h4>\n')
                    for step in result.step_results:
                        step_class = "success" if step.status == "success" else "failed-step"
                        f.write(f'            <div class="step {step_class}">\n')
                        f.write(
                            f'                <span class="step-icon">{step.status}</span> {step.step_description}\n'
                        )
                        if step.error:
                            f.write(
                                f'                <br><span class="error">错误: {step.error}</span>\n'
                            )
                        f.write("            </div>\n")
                    f.write("        </div>\n")
                    f.write("    </div>\n")
                f.write("</body>\n</html>")
EOL

# 创建核心文件
cat > core/__init__.py << 'EOL'
# 空文件，用于包初始化
EOL

cat > core/test_runner.py << 'EOL'
import logging
import asyncio
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableParallel
from modules.function_identifier import FunctionIdentifier
from modules.playwright_tester import PlaywrightTester
from modules.report_generator import ReportGenerator
from models.test_report import TestReport

logger = logging.getLogger("WebsiteTester")

class WebsiteTestRunner:
    """网站测试运行器（集成并行处理和LangChain工具包）"""
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.identifier = FunctionIdentifier(model_name=model)
        self.tester = PlaywrightTester()
        self.start_time = 0.0
        logger.info(f"初始化 WebsiteTestRunner, 模型: {model}")

    async def run_test(self, url: str) -> TestReport:
        """执行完整测试流程（带并行处理）"""
        self.start_time = datetime.now().timestamp()
        logger.info(f"🏁 开始测试网站: {url}")

        # 识别功能点
        logger.info("🔍 正在识别网站功能点...")
        functions = await self.identifier.identify_functions(url)
        if not functions:
            logger.error("未识别到任何功能点，测试终止")
            return TestReport(
                website_url=url,
                total_functions=0,
                passed=0,
                failed=0,
                results=[],
                summary="测试失败: 未识别到任何功能点",
            )
        logger.info(f"✅ 识别到 {len(functions)} 个功能点")

        # 执行测试
        logger.info("🔧 正在设置测试环境...")
        await self.tester.setup()
        
        # 并行执行功能测试
        logger.info("🚀 开始并行执行功能测试...")
        test_results = []
        
        # 使用ThreadPoolExecutor并行执行
        with ThreadPoolExecutor(max_workers=min(4, len(functions))) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            for func in functions:
                tasks.append(loop.run_in_executor(
                    executor, 
                    lambda f=func: asyncio.run(self.tester.test_function(f, url))
                ))
            
            for result in await asyncio.gather(*tasks):
                test_results.append(result)

        # 清理资源
        logger.info("🧹 正在清理资源...")
        await self.tester.close()

        # 生成报告
        logger.info("📊 正在生成测试报告...")
        report = ReportGenerator.generate_report(
            test_results, url, self.start_time
        )
        ReportGenerator.save_report(report, format="all")
        logger.info(f"🏁 测试完成! 耗时: {report.execution_time:.2f}秒")
        return report
EOL

# 创建主文件
cat > main.py << 'EOL'
import asyncio
import logging
from config import setup_logging, create_directories
from core.test_runner import WebsiteTestRunner

# 初始化日志和目录
logger = setup_logging()
create_directories()

async def main(url: str):
    """主测试流程"""
    runner = WebsiteTestRunner()
    report = await runner.run_test(url)
    print("\n" + "=" * 50)
    print(report.summary)
    print("=" * 50)

if __name__ == "__main__":
    test_url = "https://example.com"  # 替换为实际测试网站
    asyncio.run(main(test_url))
EOL

# 创建性能监控脚本
cat > monitor_performance.py << 'EOL'
import psutil
import time
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger("PerformanceMonitor")

async def monitor_system_resources(interval=5):
    """监控系统资源使用情况"""
    logger.info("开始监控系统资源...")
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(
            f"资源使用: CPU {cpu_percent}% | "
            f"内存 {memory.percent}% ({memory.used/1024/1024:.1f}MB) | "
            f"磁盘 {disk.percent}%"
        )
        
        await asyncio.sleep(interval)

def start_monitoring():
    """启动资源监控"""
    asyncio.create_task(monitor_system_resources())
EOL

cd ..
echo "项目结构已成功创建在 website_tester 目录中"
echo "请运行以下命令安装依赖："
echo "cd website_tester && pip install langchain-core langchain-openai playwright beautifulsoup4 pydantic tenacity psutil"
echo "playwright install"
echo ""
echo "运行测试："
echo "cd website_tester && python main.py"