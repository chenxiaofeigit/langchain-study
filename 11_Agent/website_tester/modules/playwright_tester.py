import os
import logging
import asyncio
from datetime import datetime
from typing import Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from models.website_function import WebsiteFunction
from models.test_result import TestStepResult, FunctionTestResult
from playwright.async_api import async_playwright, Browser, Page, BrowserContext, expect

# 修改为异步创建浏览器工具包
async def create_playwright_toolkit():
    # 在同一个事件循环中启动playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    return PlayWrightBrowserToolkit(async_browser=browser)

logger = logging.getLogger("WebsiteTester")

class PlaywrightTester:
    """使用Playwright和LangChain工具包执行自动化测试"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.current_function = None
        self.toolkit = None
        logger.info("PlaywrightTester 初始化 (集成LangChain工具包)")

    async def setup_toolkit(self):
        """异步设置Playwright工具包"""
        if self.toolkit is None:
            self.toolkit = await create_playwright_toolkit()
            logger.info("Playwright工具包已初始化")

    async def setup(self, user_data_subdir: str = ""):
        """设置Playwright环境并启用上下文缓存（修复storage_state首次创建问题）"""
        try:
            # 1. 构建正确的目录路径（避免重复层级）
            base_context_dir = os.path.join(os.getcwd(), "browser_context")
            os.makedirs(base_context_dir, exist_ok=True)  # 确保基础目录存在

            # 处理子目录（如果需要）
            if user_data_subdir:
                self.user_data_dir = os.path.join(base_context_dir, user_data_subdir)
                os.makedirs(self.user_data_dir, exist_ok=True)  # 确保子目录存在
            else:
                self.user_data_dir = base_context_dir

            # 2. 构建state.json路径（后续用于保存，而非首次读取）
            self.state_file_path = os.path.join(self.user_data_dir, "state.json")

            # 3. 首次创建上下文：不指定storage_state（避免读取不存在的文件）
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,  # 只保留必要参数，移除storage_state
            )

            # 4. 关键步骤：首次创建上下文后，立即保存初始状态到文件
            # （这会创建state.json，后续使用时可加载）
            await self.context.storage_state(path=self.state_file_path)

            # 5. 创建页面
            self.page = await self.context.new_page()
            logger.info(f"Playwright 环境设置完成（上下文缓存: {self.user_data_dir}）")
            logger.info(f"首次创建状态文件: {self.state_file_path}")

        except Exception as e:
            logger.error(f"Playwright 初始化失败: {str(e)}", exc_info=True)
            raise

    # （可选）添加关闭时保存最新状态的逻辑（确保状态更新）
    async def close(self):
        """关闭Playwright资源，同时保存最新的上下文状态"""
        if hasattr(self, "context") and self.context:
            # 关闭前保存最新状态（覆盖旧文件）
            await self.context.storage_state(path=self.state_file_path)
            await self.context.close()
            logger.info(f"已保存最新上下文状态到: {self.state_file_path}")
        
        if hasattr(self, "browser") and self.browser:
            await self.browser.close()
        
        if hasattr(self, "playwright") and self.playwright:
            await self.playwright.stop()
        
        logger.info("Playwright 资源已释放")


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
                # 解析工具名称和参数
                tool_name, tool_args = await self.parse_step_for_tool(step_description)

                # 获取工具包中的所有工具
                tools = self.toolkit.get_tools()

                # 根据工具名称找到对应的工具
                tool = next((t for t in tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"未找到名为 {tool_name} 的工具")

                # 异步执行工具（使用 arun 方法）
                await tool.arun(**tool_args)

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


