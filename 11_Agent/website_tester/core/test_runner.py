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
    
    def __init__(self):
        self.identifier = FunctionIdentifier()
        self.tester = PlaywrightTester()
        self.start_time = 0.0
        logger.info(f"初始化 WebsiteTestRunner")

    async def run_test(self, url: str) -> TestReport:
        self.identifier.toolkit = self.tester.toolkit
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
