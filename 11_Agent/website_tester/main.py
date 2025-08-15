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
    test_url = "https://python.langchain.com/docs/integrations/tools/playwright/"  # 替换为实际测试网站
    asyncio.run(main(test_url))
