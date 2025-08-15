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
