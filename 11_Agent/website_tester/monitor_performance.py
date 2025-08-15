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
