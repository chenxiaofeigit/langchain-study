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
