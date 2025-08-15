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
