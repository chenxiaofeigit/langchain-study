# models/website_function.py
from pydantic import BaseModel, Field
from typing import List

class WebsiteFunction(BaseModel):
    name: str = Field(description="功能点名称")
    description: str = Field(description="功能描述")
    selector: str = Field(description="CSS选择器")
    test_steps: List[str] = Field(description="测试步骤列表")
    priority: int = Field(description="优先级(1-最高,3-最低)")