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
