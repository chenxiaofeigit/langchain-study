from langchain.prompts import PromptTemplate

def create_company_names():
    """
    使用LangChain PromptTemplate生成电商公司名称建议
    
    该函数演示两种不同的提示模板使用方法：
    1. 简单模板 - 只基于产品类型生成名称
    2. 详细模板 - 基于产品类型和市场定位生成名称
    """
    # 示例1: 简单产品名称提示模板
    simple_template = """\
    你是业务咨询顾问。
    请为销售{product}的电商公司起三个好的名字？"""
    
    simple_prompt = PromptTemplate.from_template(simple_template)
    simple_query = simple_prompt.format(product="鲜花")
    
    print("=" * 60)
    print("【简单提示模板示例】")
    print(simple_query)
    print("=" * 60)
    
    # 示例2: 带市场定位的详细提示模板
    detailed_template = """\
    你是资深业务咨询顾问。
    请为面向{market}市场、专注于销售{product}的电商公司，
    起三个有吸引力的名字，并简要说明每个名字的寓意？"""
    
    detailed_prompt = PromptTemplate(
        input_variables=["product", "market"],
        template=detailed_template
    )
    detailed_query = detailed_prompt.format(product="高端鲜花", market="奢侈品消费者")
    
    print("【详细提示模板示例】")
    print(detailed_query)
    print("=" * 60)
    
    # 返回生成的提示内容
    return {
        "simple_prompt": simple_query,
        "detailed_prompt": detailed_query
    }

if __name__ == "__main__":
    # 执行提示生成函数
    prompts = create_company_names()
    
    print("\n提示内容已生成，可直接用于语言模型调用")
    print(f"简单提示长度: {len(prompts['simple_prompt'])} 字符")
    print(f"详细提示长度: {len(prompts['detailed_prompt'])} 字符")