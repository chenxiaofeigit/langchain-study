import os
from dotenv import load_dotenv
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    SemanticSimilarityExampleSelector
)
from langchain.vectorstores import Chroma  # 使用Chroma代替Qdrant，更轻量
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到DEEPSEEK_API_KEY环境变量")
    
    # 设置HuggingFace缓存路径
    os.environ["HF_HOME"] = "/home/fiona/data/hf_cache"
    print(f"HuggingFace缓存路径设置为: {os.environ['HF_HOME']}")

    # 1. 创建示例数据
    samples = [
        {
            "flower_type": "玫瑰",
            "occasion": "爱情",
            "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
        },
        {
            "flower_type": "康乃馨",
            "occasion": "母亲节",
            "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
        },
        {
            "flower_type": "百合",
            "occasion": "庆祝",
            "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
        },
        {
            "flower_type": "向日葵",
            "occasion": "鼓励",
            "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
        }
    ]
    
    # 2. 创建示例提示模板
    prompt_sample = PromptTemplate(
        input_variables=["flower_type", "occasion", "ad_copy"],
        template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
    )
    
    print("=" * 60)
    print("单个示例提示:")
    print(prompt_sample.format(**samples[0]))
    print("=" * 60)
    
    # 3. 创建FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=samples,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}\n文案:",
        input_variables=["flower_type", "occasion"]
    )
    
    # 测试FewShotPromptTemplate
    test_input = {"flower_type": "野玫瑰", "occasion": "爱情"}
    formatted_prompt = few_shot_prompt.format(**test_input)
    print("FewShot提示内容:")
    print(formatted_prompt)
    print("=" * 60)
    
    # 4. 调用模型生成文案
    # 使用新版langchain_openai
    # 初始化DeepSeek模型
    model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    result = model.invoke(formatted_prompt)
    print("模型生成的文案:")
    print(result)
    print("=" * 60)
    
    # 5. 使用示例选择器
    # 初始化嵌入模型,使用本地嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  # 推荐的中文模型
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 使用Chroma作为向量存储（轻量级，无需外部依赖）
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=samples,
        embeddings=embeddings,
        vectorstore_cls=Chroma,
        k=1
    )
    
    # 创建带示例选择器的FewShotPromptTemplate
    selector_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_sample,
        suffix="鲜花类型: {flower_type}\n场合: {occasion}\n文案:",
        input_variables=["flower_type", "occasion"]
    )
    
    # 测试示例选择器
    test_input = {"flower_type": "红玫瑰", "occasion": "爱情"}
    selector_formatted = selector_prompt.format(**test_input)
    print("带示例选择器的提示内容:")
    print(selector_formatted)
    print("=" * 60)
    
    # 使用示例选择器生成文案
    selector_result = model.invoke(selector_formatted)
    print("使用示例选择器生成的文案:")
    print(selector_result)
    print("=" * 60)

if __name__ == "__main__":
    main()