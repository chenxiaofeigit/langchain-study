import os
import time
from flower_rag import FlowerRAGSystem  # 假设你的类文件名为 flower_rag.py

def run_tests():
    """运行 FlowerRAGSystem 类的测试"""
    # 1. 测试初始化
    print("="*60)
    print("测试 1: 初始化 RAG 系统")
    try:
        # 使用环境变量中的 API 密钥
        rag_system = FlowerRAGSystem(
            data_dir="./OneFlower",
            use_sqlite=True,
            vector_store_path="./vector_store",
            chunk_size=300,
            chunk_overlap=20,
            search_k=3
        )
        print("✅ 初始化成功")
        print(f"加载文档数: {rag_system.get_document_count()}")
        print(f"生成文本块数: {rag_system.get_chunk_count()}")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        return
    
    # 2. 测试查询功能
    print("\n" + "="*60)
    print("测试 2: 查询功能")
    test_questions = [
        "郁金香的花语是什么？",
        "菊花适合在什么季节种植？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        start_time = time.time()
        try:
            response = rag_system.query(question)
            elapsed = time.time() - start_time
            print(f"回答: {response.content}")
            print(f"⏱️ 响应时间: {elapsed:.2f}秒")
            print("✅ 查询成功")
        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
    
    # 3. 测试文档重新加载
    print("\n" + "="*60)
    print("测试 3: 文档重新加载")
    original_count = rag_system.get_document_count()
    print(f"当前文档数: {original_count}")
    
    # 添加新文档到目录（模拟）
    print("模拟添加新文档...")
    
    try:
        rag_system.reload_documents()
        new_count = rag_system.get_document_count()
        print(f"重新加载后文档数: {new_count}")
        
        if new_count >= original_count:
            print("✅ 重新加载成功")
        else:
            print("❌ 重新加载失败: 文档数未增加")
    except Exception as e:
        print(f"❌ 重新加载失败: {str(e)}")
    
    # 4. 测试空问题处理
    print("\n" + "="*60)
    print("测试 4: 空问题处理")
    try:
        response = rag_system.query("")
        if "问题不能为空" in response.content:
            print("✅ 空问题处理成功")
        else:
            print(f"❌ 空问题处理异常: {response.content}")
    except Exception as e:
        print(f"❌ 空问题处理失败: {str(e)}")
    
    # 5. 测试无效问题处理
    print("\n" + "="*60)
    print("测试 5: 无效问题处理")
    try:
        response = rag_system.query("这是一个与鲜花无关的测试问题")
        print(f"回答: {response.content}")
        if "不知道" in response.content or "未找到" in response.content:
            print("✅ 无效问题处理成功")
        else:
            print("❌ 无效问题处理异常")
    except Exception as e:
        print(f"❌ 无效问题处理失败: {str(e)}")
    
    print("\n" + "="*60)
    print("所有测试完成")

def interactive_test():
    """交互式测试模式"""
    print("\n🌸 鲜花知识问答系统 - 交互式测试模式 🌸")
    print("输入 'exit' 退出测试")
    
    # 初始化 RAG 系统
    try:
        rag_system = FlowerRAGSystem(
            data_dir="./OneFlower",
            use_sqlite=True,
            vector_store_path="./vector_store",
            chunk_size=300,
            chunk_overlap=20,
            search_k=3
        )
        print(f"系统已初始化，加载了 {rag_system.get_document_count()} 个文档")
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return
    
    while True:
        question = input("\n请输入关于鲜花的问题: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            print("退出测试...")
            break
            
        if not question:
            print("问题不能为空，请重新输入")
            continue
            
        start_time = time.time()
        try:
            response = rag_system.query(question)
            elapsed = time.time() - start_time
            print(f"\n🤖 回答: {response.content}")
            print(f"⏱️ 响应时间: {elapsed:.2f}秒")
        except Exception as e:
            print(f"❌ 查询出错: {str(e)}")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 运行自动测试")
    print("2. 交互式测试")
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == "1":
        run_tests()
    elif choice == "2":
        interactive_test()
    else:
        print("无效选项，退出程序")