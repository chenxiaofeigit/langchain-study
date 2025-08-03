from flask import Flask, request, jsonify, render_template
from flower_rag import FlowerRAGSystem
import threading

app = Flask(__name__)

# 使用单例模式管理RAG系统
rag_system = None
rag_lock = threading.Lock()

def get_rag_system():
    global rag_system
    if rag_system is None:
        with rag_lock:
            if rag_system is None:
                rag_system = FlowerRAGSystem()
    return rag_system

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        rag = get_rag_system()
        response = rag.query(question)
        
        # 检查响应类型
        if 'error' in response:
            return jsonify({
                "error": response["error"],
                "question": question
            }), 500
        else:
            return jsonify({
                "question": question,
                "answer": response["answer"],
                "sources": response["sources"],
                "response_time": response["response_time"]
            })
    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "question": question
        }), 500

@app.route('/update_knowledge', methods=['POST'])
def update_knowledge():
    try:
        rag = get_rag_system()
        rag.build_vector_db()  # 重新构建知识库
        return jsonify({"status": "success", "message": "知识库已更新"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)