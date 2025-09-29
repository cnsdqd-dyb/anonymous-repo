import numpy as np
import json
import re
import os
from typing import List, Dict, Tuple
import random
from flask import Flask, render_template, request, jsonify, send_file
import threading
import webbrowser
from datetime import datetime

app = Flask(__name__)

class ModelComparator:
    def __init__(self):
        """初始化模型比较器"""
        self.model_size = '1.5b'
        self.benchmark = 'olympiadbench'
        self.model1_data = []
        self.model2_data = []
        self.current_index = 0
        self.selected_results = {'model1': [], 'model2': []}
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        modelMap = {
            '1.5b': {
                'maxflow': 'Structured-R1-Norm',
                'grpo': 'UNS_GRPO_1_5B_4k'
            },
            '7b': {
                'maxflow': 'MAX_FLOW_7B_4k',
                'grpo': 'UNS_GRPO_7B_4k'
            }
        }
        
        model1_file = f"D:/我的文件/研究论文相关/llm_analyse/web/web_test/web_test/static/json_data/{self.benchmark}_results_{modelMap[self.model_size]['maxflow']}_step_attention.json"
        model2_file = f"D:/我的文件/研究论文相关/llm_analyse/web/web_test/web_test/static/json_data/{self.benchmark}_results_{modelMap[self.model_size]['grpo']}.json"
        
        try:
            with open(model1_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.model1_data = data["structured"]["details"]
            
            with open(model2_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.model2_data = data["unstructured"]["details"]
                
            print(f"✅ 成功加载 Model1: {len(self.model1_data)} 条数据")
            print(f"✅ 成功加载 Model2: {len(self.model2_data)} 条数据")
            
            # 重置选择结果
            self.selected_results = {'model1': [], 'model2': []}
            self.current_index = 0
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到: {e.filename}")
            return False
        except json.JSONDecodeError:
            print(f"❌ JSON文件格式错误")
            return False
        return True
    
    def find_matching_problems(self):
        """找到两个模型中问题文本完全一致的题目对"""
        matched_pairs = []
        
        for i, item1 in enumerate(self.model1_data):
            for j, item2 in enumerate(self.model2_data):
                if item1['problem'] == item2['problem']:
                    matched_pairs.append({
                        'model1_index': i,
                        'model2_index': j,
                        'model1_data': item1,
                        'model2_data': item2
                    })
                    break
        
        return matched_pairs
    
    def get_current_pair(self):
        """获取当前题目对"""
        matched_pairs = self.find_matching_problems()
        if self.current_index < len(matched_pairs):
            return matched_pairs[self.current_index]
        return None
    
    def get_total_pairs(self):
        """获取总题目对数"""
        return len(self.find_matching_problems())
    
    def save_selected_results(self):
        """保存选择的结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建保存的数据结构
        model1_result = {
            "structured": {
                "details": self.selected_results['model1']
            }
        }
        
        model2_result = {
            "unstructured": {
                "details": self.selected_results['model2']
            }
        }
        
        # 保存文件
        model1_filename = f"selected_{self.benchmark}_{self.model_size}_model1_{timestamp}.json"
        model2_filename = f"selected_{self.benchmark}_{self.model_size}_model2_{timestamp}.json"
        
        with open(model1_filename, 'w', encoding='utf-8') as f:
            json.dump(model1_result, f, ensure_ascii=False, indent=2)
        
        with open(model2_filename, 'w', encoding='utf-8') as f:
            json.dump(model2_result, f, ensure_ascii=False, indent=2)
        
        return model1_filename, model2_filename

# 全局比较器实例
comparator = ModelComparator()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """加载数据API"""
    data = request.json
    comparator.benchmark = data.get('benchmark', 'olympiadbench')
    comparator.model_size = data.get('model_size', '1.5b')
    
    success = comparator.load_data()
    if success:
        total_pairs = comparator.get_total_pairs()
        return jsonify({
            'success': True, 
            'total_pairs': total_pairs,
            'message': f'成功加载 {total_pairs} 对匹配题目'
        })
    else:
        return jsonify({'success': False, 'message': '数据加载失败'})

@app.route('/api/get_current_pair')
def get_current_pair():
    """获取当前题目对API"""
    pair = comparator.get_current_pair()
    if pair:
        return jsonify({
            'success': True,
            'current_index': comparator.current_index,
            'total_pairs': comparator.get_total_pairs(),
            'model1': pair['model1_data'],
            'model2': pair['model2_data'],
            'selected_count': {
                'model1': len(comparator.selected_results['model1']),
                'model2': len(comparator.selected_results['model2'])
            }
        })
    else:
        return jsonify({'success': False, 'message': '没有更多题目对'})

@app.route('/api/navigate', methods=['POST'])
def navigate():
    """导航API"""
    data = request.json
    direction = data.get('direction')
    
    total_pairs = comparator.get_total_pairs()
    
    if direction == 'next' and comparator.current_index < total_pairs - 1:
        comparator.current_index += 1
    elif direction == 'prev' and comparator.current_index > 0:
        comparator.current_index -= 1
    elif direction == 'first':
        comparator.current_index = 0
    elif direction == 'last':
        comparator.current_index = total_pairs - 1
    
    return get_current_pair()

@app.route('/api/select_result', methods=['POST'])
def select_result():
    """选择结果API"""
    data = request.json
    model = data.get('model')  # 'model1' 或 'model2'
    
    pair = comparator.get_current_pair()
    if not pair:
        return jsonify({'success': False, 'message': '没有可选择的题目对'})
    
    if model == 'model1':
        # 检查是否已经选择过这个题目
        existing = [item for item in comparator.selected_results['model1'] 
                   if item['problem'] == pair['model1_data']['problem']]
        if not existing:
            comparator.selected_results['model1'].append(pair['model1_data'])
    elif model == 'model2':
        existing = [item for item in comparator.selected_results['model2'] 
                   if item['problem'] == pair['model2_data']['problem']]
        if not existing:
            comparator.selected_results['model2'].append(pair['model2_data'])
    
    return jsonify({
        'success': True,
        'selected_count': {
            'model1': len(comparator.selected_results['model1']),
            'model2': len(comparator.selected_results['model2'])
        }
    })

@app.route('/api/save_results', methods=['POST'])
def save_results():
    """保存结果API"""
    try:
        model1_file, model2_file = comparator.save_selected_results()
        return jsonify({
            'success': True,
            'message': f'结果已保存到 {model1_file} 和 {model2_file}',
            'files': [model1_file, model2_file]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})

@app.route('/api/get_stats')
def get_stats():
    """获取统计信息API"""
    return jsonify({
        'benchmark': comparator.benchmark,
        'model_size': comparator.model_size,
        'total_pairs': comparator.get_total_pairs(),
        'current_index': comparator.current_index,
        'selected_count': {
            'model1': len(comparator.selected_results['model1']),
            'model2': len(comparator.selected_results['model2'])
        }
    })

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型推理比较工具</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        select, button {
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        
        button {
            background: #007bff;
            color: white;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #0056b3;
        }
        
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .stats {
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
        }
        
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .model-panel {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .model-header {
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
        }
        
        .model1-header {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .model2-header {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .model-content {
            padding: 20px;
        }
        
        .section {
            margin-bottom: 20px;
        }
        
        .section-title {
            font-weight: bold;
            color: #555;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #eee;
        }
        
        .section-content {
            line-height: 1.6;
            max-height: 200px;
            overflow-y: auto;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        
        .select-button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            margin-top: 15px;
        }
        
        .model1-select {
            background: #2196f3;
        }
        
        .model2-select {
            background: #9c27b0;
        }
        
        .navigation {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .progress {
            margin: 15px 0;
            font-size: 16px;
            font-weight: bold;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 18px;
            color: #666;
        }
        
        .message {
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        @media (max-width: 768px) {
            .comparison {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
            
            .nav-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 模型推理比较工具</h1>
            <div class="controls">
                <div class="control-group">
                    <label>基准测试:</label>
                    <select id="benchmark">
                        <option value="olympiadbench">OlympiadBench</option>
                        <option value="lsat">LSAT</option>
                        <option value="math500">Math500</option>
                        <option value="drop">DROP</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>模型大小:</label>
                    <select id="modelSize">
                        <option value="1.5b">1.5B</option>
                        <option value="7b">7B</option>
                    </select>
                </div>
                
                <button onclick="loadData()">🔄 加载数据</button>
                <button onclick="saveResults()" id="saveBtn" disabled>💾 保存选择结果</button>
            </div>
            
            <div class="stats" id="stats">
                等待加载数据...
            </div>
        </div>
        
        <div id="content">
            <div class="loading">
                请先选择基准测试和模型大小，然后点击"加载数据"
            </div>
        </div>
    </div>

    <script>
        let currentData = null;
        
        async function loadData() {
            const benchmark = document.getElementById('benchmark').value;
            const modelSize = document.getElementById('modelSize').value;
            
            showLoading('正在加载数据...');
            
            try {
                const response = await fetch('/api/load_data', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({benchmark, model_size: modelSize})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showMessage(result.message, 'success');
                    await loadCurrentPair();
                    document.getElementById('saveBtn').disabled = false;
                } else {
                    showMessage(result.message, 'error');
                }
            } catch (error) {
                showMessage('加载数据失败: ' + error.message, 'error');
            }
        }
        
        async function loadCurrentPair() {
            try {
                const response = await fetch('/api/get_current_pair');
                const result = await response.json();
                
                if (result.success) {
                    currentData = result;
                    displayComparison(result);
                    updateStats();
                } else {
                    showMessage(result.message, 'error');
                }
            } catch (error) {
                showMessage('获取数据失败: ' + error.message, 'error');
            }
        }
        
        function displayComparison(data) {
            const content = document.getElementById('content');
            content.innerHTML = `
                <div class="comparison">
                    <div class="model-panel">
                        <div class="model-header model1-header">
                            📊 模型1 (结构化推理) - ID: ${data.model1.problem_id}
                        </div>
                        <div class="model-content">
                            <div class="section">
                                <div class="section-title">❓ 问题</div>
                                <div class="section-content">${data.model1.problem}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">✅ 预测答案</div>
                                <div class="section-content">${data.model1.prediction}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">🧠 推理过程</div>
                                <div class="section-content">${escapeHtml(data.model1.reasoning)}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">📊 令牌使用</div>
                                <div class="section-content">${data.model1.tokens}</div>
                            </div>
                            <button class="select-button model1-select" onclick="selectResult('model1')">
                                选择模型1的结果
                            </button>
                        </div>
                    </div>
                    
                    <div class="model-panel">
                        <div class="model-header model2-header">
                            📈 模型2 (非结构化推理) - ID: ${data.model2.problem_id}
                        </div>
                        <div class="model-content">
                            <div class="section">
                                <div class="section-title">❓ 问题</div>
                                <div class="section-content">${data.model2.problem}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">✅ 预测答案</div>
                                <div class="section-content">${data.model2.prediction}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">🧠 推理过程</div>
                                <div class="section-content">${data.model2.reasoning}</div>
                            </div>
                            <div class="section">
                                <div class="section-title">📊 令牌使用</div>
                                <div class="section-content">${data.model2.tokens}</div>
                            </div>
                            <button class="select-button model2-select" onclick="selectResult('model2')">
                                选择模型2的结果
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="progress">
                        题目进度: ${data.current_index + 1} / ${data.total_pairs}
                    </div>
                    <div class="nav-buttons">
                        <button onclick="navigate('first')" ${data.current_index === 0 ? 'disabled' : ''}>
                            ⏮️ 第一题
                        </button>
                        <button onclick="navigate('prev')" ${data.current_index === 0 ? 'disabled' : ''}>
                            ⬅️ 上一题
                        </button>
                        <button onclick="navigate('next')" ${data.current_index >= data.total_pairs - 1 ? 'disabled' : ''}>
                            ➡️ 下一题
                        </button>
                        <button onclick="navigate('last')" ${data.current_index >= data.total_pairs - 1 ? 'disabled' : ''}>
                            ⏭️ 最后一题
                        </button>
                    </div>
                </div>
            `;
        }
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function navigate(direction) {
            try {
                const response = await fetch('/api/navigate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({direction})
                });
                
                const result = await response.json();
                if (result.success) {
                    currentData = result;
                    displayComparison(result);
                    updateStats();
                }
            } catch (error) {
                showMessage('导航失败: ' + error.message, 'error');
            }
        }
        
        async function selectResult(model) {
            try {
                const response = await fetch('/api/select_result', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model})
                });
                
                const result = await response.json();
                if (result.success) {
                    showMessage(`已选择${model === 'model1' ? '模型1' : '模型2'}的结果`, 'success');
                    updateStats();
                } else {
                    showMessage(result.message, 'error');
                }
            } catch (error) {
                showMessage('选择失败: ' + error.message, 'error');
            }
        }
        
        async function saveResults() {
            try {
                const response = await fetch('/api/save_results', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                const result = await response.json();
                if (result.success) {
                    showMessage(result.message, 'success');
                } else {
                    showMessage(result.message, 'error');
                }
            } catch (error) {
                showMessage('保存失败: ' + error.message, 'error');
            }
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/api/get_stats');
                const stats = await response.json();
                
                document.getElementById('stats').innerHTML = `
                    📊 当前状态: ${stats.benchmark} | ${stats.model_size} | 
                    进度: ${stats.current_index + 1}/${stats.total_pairs} | 
                    已选择: 模型1 (${stats.selected_count.model1}) | 模型2 (${stats.selected_count.model2})
                `;
            } catch (error) {
                console.error('更新统计信息失败:', error);
            }
        }
        
        function showLoading(message) {
            document.getElementById('content').innerHTML = `
                <div class="loading">${message}</div>
            `;
        }
        
        function showMessage(message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.textContent = message;
            
            const container = document.querySelector('.container');
            container.insertBefore(messageDiv, container.firstChild);
            
            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }
        
        // 键盘快捷键
        document.addEventListener('keydown', function(e) {
            if (!currentData) return;
            
            switch(e.key) {
                case 'ArrowLeft':
                    if (currentData.current_index > 0) navigate('prev');
                    break;
                case 'ArrowRight':
                    if (currentData.current_index < currentData.total_pairs - 1) navigate('next');
                    break;
                case '1':
                    selectResult('model1');
                    break;
                case '2':
                    selectResult('model2');
                    break;
            }
        });
    </script>
</body>
</html>
'''

# 创建模板目录和文件
def create_template():
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
PORT = 5001
def open_browser():
    """延迟打开浏览器"""
    import time
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    """主函数"""
    create_template()
    print("🚀 启动模型比较工具...")
    print(f"📱 Web界面将在 http://localhost:{PORT} 打开")
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动Flask应用
    app.run(debug=False, host='0.0.0.0', port=PORT)

if __name__ == "__main__":
    main()
