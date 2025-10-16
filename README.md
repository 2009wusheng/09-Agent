# 09 Agent

一个基于 Streamlit + LangChain + FAISS 的多模态数据智能 Agent，支持：
- PDF 智能问答（本地向量检索）
- CSV 数据分析（工具调用 + 图表输出）
- 多轮对话记忆：
  - 短期记忆：按窗口（轮数）注入 `chat_history`
  - 长期记忆：基于 FAISS 的持久化记忆库（可开关、可清空）

## 目录结构
```
.
├─ Data_Agent.py            # 主应用入口
├─ requirements.txt         # 依赖列表
├─ faiss_db/                # PDF 文档向量库（运行时生成）
└─ mem_db/                  # 长期记忆库（运行时生成）
```

## 运行环境
- Python 3.12（推荐）
- Windows / macOS / Linux

## 环境变量
在项目根目录创建 `.env`：
```
DEEPSEEK_API_KEY=your_deepseek_key
DASHSCOPE_API_KEY=your_dashscope_key
# 或兼容你当前代码的命名：
# DEEPSEEK_API_KEY=...
# dashscope_api_key=...
```

## 安装
建议使用虚拟环境：
```bash
py -3.12 -m venv venv
./venv/Scripts/activate    # Windows PowerShell
pip install -r requirements.txt
```
若已存在 `venv`：
```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 启动
```bash
./venv/Scripts/streamlit run Data_Agent.py
```
启动后：
- 在“📄 PDF智能问答”上传并处理 PDF，完成后可在左侧聊天区提问；
- 在“📊 CSV数据分析”上传 CSV，可进行自然语言数据分析和绘图（图片保存为 `plot.png`）。

## 对话记忆
- 右侧面板提供：
  - 对话记忆窗口(轮数)：控制短期记忆注入窗口
  - 启用长期记忆：是否从记忆库检索，并在对话后写入新记忆
  - 长期记忆检索条目数：控制检索注入到提示词的记忆数量
  - 清除长期记忆库：一键清空 `mem_db/`

## 常见问题
- 首次运行缺少 `faiss_db/` 或 `mem_db/`：与运行时自动创建，或在 UI 中通过按钮清理/重建。
- IDE 提示导入告警（Import could not be resolved）：多为环境索引问题，不影响实际运行。

## 许可证
本项目示例默认 MIT，可按需修改。
