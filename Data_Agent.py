import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from langchain_experimental.tools import PythonAstREPLTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
import matplotlib

matplotlib.use('Agg')
import os
from dotenv import load_dotenv

load_dotenv(override=True)

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 记忆库目录
MEMORY_DB_DIR = "mem_db"


# 自定义流式输出回调处理器
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response, **kwargs):
        self.container.markdown(self.text)


# 页面配置
st.set_page_config(
    page_title="09 Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主题色彩 */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --error-color: #d62728;
        --background-color: #f8f9fa;
    }

    /* 隐藏默认的Streamlit样式 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 标题样式 */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 卡片样式 */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }

    .success-card {
        background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
        border-left: 4px solid var(--success-color);
    }

    .warning-card {
        background: linear-gradient(135deg, #fff8e1, #fffbf0);
        border-left: 4px solid var(--warning-color);
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(45deg, #1f77b4, #2196F3);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }

    /* Tab样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: white;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1f77b4, #2196F3);
        color: white !important;
        border: 2px solid #1f77b4;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa, #ffffff);
    }

    /* 文件上传区域 */
    .uploadedFile {
        background: #f8f9fa;
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* 状态指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .status-ready {
        background: #e8f5e8;
        color: #2ca02c;
        border: 1px solid #2ca02c;
    }

    .status-waiting {
        background: #fff8e1;
        color: #ff9800;
        border: 1px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


# 初始化embeddings
@st.cache_resource
def init_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=dashscope_api_key
    )


# 初始化LLM
@st.cache_resource
def init_llm():
    return init_chat_model("deepseek-chat", model_provider="deepseek", api_key=DeepSeek_API_KEY)


# 初始化会话状态
def init_session_state():
    if 'pdf_messages' not in st.session_state:
        st.session_state.pdf_messages = []
    if 'csv_messages' not in st.session_state:
        st.session_state.csv_messages = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    # 对话记忆窗口（按轮数：user+assistant 为一轮）
    if 'pdf_history_pairs' not in st.session_state:
        st.session_state.pdf_history_pairs = 6
    if 'csv_history_pairs' not in st.session_state:
        st.session_state.csv_history_pairs = 6
    # 长期记忆开关与检索配置
    if 'enable_long_memory_pdf' not in st.session_state:
        st.session_state.enable_long_memory_pdf = True
    if 'enable_long_memory_csv' not in st.session_state:
        st.session_state.enable_long_memory_csv = True
    if 'memory_k_pdf' not in st.session_state:
        st.session_state.memory_k_pdf = 5
    if 'memory_k_csv' not in st.session_state:
        st.session_state.memory_k_csv = 5


def build_chat_history(messages: List[dict], max_pairs: int) -> List:
    """将会话中的历史文本消息转换为 LangChain 消息，并按窗口限制。

    - 仅纳入 type == 'text' 的消息
    - 排除当前最新一条用户输入（该条由 input 提供）
    - 按最近 max_pairs 轮（约 2*max_pairs 条消息）截断
    """
    if max_pairs <= 0:
        return []

    text_messages = [m for m in messages if m.get("type") == "text"]
    # 排除当前用户最新输入（调用方已先 append user，再调用响应函数）
    if text_messages and text_messages[-1].get("role") == "user":
        text_messages = text_messages[:-1]

    selected = text_messages[-2 * max_pairs:]
    lc_messages = []
    for m in selected:
        content = m.get("content", "")
        if m.get("role") == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def memory_db_exists():
    return os.path.exists(MEMORY_DB_DIR) and os.path.exists(os.path.join(MEMORY_DB_DIR, "index.faiss"))


def retrieve_memory(query: str, k: int) -> str:
    """从长期记忆向量库检索与当前查询相关的记忆片段，拼接为上下文文本。"""
    if k <= 0 or not memory_db_exists():
        return ""
    embeddings = init_embeddings()
    store = FAISS.load_local(MEMORY_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return ""
    joined = "\n".join([d.page_content for d in docs])
    return joined[:2000]


def add_memories(texts: List[str]):
    """将若干记忆片段写入向量库。每个片段作为独立文档。"""
    clean_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        return
    embeddings = init_embeddings()
    if memory_db_exists():
        store = FAISS.load_local(MEMORY_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        store.add_texts(clean_texts)
        store.save_local(MEMORY_DB_DIR)
    else:
        store = FAISS.from_texts(clean_texts, embedding=embeddings)
        store.save_local(MEMORY_DB_DIR)


def extract_memory_snippets(llm, user_input: str, assistant_output: str) -> List[str]:
    """调用 LLM 从一轮对话中提炼出简洁、稳定的长期记忆片段（中文，1-3 条）。
    若无稳定信息则返回空列表。"""
    prompt = (
        "你是对话记忆提取器。基于以下用户输入与助手回复，提取可长期复用的简洁事实或偏好，不要包含隐私。\n"
        "- 以中文短句输出，每条一行，不要编号。\n"
        "- 仅在有明确、稳定事实时输出，最多3条；若无则输出空字符串。\n\n"
        f"[用户]\n{user_input}\n\n[助手]\n{assistant_output}\n\n[输出]"
    )
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        text = getattr(res, "content", "") or str(res)
        lines = [ln.strip("•- \t") for ln in text.splitlines()]
        return [ln for ln in lines if ln]
    except Exception:
        return []


# PDF处理函数
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    embeddings = init_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def check_database_exists():
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")


def get_pdf_response(user_question, message_placeholder):
    if not check_database_exists():
        return "❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！"

    try:
        embeddings = init_embeddings()
        llm = init_llm()

        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()

        memory_context = ""
        if st.session_state.enable_long_memory_pdf:
            memory_context = retrieve_memory(user_question, st.session_state.memory_k_pdf)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """你是09 Agent，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说"答案不在上下文中"，不要提供错误的答案"""),
            ("system", "以下为你的长期记忆（可能为空）：\n{memory_context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor",
                                                "This tool is to give answer to queries from the pdf")

        # 创建流式处理器
        stream_handler = StreamHandler(message_placeholder)

        agent = create_tool_calling_agent(llm, [retrieval_chain], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[retrieval_chain], verbose=True)

        chat_history = build_chat_history(st.session_state.pdf_messages, st.session_state.pdf_history_pairs)
        response = agent_executor.invoke(
            {"input": user_question, "chat_history": chat_history, "memory_context": memory_context},
            config={"callbacks": [stream_handler]}
        )
        output_text = response['output']

        # 对话后提取长期记忆并写入
        if st.session_state.enable_long_memory_pdf:
            snippets = extract_memory_snippets(llm, user_question, output_text)
            add_memories(snippets)

        return output_text

    except Exception as e:
        return f"❌ 处理问题时出错: {str(e)}"


# CSV处理函数
def get_csv_response(query: str, message_placeholder):
    if st.session_state.df is None:
        return "请先上传CSV文件"

    llm = init_llm()
    locals_dict = {'df': st.session_state.df}
    tools = [PythonAstREPLTool(locals=locals_dict)]

    system = f"""Given a pandas dataframe `df` answer user's query.
    Here's the output of `df.head().to_markdown()` for your reference, you have access to full dataframe as `df`:
    ```
    {st.session_state.df.head().to_markdown()}
    ```
    Give final answer as soon as you have enough data, otherwise generate code using `df` and call required tool.
    If user asks you to make a graph, save it as `plot.png`, and output GRAPH:<graph title>.
    Example:
    ```
    plt.hist(df['Age'])
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Histogram')
    plt.savefig('plot.png')
    ``` output: GRAPH:Age histogram
    Query:"""

    memory_context = ""
    if st.session_state.enable_long_memory_csv:
        memory_context = retrieve_memory(query, st.session_state.memory_k_csv)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("system", "以下为你的长期记忆（可能为空）：\n{memory_context}"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 创建流式处理器
    stream_handler = StreamHandler(message_placeholder)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    chat_history = build_chat_history(st.session_state.csv_messages, st.session_state.csv_history_pairs)
    response = agent_executor.invoke(
        {"input": query, "chat_history": chat_history, "memory_context": memory_context},
        config={"callbacks": [stream_handler]}
    )
    output_text = response['output']

    if st.session_state.enable_long_memory_csv:
        snippets = extract_memory_snippets(llm, query, output_text)
        add_memories(snippets)

    return output_text


def main():
    init_session_state()

    # 主标题
    st.markdown('<h1 class="main-header">🤖 09 Agent </h1>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; margin-bottom: 2rem; color: #666;">PDF问答 & 数据分析</div>',
        unsafe_allow_html=True)

    # 创建两个主要功能的标签页
    tab1, tab2 = st.tabs(["📄 PDF智能问答", "📊 CSV数据分析"])

    # PDF问答模块
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 💬 与PDF文档对话")

            # 显示数据库状态
            if check_database_exists():
                st.markdown(
                    '<div class="info-card success-card"><span class="status-indicator status-ready">✅ PDF数据库已准备就绪</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请先上传并处理PDF文件</span></div>',
                    unsafe_allow_html=True)

            chat_container = st.container()

            with chat_container:
                # 聊天界面
                for message in st.session_state.pdf_messages:
                    with st.chat_message(message["role"]):
                        if message["type"] == "text":
                            st.markdown(message["content"])
                        elif message["type"] == "streaming":
                            # 对于流式消息，直接显示最终内容
                            st.markdown(message["content"])

            # 用户输入
            if pdf_query := st.chat_input("💭 向PDF提问...", disabled=not check_database_exists(), key="pdf_input"):
                st.session_state.pdf_messages.append({"role": "user", "content": pdf_query, "type": "text"})
                with st.chat_message("user"):
                    st.markdown(pdf_query)

                with st.chat_message("assistant"):
                    # 创建消息占位符用于流式输出
                    message_placeholder = st.empty()

                    with st.spinner("🤔 AI正在分析文档..."):
                        response = get_pdf_response(pdf_query, message_placeholder)

                    # 确保最终内容正确显示
                    message_placeholder.markdown(response)
                    st.session_state.pdf_messages.append({"role": "assistant", "content": response, "type": "text"})
                st.rerun()
        with col2:
            st.markdown("### 📁 文档管理")
            st.slider("对话记忆窗口(轮数)", min_value=0, max_value=20, value=st.session_state.pdf_history_pairs,
                      key="pdf_history_pairs")
            st.checkbox("启用长期记忆", value=st.session_state.enable_long_memory_pdf, key="enable_long_memory_pdf")
            st.slider("长期记忆检索条目数", min_value=0, max_value=20, value=st.session_state.memory_k_pdf, key="memory_k_pdf")

            # 文件上传
            pdf_docs = st.file_uploader(
                "📎 上传PDF文件",
                accept_multiple_files=True,
                type=['pdf'],
                help="支持上传多个PDF文件",
                key="pdf_uploader"
            )

            if pdf_docs:
                st.success(f"📄 已选择 {len(pdf_docs)} 个文件")
                for i, pdf in enumerate(pdf_docs, 1):
                    st.write(f"• {pdf.name}")

            # 处理按钮
            if st.button("🚀 上传并处理PDF文档", disabled=not pdf_docs, use_container_width=True, key="process_pdf"):
                with st.spinner("📊 正在处理PDF文件..."):
                    try:
                        raw_text = pdf_read(pdf_docs)
                        if not raw_text.strip():
                            st.error("❌ 无法从PDF中提取文本")
                            return

                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 文本已分割为 {len(text_chunks)} 个片段")

                        vector_store(text_chunks)
                        st.success("✅ PDF处理完成！")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 处理PDF时出错: {str(e)}")

            # 清除数据库
            if st.button("🗑️ 清除PDF数据库", use_container_width=True, key="clear_pdf"):
                try:
                    import shutil
                    if os.path.exists("faiss_db"):
                        shutil.rmtree("faiss_db")
                    st.session_state.pdf_messages = []
                    st.success("数据库已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")

            # 清除长期记忆库
            if st.button("🗑️ 清除长期记忆库", use_container_width=True, key="clear_mem"):
                try:
                    import shutil
                    if os.path.exists(MEMORY_DB_DIR):
                        shutil.rmtree(MEMORY_DB_DIR)
                    st.success("长期记忆库已清除")
                    st.rerun()
                except Exception as e:
                    st.error(f"清除失败: {e}")

            # 添加清除聊天记录按钮
            if st.button("🗑️ 清除聊天记录", use_container_width=True, key="clear_pdf_chat"):
                st.session_state.pdf_messages = []
                st.success("聊天记录已清除")
                st.rerun()

    # CSV数据分析模块
    with tab2:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📈 数据分析对话")

            # 显示数据状态
            if st.session_state.df is not None:
                st.markdown(
                    '<div class="info-card success-card"><span class="status-indicator status-ready">✅ 数据已加载完成</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-card warning-card"><span class="status-indicator status-waiting">⚠️ 请先上传CSV文件</span></div>',
                    unsafe_allow_html=True)

            chat_container = st.container()
            with chat_container:
                # 聊天界面
                for message in st.session_state.csv_messages:
                    with st.chat_message(message["role"]):
                        if message["type"] == "dataframe":
                            st.dataframe(message["content"])
                        elif message["type"] == "image":
                            st.write(message["content"])
                            if os.path.exists('plot.png'):
                                st.image('plot.png')
                        else:
                            st.markdown(message["content"])

            # 用户输入
            if csv_query := st.chat_input("📊 分析数据...", disabled=st.session_state.df is None, key="csv_input"):
                st.session_state.csv_messages.append({"role": "user", "content": csv_query, "type": "text"})
                with st.chat_message("user"):
                    st.markdown(csv_query)

                with st.chat_message("assistant"):
                    # 创建消息占位符用于流式输出
                    message_placeholder = st.empty()

                    with st.spinner("🔄 正在分析数据..."):
                        response = get_csv_response(csv_query, message_placeholder)

                    # 处理不同类型的响应
                    if isinstance(response, pd.DataFrame):
                        message_placeholder.empty()  # 清空流式输出
                        st.dataframe(response)
                        st.session_state.csv_messages.append(
                            {"role": "assistant", "content": response, "type": "dataframe"})
                    elif "GRAPH" in str(response):
                        message_placeholder.empty()  # 清空流式输出
                        text = str(response)[str(response).find("GRAPH") + 6:]
                        st.write(text)
                        if os.path.exists('plot.png'):
                            st.image('plot.png')
                        st.session_state.csv_messages.append({"role": "assistant", "content": text, "type": "image"})
                    else:
                        # 确保最终文本正确显示
                        message_placeholder.markdown(response)
                        st.session_state.csv_messages.append({"role": "assistant", "content": response, "type": "text"})
                st.rerun()

        with col2:
            st.markdown("### 📊 数据管理")
            st.slider("对话记忆窗口(轮数)", min_value=0, max_value=20, value=st.session_state.csv_history_pairs,
                      key="csv_history_pairs")
            st.checkbox("启用长期记忆", value=st.session_state.enable_long_memory_csv, key="enable_long_memory_csv")
            st.slider("长期记忆检索条目数", min_value=0, max_value=20, value=st.session_state.memory_k_csv, key="memory_k_csv")

            # CSV文件上传
            csv_file = st.file_uploader("📈 上传CSV文件", type='csv', key="csv_uploader")
            if csv_file:
                st.session_state.df = pd.read_csv(csv_file)
                st.success(f"✅ 数据加载成功!")

                # 显示数据预览
                with st.expander("👀 数据预览", expanded=True):
                    st.dataframe(st.session_state.df.head())
                    st.write(f"📏 数据维度: {st.session_state.df.shape[0]} 行 × {st.session_state.df.shape[1]} 列")

            # 数据信息
            if st.session_state.df is not None:
                if st.button("📋 上传并显示数据信息", use_container_width=True, key="show_info"):
                    with st.expander("📊 数据统计信息", expanded=True):
                        st.write("**基本信息:**")
                        st.text(f"行数: {st.session_state.df.shape[0]}")
                        st.text(f"列数: {st.session_state.df.shape[1]}")
                        st.write("**列名:**")
                        st.write(list(st.session_state.df.columns))
                        st.write("**数据类型:**")
                        dtype_info = pd.DataFrame({
                            '列名': st.session_state.df.columns,
                            '数据类型': [str(dtype) for dtype in st.session_state.df.dtypes]
                        })
                        st.dataframe(dtype_info, use_container_width=True)

            # 清除数据
            if st.button("🗑️ 清除CSV数据", use_container_width=True, key="clear_csv"):
                st.session_state.df = None
                st.session_state.csv_messages = []
                if os.path.exists('plot.png'):
                    os.remove('plot.png')
                st.success("数据已清除")
                st.rerun()

            # 添加清除聊天记录按钮
            if st.button("🗑️ 清除聊天记录", use_container_width=True, key="clear_csv_chat"):
                st.session_state.csv_messages = []
                st.success("聊天记录已清除")
                st.rerun()

    # 底部信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔧 技术栈:**")
        st.markdown("• LangChain • Streamlit • FAISS • DeepSeek")
    with col2:
        st.markdown("**✨ 功能特色:**")
        st.markdown("• PDF智能问答 • 数据可视化分析")
    with col3:
        st.markdown("**💡 使用提示:**")
        st.markdown("• 支持多文件上传 • 实时对话交互")


if __name__ == "__main__":
    main()