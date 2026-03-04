import streamlit as st
import os
from configs import *
from core.utils import get_db
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================================
# 1. 页面配置与美化
# ==========================================================
st.set_page_config(
    page_title="智域 RAG - 馒头系统",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #f1f3f5 !important; }
    .stStatus, .stExpander { border-radius: 6px !important; border: 1px solid #e9ecef !important; background-color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. 核心资源初始化
# ==========================================================
@st.cache_resource
def init_resource():
    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        llm = ChatOpenAI(
            model=LLM_MODEL_NAME, 
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )
        return emb, llm
    except Exception as e:
        st.error(f"模型初始化失败：{e}")
        return None, None

embedding_model, llm = init_resource()

dbs = {}
for key in DOMAINS:
    dbs[key] = get_db(key, DOMAINS, embedding_model, CHUNK_SIZE, CHUNK_OVERLAP)

# ==========================================================
# 3. 侧边栏
# ==========================================================
with st.sidebar:
    st.title("🐳 系统控制台")
    st.divider()
    st.subheader("📚 知识库监控")
    for name, (folder, _) in DOMAINS.items():
        doc_count = len(os.listdir(folder)) if os.path.exists(folder) else 0
        st.caption(f"**{name.upper()}** 库：已加载 {doc_count} 份文档")
    
    st.divider()
    top_k = st.slider("检索深度 (Top-K)", 1, 10, 5)
    
    if st.button("🗑️ 清除当前对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================================
# 4. 主界面：对话流处理
# ==========================================================
st.title("馒头知识库")
st.caption("基于模块化架构的多领域 RAG 系统")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"]=="user" else "🐳"):
        st.markdown(msg["content"])

# --- 重点：这里开始所有逻辑都缩进 4 个空格 ---
if query := st.chat_input("问问关于 Docker、NBA 或元宇宙的问题..."):
    # 1. 展示用户提问
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)
    
    # 2. 路由逻辑 (已经包含在 query 存在的前提下)
    query_l = query.lower()
    DOCKER_KEYS = ["docker", "容器", "镜像", "k8s", "部署"]
    NBA_KEYS = ["nba", "得分", "赛季", "球员", "球星", "篮板", "助攻", "表现", "数据", "场均"]
    
    # 获取本地球星列表
    existing_stars = [f.split('_')[0].lower() for f in os.listdir("nba_docs") if f.endswith('.txt')]
    
    if any(k in query_l for k in DOCKER_KEYS):
        domain = "docker"
    elif any(k in query_l for k in NBA_KEYS) or any(s in query_l for s in existing_stars):
        domain = "nba"
    else:
        domain = "metaverse"

    st.toast(f"💡 系统已自动切换至：{domain.upper()} 知识库", icon="🚀")
    
    current_db = dbs.get(domain)
    
    if current_db is None:
        st.warning(f"检测到领域 {domain} 未初始化。")
    else:
        retriever = current_db.as_retriever(search_kwargs={"k": top_k})
        
        prompt = ChatPromptTemplate.from_template("""
        你是一个专业的技术顾问。请结合【历史背景】和【已知资料】回答问题。
        
        【历史背景】: {history}
        【已知资料】: {context}
        【当前提问】: {question}
        
        如果资料中没有答案，请诚实说明。禁止编造。
        """)

        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "history": lambda x: history_str,
                "question": RunnablePassthrough()
            }
            | prompt | llm | StrOutputParser()
        )

        with st.chat_message("assistant", avatar="🐳"):
            res_placeholder = st.empty()
            full_ans = ""
            
            with st.status(f"🚀 正在检索 {domain.upper()} 知识库...", expanded=False) as status:
                for chunk in chain.stream(query):
                    full_ans += chunk
                    res_placeholder.markdown(full_ans + "▌")
                status.update(label=f"✅ {domain.upper()} 检索完成", state="complete")
            
            res_placeholder.markdown(full_ans)
            
            with st.expander("🔍 资料溯源"):
                source_docs = retriever.invoke(query)
                for i, d in enumerate(source_docs):
                    st.write(f"**片段 {i+1}** 来自：`{d.metadata.get('source', '文本资料')}`")
                    st.caption(d.page_content[:150] + "...")

        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": full_ans})