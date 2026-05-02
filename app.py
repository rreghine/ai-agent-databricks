"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  E-Commerce AI Agent — Parte 3 · Databricks Edition                          ║
║  Medallion Architecture · Delta Lake · Spark SQL · MLflow                    ║
║  Autoria: Rafael Reghine Munhoz | Data Analyst | MBA USP ESALQ               ║
╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import time, json, os, re
from datetime import datetime

# ─── Conexão com Databricks ───────────────────────────────────────────────────
# Tenta conectar via databricks-connect (Streamlit Cloud / local)
# Se já estiver dentro do Databricks, usa o spark nativo

DATABRICKS_MODE = False  # True = dentro do Databricks · False = remoto/local

def get_spark():
    """Retorna SparkSession — nativa (Databricks) ou remota (databricks-connect)."""
    global DATABRICKS_MODE
    try:
        # Dentro do Databricks — spark já existe no namespace
        s = spark  # noqa
        DATABRICKS_MODE = True
        return s
    except NameError:
        pass

    # Fora do Databricks — conecta via databricks-connect
    try:
        from databricks.connect import DatabricksSession
        spark_remote = DatabricksSession.builder.getOrCreate()
        return spark_remote
    except Exception:
        return None

_spark = get_spark()

def run_spark_query(sql: str) -> pd.DataFrame:
    """Executa Spark SQL e retorna DataFrame pandas."""
    if _spark:
        return _spark.sql(sql).toPandas()
    # Fallback: retorna DataFrame vazio com aviso
    st.warning("⚠️ Sem conexão com Databricks. Configure databricks-connect ou rode dentro do Databricks.")
    return pd.DataFrame()

# ─── Multi-LLM Router (igual orchestrator.py + run_databricks_agent) ──────────
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import mlflow

DB_SCHEMA_GOLD = """
ecommerce_ai.gold_performance_estados_ibge
  customer_state, regiao, pib_per_capita, classe_economica,
  total_pedidos, ticket_medio, media_itens, media_parcelas,
  pct_parcelados, satisfacao_media, pct_atrasados,
  media_dias_atraso, ticket_por_pib
"""

GROQ_MODELS = {
    "simple":  "llama-3.1-8b-instant",
    "medium":  "llama-3.3-70b-versatile",
}

LLM_COSTS = {
    "claude-sonnet-4-6":        {"input": 3.0,  "output": 15.0},
    "llama-3.1-8b-instant":     {"input": 0.0,  "output": 0.0},
    "llama-3.3-70b-versatile":  {"input": 0.0,  "output": 0.0},
}

BLOCKED_PATTERNS = [
    r"\b(cpf|cnpj|rg|senha|password|token|secret|email)\b",
    r"\b(drop|delete|truncate|insert|update|alter|create)\b",
]

def _get_secret(key: str) -> str | None:
    for fn in [
        lambda: os.environ.get(key),
        lambda: st.secrets.get(key),
    ]:
        try:
            v = fn()
            if v: return v
        except Exception:
            pass
    return None

def get_llm(complexity: str):
    groq_key      = _get_secret("GROQ_API_KEY")
    anthropic_key = _get_secret("ANTHROPIC_API_KEY")
    if complexity == "simple" and groq_key:
        return ChatGroq(model="llama-3.1-8b-instant",    api_key=groq_key), "llama-3.1-8b-instant"
    elif complexity == "medium" and groq_key:
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key), "llama-3.3-70b-versatile"
    elif anthropic_key:
        return ChatAnthropic(model="claude-sonnet-4-6",  api_key=anthropic_key, max_tokens=512), "claude-sonnet-4-6"
    raise ValueError("Configure GROQ_API_KEY ou ANTHROPIC_API_KEY")

def classify_complexity(p: str) -> str:
    n = len(p.split())
    if n <= 40: return "simple"
    if n <= 80: return "medium"
    return "complex"

def check_guardrails(p: str) -> tuple:
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, p.lower()):
            return False, "🚫 Bloqueado: contém termo sensível ou comando DDL."
    if len(p.split()) < 2:
        return False, "🚫 Pergunta muito curta."
    return True, ""

def run_agent(pergunta: str) -> dict:
    """Orquestrador integrado ao Gold Layer — igual run_databricks_agent."""
    t0 = time.time()

    ok, msg = check_guardrails(pergunta)
    if not ok:
        return {"guardrail": True, "resposta": msg, "pergunta": pergunta}

    complexity = classify_complexity(pergunta)
    try:
        llm, llm_name = get_llm(complexity)
    except ValueError as e:
        return {"guardrail": False, "erro": str(e), "pergunta": pergunta}

    # Gera SQL
    sql_prompt = f"""Especialista Spark SQL. Retorne APENAS o SQL sem markdown.
Schema: {DB_SCHEMA_GOLD}
Pergunta: {pergunta}
SQL:"""
    try:
        response  = llm.invoke([HumanMessage(content=sql_prompt)])
        sql_clean = re.sub(r"```sql|```", "", response.content).strip()
    except Exception as e:
        return {"guardrail": False, "erro": f"Erro LLM: {e}", "pergunta": pergunta}

    # Executa Spark SQL
    rows = []
    sql_error = ""
    sql_attempts = 0
    for attempt in range(2):
        sql_attempts += 1
        try:
            df_result = run_spark_query(sql_clean)
            rows = df_result.to_dict("records")
            break
        except Exception as e:
            sql_error = str(e)
            if attempt == 0:
                fix = llm.invoke([HumanMessage(content=f"Corrija o SQL:\n{sql_clean}\nErro:{sql_error}\nSQL corrigido:")])
                sql_clean = re.sub(r"```sql|```", "", fix.content).strip()

    if sql_error and not rows:
        return {"guardrail": False, "erro": sql_error, "sql_gerado": sql_clean,
                "pergunta": pergunta, "llm": llm_name, "complexity": complexity}

    # Interpreta
    if rows:
        interp = f"""Analista de dados brasileiro. Interprete em português. Máx 2 linhas. Use números.
Contexto: e-commerce + IBGE. Pergunta: {pergunta}
Resultado:\n{pd.DataFrame(rows).to_string(index=False)}\nResposta:"""
        try:
            resp2    = llm.invoke([HumanMessage(content=interp)])
            resposta = resp2.content.strip()
            to_extra = getattr(getattr(resp2, "usage_metadata", None), "output_tokens", 80)
        except Exception:
            resposta = str(rows[0])
            to_extra = 50
    else:
        resposta = "Nenhum dado encontrado."
        to_extra = 20

    # Métricas
    usage    = getattr(response, "usage_metadata", None)
    ti       = getattr(usage, "input_tokens",  300)
    to       = getattr(usage, "output_tokens", 100) + to_extra
    latencia = int((time.time() - t0) * 1000)
    costs    = LLM_COSTS.get(llm_name, {"input": 0.0, "output": 0.0})
    custo    = (ti / 1e6 * costs["input"]) + (to / 1e6 * costs["output"])

    return {
        "guardrail":    False,
        "pergunta":     pergunta,
        "resposta":     resposta,
        "sql_gerado":   sql_clean,
        "sql_attempts": sql_attempts,
        "resultado":    rows[:5],
        "llm":          llm_name,
        "complexity":   complexity,
        "tokens_input": ti,
        "tokens_output":to,
        "custo_usd":    round(custo, 6),
        "latencia_ms":  latencia,
    }

# ─── Paleta de cores ──────────────────────────────────────────────────────────
BG      = "#0d0f14"
SURF    = "#13161e"
SURF2   = "#1a1e2a"
BORDER  = "#252a38"
ACCENT  = "#4f7fe8"
ACCENT2 = "#38bdf8"
ACCENT3 = "#a78bfa"
SUCCESS = "#34d399"
WARN    = "#fbbf24"
DANGER  = "#f87171"
TXT     = "#e2e8f0"
TXT2    = "#8892a4"
TXT3    = "#4a5568"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent · Databricks Edition",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Global ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=Figtree:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Figtree', sans-serif; background: {BG}; color: {TXT}; }}
.main {{ background: {BG}; }}
.block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Header */
.orch-header {{
    background: linear-gradient(135deg, {SURF2} 0%, {SURF} 100%);
    border: 1px solid {BORDER}; border-top: 2px solid {ACCENT3};
    border-radius: 12px; padding: 20px 28px 18px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
}}
.orch-title {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 500; color: {TXT}; }}
.pill {{ font-size: .6rem; font-weight: 500; font-family: 'IBM Plex Mono', monospace;
         padding: 3px 10px; border-radius: 20px; letter-spacing: 0.5px; text-transform: uppercase; }}
.pill-purple {{ background: {ACCENT3}18; color: {ACCENT3}; border: 1px solid {ACCENT3}35; }}
.pill-blue   {{ background: {ACCENT}18;  color: {ACCENT};  border: 1px solid {ACCENT}35; }}
.pill-cyan   {{ background: {ACCENT2}18; color: {ACCENT2}; border: 1px solid {ACCENT2}35; }}
.pill-green  {{ background: {SUCCESS}18; color: {SUCCESS}; border: 1px solid {SUCCESS}35; }}

/* KPI Cards */
.kpi-card {{
    background: {SURF}; border: 1px solid {BORDER}; border-top: 2px solid var(--accent, {ACCENT});
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
}}
.kpi-label {{ font-size: .65rem; font-weight: 600; color: {TXT2}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
.kpi-value {{ font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; line-height: 1; }}
.kpi-sub   {{ font-size: .7rem; color: {TXT3}; margin-top: 4px; font-family: 'IBM Plex Mono', monospace; }}

/* Chat */
.msg-user {{ background: {SURF2}; border: 1px solid {BORDER}; border-radius: 10px 10px 2px 10px;
             padding: 12px 16px; margin: 8px 0; font-size: .9rem; }}
.msg-agent {{ background: {SURF}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT3};
              border-radius: 2px 10px 10px 10px; padding: 12px 16px; margin: 8px 0; font-size: .9rem; }}
.msg-block {{ background: {SURF}; border: 1px solid {DANGER}33; border-left: 3px solid {DANGER};
              border-radius: 2px 10px 10px 10px; padding: 12px 16px; margin: 8px 0; font-size: .9rem; }}
.msg-meta  {{ font-size: .65rem; color: {TXT3}; font-family: 'IBM Plex Mono', monospace;
              margin-top: 8px; display: flex; gap: 16px; flex-wrap: wrap; }}

/* Insight */
.insight-wrap  {{ background: {SURF}; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px; margin-bottom: 8px; }}
.insight-header{{ font-family: 'IBM Plex Mono', monospace; font-size: .75rem; font-weight: 600;
                  padding-bottom: 10px; margin-bottom: 10px; }}
.insight-item  {{ display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid {BORDER}; font-size: .83rem; }}
.insight-num   {{ font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: .75rem; flex-shrink: 0; }}

/* Medalion badges */
.layer-badge {{
    display: inline-block; padding: 3px 12px; border-radius: 6px;
    font-size: .65rem; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-right: 6px;
}}
.badge-bronze {{ background: #cd7f3222; color: #cd7f32; border: 1px solid #cd7f3244; }}
.badge-silver {{ background: #c0c0c022; color: #c0c0c0; border: 1px solid #c0c0c044; }}
.badge-gold   {{ background: {WARN}22;   color: {WARN};   border: 1px solid {WARN}44; }}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="orch-header">
  <div>
    <div class="orch-title">
      ◈ E-Commerce AI Agent
      <span style="font-size:.6rem;background:{ACCENT3}22;color:{ACCENT3};
                   border:1px solid {ACCENT3}44;border-radius:4px;padding:1px 7px;
                   letter-spacing:1.5px;margin-left:10px;vertical-align:middle;
                   text-transform:uppercase;">PARTE 4 · DATABRICKS</span>
    </div>
    <div style="font-size:.72rem;color:{TXT3};margin-top:5px;font-family:'IBM Plex Mono',monospace;">
      Medallion Architecture · Delta Lake · Spark SQL · MLflow · IBGE
    </div>
    <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;">
      <span class="pill pill-purple">LangGraph</span>
      <span class="pill pill-blue">Claude Sonnet</span>
      <span class="pill pill-cyan">Llama 3.3 70B</span>
      <span class="pill pill-green">Delta Lake</span>
      <span class="pill pill-green">MLflow</span>
    </div>
  </div>
  <div style="text-align:right;font-size:.7rem;color:{TXT3};font-family:'IBM Plex Mono',monospace;">
    <div>Rafael Reghine Munhoz</div>
    <div style="margin-top:4px;color:{TXT3}">MBA USP ESALQ · Data Science & Analytics</div>
    <div style="margin-top:4px;">
      <a href="https://linkedin.com/in/rafaelreghine" style="color:{ACCENT2};text-decoration:none;">linkedin</a>
      &nbsp;·&nbsp;
      <a href="https://github.com/rreghine" style="color:{ACCENT2};text-decoration:none;">github</a>
    </div>
    <div style="margin-top:6px;font-size:.6rem;">
      {'🟢 Databricks conectado' if DATABRICKS_MODE else '🔵 Modo remoto/local'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:{TXT2};
                border-bottom:1px solid {BORDER};padding-bottom:12px;margin-bottom:16px;">
      ◈ CONFIGURAÇÃO
    </div>""", unsafe_allow_html=True)

    # Status da conexão
    st.markdown(f"""
    <div style="font-size:.65rem;color:{TXT3};margin-bottom:8px;
                font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1px;">
      Status
    </div>""", unsafe_allow_html=True)

    spark_ok = _spark is not None
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{'#34d399' if spark_ok else '#f87171'}">
      <div class="kpi-label">Databricks</div>
      <div class="kpi-value" style="font-size:1rem;color:{'#34d399' if spark_ok else '#f87171'}">
        {'✅ Conectado' if spark_ok else '❌ Offline'}
      </div>
      <div class="kpi-sub">{'Gold Layer disponível' if spark_ok else 'Configure databricks-connect'}</div>
    </div>""", unsafe_allow_html=True)

    groq_ok      = bool(_get_secret("GROQ_API_KEY"))
    anthropic_ok = bool(_get_secret("ANTHROPIC_API_KEY"))
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{ACCENT}">
      <div class="kpi-label">LLMs</div>
      <div class="kpi-value" style="font-size:.85rem;color:{TXT}">
        {'✅' if groq_ok else '❌'} Groq &nbsp;·&nbsp; {'✅' if anthropic_ok else '❌'} Claude
      </div>
      <div class="kpi-sub">
        {'simple/medium: Llama (gratuito)' if groq_ok else 'Configure GROQ_API_KEY'}
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Arquitetura Medalão
    st.markdown(f"""
    <div style="font-size:.65rem;color:{TXT3};margin-bottom:10px;
                font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1px;">
      Arquitetura Medalão
    </div>
    <div style="font-size:.78rem;line-height:1.8;color:{TXT2};">
      <span class="layer-badge badge-bronze">Bronze</span> CSVs Olist + IBGE<br>
      <span class="layer-badge badge-silver">Silver</span> Joins + Limpeza<br>
      <span class="layer-badge badge-gold">Gold</span> Tabela Analítica<br>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Routing info
    st.markdown(f"""
    <div style="font-size:.65rem;color:{TXT3};margin-bottom:10px;
                font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:1px;">
      Multi-LLM Routing
    </div>
    <div style="font-size:.75rem;line-height:2;color:{TXT2};font-family:'IBM Plex Mono',monospace;">
      <span style="color:{SUCCESS}">simple</span>  → Llama 3.1 8B<br>
      <span style="color:{WARN}">medium</span>  → Llama 3.3 70B<br>
      <span style="color:{ACCENT}">complex</span> → Claude Sonnet<br>
    </div>""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Agente SQL",
    "📊 Dashboard",
    "🔬 Benchmark",
    "🏗️ Arquitetura",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AGENTE SQL (substitui SQLite pelo Gold Layer Databricks)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div style="font-size:.7rem;color:{TXT3};font-family:'IBM Plex Mono',monospace;
                margin-bottom:16px;">
      SQL Agent · Gold Layer → ecommerce_ai.gold_performance_estados_ibge
    </div>""", unsafe_allow_html=True)

    # Perguntas sugeridas
    sugestoes = [
        "Qual região tem maior taxa de atraso?",
        "Estados de baixa renda parcelam mais?",
        "Qual a relação entre PIB e satisfação?",
        "Quais os 3 estados mais eficientes em entrega?",
        "Compare ticket médio por classe econômica",
    ]
    st.markdown(f'<div style="font-size:.65rem;color:{TXT3};margin-bottom:6px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">Perguntas sugeridas</div>', unsafe_allow_html=True)
    cols_sug = st.columns(len(sugestoes))
    pergunta_selecionada = ""
    for col, sug in zip(cols_sug, sugestoes):
        with col:
            if st.button(sug, key=f"sug_{sug[:20]}", use_container_width=True):
                pergunta_selecionada = sug

    # Input
    pergunta = st.chat_input("Faça uma pergunta sobre o e-commerce brasileiro...")
    if pergunta_selecionada:
        pergunta = pergunta_selecionada

    # Histórico
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Processa pergunta
    if pergunta:
        st.session_state.chat_history.append({"role": "user", "content": pergunta})

        with st.spinner("Consultando Gold Layer..."):
            result = run_agent(pergunta)

        st.session_state.chat_history.append({"role": "agent", "result": result})

    # Renderiza histórico
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
              <span style="font-size:.65rem;color:{TXT3};font-family:IBM Plex Mono,monospace;">
                👤 VOCÊ
              </span><br>{msg['content']}
            </div>""", unsafe_allow_html=True)

        elif msg["role"] == "agent":
            r = msg["result"]
            if r.get("guardrail"):
                st.markdown(f"""
                <div class="msg-block">
                  <span style="font-size:.65rem;color:{DANGER};font-family:IBM Plex Mono,monospace;">
                    🚫 GUARDRAIL
                  </span><br>{r['resposta']}
                </div>""", unsafe_allow_html=True)

            elif r.get("erro"):
                st.markdown(f"""
                <div class="msg-block">
                  <span style="font-size:.65rem;color:{WARN};font-family:IBM Plex Mono,monospace;">
                    ⚠️ ERRO
                  </span><br>{r['erro'][:200]}
                </div>""", unsafe_allow_html=True)
            else:
                custo_str = f"${r['custo_usd']:.6f}" if r['custo_usd'] > 0 else "Gratuito"
                st.markdown(f"""
                <div class="msg-agent">
                  <span style="font-size:.65rem;color:{ACCENT3};font-family:IBM Plex Mono,monospace;">
                    ◈ AGENTE
                  </span><br>
                  {r['resposta']}
                  <div class="msg-meta">
                    <span>🤖 {r['llm']}</span>
                    <span>⚡ {r['complexity']}</span>
                    <span>⏱️ {r['latencia_ms']}ms</span>
                    <span>💰 {custo_str}</span>
                    <span>🔄 {r['sql_attempts']} tentativa(s)</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                with st.expander("🔍 SQL gerado + resultado"):
                    st.code(r.get("sql_gerado", ""), language="sql")
                    if r.get("resultado"):
                        st.dataframe(pd.DataFrame(r["resultado"]), use_container_width=True)

    if st.button("🗑️ Limpar conversa"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD (substitui SQLite pelo Gold Layer Databricks)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div style="font-size:.7rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:16px;">Gold Layer · ecommerce_ai.gold_performance_estados_ibge + IBGE</div>', unsafe_allow_html=True)

    if not spark_ok:
        st.warning("Dashboard indisponível sem conexão com Databricks.")
    else:
        @st.cache_data(ttl=300)
        def load_gold():
            return run_spark_query("""
                SELECT * FROM ecommerce_ai.gold_performance_estados_ibge
                ORDER BY total_pedidos DESC
            """)

        df = load_gold()

        if df.empty:
            st.warning("Tabela Gold não encontrada. Execute o notebook de construção primeiro.")
        else:
            # KPIs
            st.markdown(f'<div style="font-size:.65rem;color:{TXT3};margin-bottom:8px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">KPIs — Brasil</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:{ACCENT3}">
                  <div class="kpi-label">Total Pedidos</div>
                  <div class="kpi-value" style="color:{ACCENT3}">{df['total_pedidos'].sum():,.0f}</div>
                  <div class="kpi-sub">pedidos entregues</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:{ACCENT}">
                  <div class="kpi-label">Ticket Médio BR</div>
                  <div class="kpi-value" style="color:{ACCENT}">R${df['ticket_medio'].mean():.2f}</div>
                  <div class="kpi-sub">média nacional</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:{DANGER}">
                  <div class="kpi-label">Taxa Atraso</div>
                  <div class="kpi-value" style="color:{DANGER}">{df['pct_atrasados'].mean():.1f}%</div>
                  <div class="kpi-sub">média nacional</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""
                <div class="kpi-card" style="--accent:{SUCCESS}">
                  <div class="kpi-label">Satisfação</div>
                  <div class="kpi-value" style="color:{SUCCESS}">{df['satisfacao_media'].mean():.2f}/5</div>
                  <div class="kpi-sub">nota média</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # Gráficos
            fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=SURF)
            fig.patch.set_facecolor(SURF)

            def style_ax(ax, title):
                ax.set_facecolor(SURF2)
                ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=10)
                ax.tick_params(colors=TXT2, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color(BORDER)

            cmap_blue = LinearSegmentedColormap.from_list("b", [ACCENT+"44", ACCENT])
            cmap_purp = LinearSegmentedColormap.from_list("p", [ACCENT3+"44", ACCENT3])

            # 1. Top estados por pedidos
            top5 = df.nlargest(5, "total_pedidos")
            colors1 = [cmap_blue(v) for v in np.linspace(.3, 1., 5)]
            bars = axes[0,0].barh(top5["customer_state"], top5["total_pedidos"], color=colors1[::-1], height=.6)
            for b, v in zip(bars, top5["total_pedidos"]):
                axes[0,0].text(b.get_width() + top5["total_pedidos"].max()*.02,
                               b.get_y() + b.get_height()/2,
                               f"{int(v):,}", va="center", fontsize=8, color=TXT)
            axes[0,0].set_xlim(0, top5["total_pedidos"].max()*1.25)
            style_ax(axes[0,0], "Top 5 Estados — Pedidos")

            # 2. Parcelamento por classe econômica
            by_class = df.groupby("classe_economica")[["media_parcelas","pct_parcelados"]].mean().reset_index()
            x = np.arange(len(by_class))
            b1 = axes[0,1].bar(x - .2, by_class["media_parcelas"], .35,
                                color=ACCENT, label="Média Parcelas")
            b2 = axes[0,1].bar(x + .2, by_class["pct_parcelados"] / 10, .35,
                                color=ACCENT3, label="% Parcelados / 10")
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(by_class["classe_economica"], color=TXT2, fontsize=8)
            axes[0,1].legend(fontsize=7, labelcolor=TXT2, facecolor=SURF)
            style_ax(axes[0,1], "Parcelamento por Classe Econômica")

            # 3. PIB vs Ticket médio
            scatter_colors = [ACCENT if c=="Rico" else ACCENT3 if c=="Médio" else DANGER
                              for c in df["classe_economica"]]
            axes[1,0].scatter(df["pib_per_capita"], df["ticket_medio"],
                              c=scatter_colors, s=80, alpha=.85, edgecolors=BORDER, linewidth=.5)
            for _, row in df.iterrows():
                axes[1,0].annotate(row["customer_state"],
                                   (row["pib_per_capita"], row["ticket_medio"]),
                                   fontsize=6, color=TXT3, xytext=(3, 3),
                                   textcoords="offset points")
            axes[1,0].set_xlabel("PIB per capita (R$)", color=TXT2, fontsize=8)
            axes[1,0].set_ylabel("Ticket Médio (R$)", color=TXT2, fontsize=8)
            style_ax(axes[1,0], "PIB per capita × Ticket Médio — IBGE + Olist")

            # 4. Taxa de atraso por região
            by_reg = df.groupby("regiao")["pct_atrasados"].mean().sort_values(ascending=False)
            bar_colors = [DANGER if v > 12 else WARN if v > 8 else SUCCESS for v in by_reg.values]
            b4 = axes[1,1].bar(by_reg.index, by_reg.values, color=bar_colors, width=.6)
            for b, v in zip(b4, by_reg.values):
                axes[1,1].text(b.get_x() + b.get_width()/2, b.get_height() + .3,
                               f"{v:.1f}%", ha="center", fontsize=8, color=TXT)
            axes[1,1].set_ylabel("Taxa Atraso (%)", color=TXT2, fontsize=8)
            style_ax(axes[1,1], "Taxa de Atraso por Região")

            fig.text(.5, -.01,
                     "Rafael Reghine Munhoz · github.com/rreghine · MBA USP ESALQ · Dados: Olist + IBGE 2018",
                     ha="center", fontsize=7.5, color=TXT3)
            plt.tight_layout()
            st.pyplot(fig)

            st.divider()
            st.markdown(f'<div style="font-size:.65rem;color:{TXT3};margin-bottom:8px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">Tabela Gold Completa</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK MULTI-LLM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div style="font-size:.7rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:16px;">Benchmark · 3 LLMs · mesmas perguntas · mesmo Gold Layer</div>', unsafe_allow_html=True)

    BENCHMARK_QUERIES = [
        "Qual região tem maior taxa de atraso?",
        "Estados de baixa renda parcelam mais?",
        "Qual estado tem maior ticket médio?",
        "Compare satisfação por classe econômica",
    ]

    if st.button("▶️ Rodar Benchmark", type="primary"):
        resultados_bench = {}
        prog = st.progress(0)
        total = len(BENCHMARK_QUERIES)

        for i, q in enumerate(BENCHMARK_QUERIES):
            resultados_bench[q] = run_agent(q)
            prog.progress((i + 1) / total)

        st.session_state["bench_results"] = resultados_bench
        prog.empty()

    if st.session_state.get("bench_results"):
        res = st.session_state["bench_results"]

        # Métricas consolidadas
        st.markdown(f'<div style="font-size:.65rem;color:{TXT3};margin-bottom:8px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">Métricas Consolidadas</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        total_latencia = sum(r.get("latencia_ms", 0) for r in res.values())
        total_custo    = sum(r.get("custo_usd", 0) for r in res.values())
        total_tokens   = sum(r.get("tokens_input", 0) + r.get("tokens_output", 0) for r in res.values())
        modelos_usados = list(set(r.get("llm", "") for r in res.values() if not r.get("guardrail")))

        with m1:
            st.markdown(f'<div class="kpi-card" style="--accent:{ACCENT3}"><div class="kpi-label">Latência Total</div><div class="kpi-value" style="color:{ACCENT3}">{total_latencia:,}ms</div><div class="kpi-sub">{len(BENCHMARK_QUERIES)} queries</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="kpi-card" style="--accent:{ACCENT}"><div class="kpi-label">Custo Total</div><div class="kpi-value" style="color:{ACCENT}">${total_custo:.6f}</div><div class="kpi-sub">USD</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="kpi-card" style="--accent:{SUCCESS}"><div class="kpi-label">Tokens Total</div><div class="kpi-value" style="color:{SUCCESS}">{total_tokens:,}</div><div class="kpi-sub">input + output</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="kpi-card" style="--accent:{WARN}"><div class="kpi-label">LLMs usados</div><div class="kpi-value" style="font-size:.85rem;color:{WARN}">{len(modelos_usados)}</div><div class="kpi-sub">{" · ".join(modelos_usados)[:40]}</div></div>', unsafe_allow_html=True)

        st.divider()

        # Resultados por query
        for q, r in res.items():
            custo_str = f"${r['custo_usd']:.6f}" if r.get("custo_usd", 0) > 0 else "Gratuito"
            with st.expander(f"❓ {q}"):
                if r.get("guardrail"):
                    st.error(r["resposta"])
                elif r.get("erro"):
                    st.warning(r["erro"])
                else:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**💬 Resposta:** {r['resposta']}")
                        st.code(r.get("sql_gerado", ""), language="sql")
                    with c2:
                        st.markdown(f"""
                        **🤖 LLM:** `{r['llm']}`
                        **⚡ Complexidade:** `{r['complexity']}`
                        **⏱️ Latência:** `{r['latencia_ms']}ms`
                        **💰 Custo:** `{custo_str}`
                        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARQUITETURA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div style="font-size:.7rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:16px;">Arquitetura do Projeto — Parte 1 → 4</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BORDER};border-radius:12px;padding:24px;font-family:'IBM Plex Mono',monospace;font-size:.78rem;line-height:2;color:{TXT2};">
      <div style="color:{ACCENT3};font-weight:600;margin-bottom:12px;font-size:.85rem;">Jornada das 4 Partes</div>

      <span style="color:{TXT3}">Parte 1</span> → <span style="color:{ACCENT2}">RAG Agent</span>
        · FAISS · Embeddings · CSV Olist<br>

      <span style="color:{TXT3}">Parte 2</span> → <span style="color:{ACCENT2}">SQL Agent</span>
        · Text-to-SQL · SQLite · Ground Truth · LLM-as-Judge<br>

      <span style="color:{TXT3}">Parte 3</span> → <span style="color:{ACCENT3}">Orquestrador</span>
        · LangGraph · Multi-LLM Routing · MLflow · Streamlit<br>

      <span style="color:{TXT3}">Parte 4</span> → <span style="color:{WARN}">Produção Databricks</span>
        · Medallion · Delta Lake · Spark SQL · Unity Catalog · IBGE<br>

      <div style="color:{ACCENT3};font-weight:600;margin:20px 0 12px;font-size:.85rem;">Arquitetura Medalão</div>

      <span class="layer-badge badge-bronze">Bronze</span>
        7 tabelas Olist (99k pedidos) + IBGE → Delta Lake raw<br>
      <span class="layer-badge badge-silver">Silver</span>
        Joins, limpeza, tipagem, flags de atraso e satisfação<br>
      <span class="layer-badge badge-gold">Gold</span>
        Tabela analítica com PIB per capita — insights impossíveis sem cruzamento<br>

      <div style="color:{ACCENT3};font-weight:600;margin:20px 0 12px;font-size:.85rem;">Multi-LLM Routing (cost-aware)</div>

      <span style="color:{SUCCESS}">simple  (≤40 palavras)</span> → Llama 3.1 8B  via Groq  · gratuito · ~900ms<br>
      <span style="color:{WARN}">medium  (41-80 palavras)</span> → Llama 3.3 70B via Groq  · gratuito · ~1.5s<br>
      <span style="color:{ACCENT}">complex (>80 palavras)</span>  → Claude Sonnet via Anthropic · $3/1M tokens<br>

      <div style="color:{ACCENT3};font-weight:600;margin:20px 0 12px;font-size:.85rem;">Insights Descobertos</div>

      <span style="color:{DANGER}">▲</span> Nordeste tem 15.2% de atraso — 2x mais que o Sul (7.3%)<br>
      <span style="color:{WARN}">▲</span> Baixa Renda: 59.6% parcelam · Rico: 50.9% — diferença real<br>
      <span style="color:{SUCCESS}">▲</span> Paradoxo: estados pobres têm ticket_por_pib = 12.89 vs 3.52 nos ricos<br>
      <span style="color:{ACCENT2}">▲</span> PR, MG, SP lideram em eficiência logística<br>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<div style="font-size:.65rem;color:{TXT3};text-align:center;font-family:IBM Plex Mono,monospace;">Rafael Reghine Munhoz · MBA USP ESALQ · Data Science & Analytics · github.com/rreghine</div>', unsafe_allow_html=True)
