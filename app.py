"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  E-Commerce AI Agent - Parte 4 · Databricks Edition                          ║
║  Medallion Architecture · Delta Lake · Spark SQL · MLflow                    ║
║  Olist E-Commerce + IBGE PIB per Capita                                      ║
║  Autoria: Rafael Reghine Munhoz | Data Analyst                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Gold Layer criado no Databricks workspace:
  ecommerce_ai.gold_performance_estados_ibge
  27 estados · Olist + IBGE · Delta Lake · Unity Catalog
"""

# ─── Imports básicos ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import time
import os
import re

# ─── Page config - DEVE ser o primeiro comando st.* ───────────────────────────
st.set_page_config(
    page_title="AI Agent · Databricks",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports com fallback (nao crasham o app) ─────────────────────────────────
LANGCHAIN_OK = False
MLFLOW_OK    = False
SPARK_OK     = False
_spark       = None

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    LANGCHAIN_OK = True
except Exception:
    pass

# OpenAI opcional
OPENAI_OK = False
try:
    from langchain_openai import ChatOpenAI
    OPENAI_OK = True
except Exception:
    pass

# Gemini opcional
GEMINI_OK = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_OK = True
except Exception:
    pass

try:
    import mlflow
    MLFLOW_OK = True
except Exception:
    pass

try:
    _spark   = spark  # noqa: F821 - nativo no Databricks Apps
    SPARK_OK = True
except NameError:
    try:
        from databricks.connect import DatabricksSession
        _spark   = DatabricksSession.builder.getOrCreate()
        SPARK_OK = True
    except Exception:
        pass

# ─── Dados estáticos do Gold Layer ────────────────────────────────────────────
# Espelham exatamente ecommerce_ai.gold_performance_estados_ibge
# criado no notebook do Databricks workspace
GOLD_STATIC = [
    {"customer_state":"SP","regiao":"Sudeste","pib_per_capita":51559.3,"classe_economica":"Rico","total_pedidos":40501,"ticket_medio":163.40,"media_itens":1.12,"media_parcelas":2.87,"pct_parcelados":49.8,"satisfacao_media":4.25,"pct_atrasados":5.9,"media_dias_atraso":-3.2,"ticket_por_pib":3.17},
    {"customer_state":"RJ","regiao":"Sudeste","pib_per_capita":43685.4,"classe_economica":"Rico","total_pedidos":12767,"ticket_medio":168.30,"media_itens":1.13,"media_parcelas":3.01,"pct_parcelados":52.1,"satisfacao_media":4.08,"pct_atrasados":9.8,"media_dias_atraso":-1.8,"ticket_por_pib":3.85},
    {"customer_state":"MG","regiao":"Sudeste","pib_per_capita":30977.8,"classe_economica":"Médio","total_pedidos":11354,"ticket_medio":162.10,"media_itens":1.11,"media_parcelas":3.04,"pct_parcelados":53.2,"satisfacao_media":4.19,"pct_atrasados":5.6,"media_dias_atraso":-3.5,"ticket_por_pib":5.23},
    {"customer_state":"RS","regiao":"Sul","pib_per_capita":41717.2,"classe_economica":"Rico","total_pedidos":5345,"ticket_medio":171.20,"media_itens":1.14,"media_parcelas":2.95,"pct_parcelados":51.3,"satisfacao_media":4.19,"pct_atrasados":7.1,"media_dias_atraso":-2.9,"ticket_por_pib":4.10},
    {"customer_state":"PR","regiao":"Sul","pib_per_capita":40696.3,"classe_economica":"Rico","total_pedidos":4923,"ticket_medio":167.80,"media_itens":1.13,"media_parcelas":2.91,"pct_parcelados":50.2,"satisfacao_media":4.24,"pct_atrasados":5.0,"media_dias_atraso":-3.8,"ticket_por_pib":4.12},
    {"customer_state":"SC","regiao":"Sul","pib_per_capita":46645.1,"classe_economica":"Rico","total_pedidos":3546,"ticket_medio":174.50,"media_itens":1.15,"media_parcelas":2.88,"pct_parcelados":49.1,"satisfacao_media":4.21,"pct_atrasados":6.2,"media_dias_atraso":-3.1,"ticket_por_pib":3.74},
    {"customer_state":"BA","regiao":"Nordeste","pib_per_capita":18301.6,"classe_economica":"Baixa Renda","total_pedidos":3380,"ticket_medio":208.40,"media_itens":1.18,"media_parcelas":3.38,"pct_parcelados":58.9,"satisfacao_media":3.99,"pct_atrasados":13.1,"media_dias_atraso":1.2,"ticket_por_pib":11.39},
    {"customer_state":"DF","regiao":"Centro-Oeste","pib_per_capita":84209.9,"classe_economica":"Rico","total_pedidos":2144,"ticket_medio":178.90,"media_itens":1.16,"media_parcelas":2.85,"pct_parcelados":48.7,"satisfacao_media":4.18,"pct_atrasados":7.8,"media_dias_atraso":-2.5,"ticket_por_pib":2.12},
    {"customer_state":"GO","regiao":"Centro-Oeste","pib_per_capita":30011.5,"classe_economica":"Médio","total_pedidos":2020,"ticket_medio":165.70,"media_itens":1.12,"media_parcelas":3.01,"pct_parcelados":52.8,"satisfacao_media":4.14,"pct_atrasados":8.2,"media_dias_atraso":-2.1,"ticket_por_pib":5.52},
    {"customer_state":"PE","regiao":"Nordeste","pib_per_capita":19023.5,"classe_economica":"Baixa Renda","total_pedidos":1652,"ticket_medio":212.10,"media_itens":1.19,"media_parcelas":3.42,"pct_parcelados":59.8,"satisfacao_media":3.95,"pct_atrasados":14.8,"media_dias_atraso":1.8,"ticket_por_pib":11.15},
    {"customer_state":"CE","regiao":"Nordeste","pib_per_capita":15673.4,"classe_economica":"Baixa Renda","total_pedidos":1336,"ticket_medio":215.30,"media_itens":1.20,"media_parcelas":3.51,"pct_parcelados":61.2,"satisfacao_media":3.93,"pct_atrasados":16.4,"media_dias_atraso":2.1,"ticket_por_pib":13.74},
    {"customer_state":"ES","regiao":"Sudeste","pib_per_capita":37182.1,"classe_economica":"Médio","total_pedidos":1333,"ticket_medio":163.90,"media_itens":1.11,"media_parcelas":2.98,"pct_parcelados":52.0,"satisfacao_media":4.15,"pct_atrasados":6.8,"media_dias_atraso":-3.0,"ticket_por_pib":4.41},
    {"customer_state":"MA","regiao":"Nordeste","pib_per_capita":11329.4,"classe_economica":"Baixa Renda","total_pedidos":747,"ticket_medio":221.50,"media_itens":1.21,"media_parcelas":3.58,"pct_parcelados":62.4,"satisfacao_media":3.90,"pct_atrasados":18.2,"media_dias_atraso":2.8,"ticket_por_pib":19.55},
    {"customer_state":"MT","regiao":"Centro-Oeste","pib_per_capita":43415.2,"classe_economica":"Rico","total_pedidos":886,"ticket_medio":172.30,"media_itens":1.14,"media_parcelas":2.93,"pct_parcelados":50.8,"satisfacao_media":4.15,"pct_atrasados":6.8,"media_dias_atraso":-2.8,"ticket_por_pib":3.97},
    {"customer_state":"MS","regiao":"Centro-Oeste","pib_per_capita":35441.7,"classe_economica":"Médio","total_pedidos":715,"ticket_medio":166.80,"media_itens":1.12,"media_parcelas":3.00,"pct_parcelados":52.5,"satisfacao_media":4.12,"pct_atrasados":8.5,"media_dias_atraso":-2.2,"ticket_por_pib":4.71},
    {"customer_state":"PB","regiao":"Nordeste","pib_per_capita":15073.2,"classe_economica":"Baixa Renda","total_pedidos":536,"ticket_medio":266.61,"media_itens":1.22,"media_parcelas":3.62,"pct_parcelados":63.4,"satisfacao_media":3.88,"pct_atrasados":11.0,"media_dias_atraso":0.8,"ticket_por_pib":17.69},
    {"customer_state":"PI","regiao":"Nordeste","pib_per_capita":13261.8,"classe_economica":"Baixa Renda","total_pedidos":412,"ticket_medio":219.80,"media_itens":1.20,"media_parcelas":3.55,"pct_parcelados":61.8,"satisfacao_media":3.91,"pct_atrasados":17.1,"media_dias_atraso":2.4,"ticket_por_pib":16.58},
    {"customer_state":"RN","regiao":"Nordeste","pib_per_capita":17951.3,"classe_economica":"Baixa Renda","total_pedidos":481,"ticket_medio":213.60,"media_itens":1.19,"media_parcelas":3.44,"pct_parcelados":60.1,"satisfacao_media":3.94,"pct_atrasados":15.2,"media_dias_atraso":1.9,"ticket_por_pib":11.90},
    {"customer_state":"SE","regiao":"Nordeste","pib_per_capita":17603.1,"classe_economica":"Baixa Renda","total_pedidos":348,"ticket_medio":210.90,"media_itens":1.19,"media_parcelas":3.40,"pct_parcelados":59.4,"satisfacao_media":3.96,"pct_atrasados":14.1,"media_dias_atraso":1.5,"ticket_por_pib":11.98},
    {"customer_state":"AL","regiao":"Nordeste","pib_per_capita":14271.2,"classe_economica":"Baixa Renda","total_pedidos":413,"ticket_medio":237.21,"media_itens":1.21,"media_parcelas":3.48,"pct_parcelados":67.0,"satisfacao_media":3.85,"pct_atrasados":23.9,"media_dias_atraso":3.5,"ticket_por_pib":16.62},
    {"customer_state":"PA","regiao":"Norte","pib_per_capita":17118.6,"classe_economica":"Baixa Renda","total_pedidos":472,"ticket_medio":218.70,"media_itens":1.20,"media_parcelas":3.42,"pct_parcelados":60.2,"satisfacao_media":4.05,"pct_atrasados":8.9,"media_dias_atraso":-1.2,"ticket_por_pib":12.78},
    {"customer_state":"AM","regiao":"Norte","pib_per_capita":22644.3,"classe_economica":"Baixa Renda","total_pedidos":340,"ticket_medio":225.40,"media_itens":1.20,"media_parcelas":3.38,"pct_parcelados":59.8,"satisfacao_media":4.08,"pct_atrasados":7.2,"media_dias_atraso":-1.8,"ticket_por_pib":9.95},
    {"customer_state":"AC","regiao":"Norte","pib_per_capita":16423.5,"classe_economica":"Baixa Renda","total_pedidos":81,"ticket_medio":244.69,"media_itens":1.21,"media_parcelas":3.52,"pct_parcelados":65.0,"satisfacao_media":4.10,"pct_atrasados":3.8,"media_dias_atraso":-4.1,"ticket_por_pib":14.90},
    {"customer_state":"AP","regiao":"Norte","pib_per_capita":17463.8,"classe_economica":"Baixa Renda","total_pedidos":67,"ticket_medio":240.92,"media_itens":1.21,"media_parcelas":3.45,"pct_parcelados":56.7,"satisfacao_media":4.12,"pct_atrasados":4.5,"media_dias_atraso":-3.8,"ticket_por_pib":13.80},
    {"customer_state":"RO","regiao":"Norte","pib_per_capita":23414.5,"classe_economica":"Baixa Renda","total_pedidos":253,"ticket_medio":234.43,"media_itens":1.21,"media_parcelas":3.38,"pct_parcelados":59.3,"satisfacao_media":4.10,"pct_atrasados":2.9,"media_dias_atraso":-4.5,"ticket_por_pib":10.01},
    {"customer_state":"RR","regiao":"Norte","pib_per_capita":21012.6,"classe_economica":"Baixa Renda","total_pedidos":46,"ticket_medio":231.80,"media_itens":1.20,"media_parcelas":3.41,"pct_parcelados":58.7,"satisfacao_media":4.11,"pct_atrasados":5.2,"media_dias_atraso":-3.2,"ticket_por_pib":11.03},
    {"customer_state":"TO","regiao":"Norte","pib_per_capita":23591.4,"classe_economica":"Baixa Renda","total_pedidos":537,"ticket_medio":222.10,"media_itens":1.20,"media_parcelas":3.35,"pct_parcelados":58.1,"satisfacao_media":4.09,"pct_atrasados":6.8,"media_dias_atraso":-2.1,"ticket_por_pib":9.41},
]

def get_gold_data() -> pd.DataFrame:
    """Gold Layer: tenta Spark SQL, fallback para dados estáticos."""
    if SPARK_OK and _spark:
        try:
            df = _spark.sql("SELECT * FROM ecommerce_ai.gold_performance_estados_ibge ORDER BY total_pedidos DESC").toPandas()
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame(GOLD_STATIC).sort_values("total_pedidos", ascending=False).reset_index(drop=True)

def _get_secret(key: str) -> str:
    # 1. Variavel de ambiente (Databricks Apps Environment tab)
    v = os.environ.get(key)
    if v: return v
    # 2. Streamlit secrets (secrets.toml) - silencia o warning
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = st.secrets.get(key)
            if v: return v
    except Exception:
        pass
    return ""

# ─── LLM e Agente ─────────────────────────────────────────────────────────────
LLM_COSTS = {
    "claude-sonnet-4-6":        {"input":3.0,   "output":15.0},
    "gpt-4o-mini":              {"input":0.15,  "output":0.60},
    "gpt-4o":                   {"input":2.50,  "output":10.0},
    "gemini-2.0-flash":         {"input":0.10,  "output":0.40},
    "gemini-1.5-pro":           {"input":1.25,  "output":5.0},
    "llama-3.1-8b-instant":     {"input":0.0,   "output":0.0},
    "llama-3.3-70b-versatile":  {"input":0.0,   "output":0.0},
    "mistral-saba-24b":         {"input":0.0,   "output":0.0},
    "qwen-qwq-32b":             {"input":0.0,   "output":0.0},
}

# Mapa nome legivel -> config
ALL_MODELS = {
    "Llama 3.1 8B (Gratis)":    {"key":"GROQ_API_KEY",      "model":"llama-3.1-8b-instant",    "lib":"groq"},
    "Llama 3.3 70B (Gratis)":   {"key":"GROQ_API_KEY",      "model":"llama-3.3-70b-versatile", "lib":"groq"},
    "Mistral Saba 24B (Gratis)":{"key":"GROQ_API_KEY",      "model":"mistral-saba-24b",        "lib":"groq"},
    "Qwen QwQ 32B (Gratis)":    {"key":"GROQ_API_KEY",      "model":"qwen-qwq-32b",            "lib":"groq"},
    "Claude Sonnet 4.6":        {"key":"ANTHROPIC_API_KEY",  "model":"claude-sonnet-4-6",       "lib":"anthropic"},
    "GPT-4o Mini":              {"key":"OPENAI_API_KEY",     "model":"gpt-4o-mini",             "lib":"openai"},
    "GPT-4o":                   {"key":"OPENAI_API_KEY",     "model":"gpt-4o",                  "lib":"openai"},
    "Gemini 2.0 Flash":         {"key":"GEMINI_API_KEY",     "model":"gemini-2.0-flash",        "lib":"gemini"},
    "Gemini 1.5 Pro":           {"key":"GEMINI_API_KEY",     "model":"gemini-1.5-pro",          "lib":"gemini"},
}

def build_llm(model_name: str, api_key: str):
    """Instancia o LLM correto baseado no nome do modelo."""
    cfg = ALL_MODELS.get(model_name, {})
    lib = cfg.get("lib","")
    mdl = cfg.get("model", model_name)
    if lib == "groq":      return ChatGroq(model=mdl, api_key=api_key)
    if lib == "anthropic": return ChatAnthropic(model=mdl, api_key=api_key, max_tokens=512)
    if lib == "openai" and OPENAI_OK: return ChatOpenAI(model=mdl, api_key=api_key, max_tokens=512)
    if lib == "gemini" and GEMINI_OK: return ChatGoogleGenerativeAI(model=mdl, google_api_key=api_key, max_output_tokens=512)
    return None

def get_available_models() -> list:
    """Retorna lista de modelos disponiveis com base nas keys configuradas."""
    available = []
    for name, cfg in ALL_MODELS.items():
        key_val = _get_secret(cfg["key"])
        if not key_val: continue
        lib = cfg["lib"]
        if lib == "groq"     and not LANGCHAIN_OK: continue
        if lib == "openai"   and not OPENAI_OK:    continue
        if lib == "gemini"   and not GEMINI_OK:    continue
        available.append(name)
    return available

DB_SCHEMA = """
Tabela: ecommerce_ai.gold_performance_estados_ibge (27 linhas - 1 por estado)
Colunas: customer_state, regiao, pib_per_capita, classe_economica,
         total_pedidos, ticket_medio, media_itens, media_parcelas,
         pct_parcelados, satisfacao_media, pct_atrasados, media_dias_atraso, ticket_por_pib
Regras: use Spark SQL, sempre LIMIT, não use DDL.
"""

BLOCKED = [r"\b(cpf|cnpj|senha|password|token|secret)\b", r"\b(drop|delete|truncate|insert|update|alter)\b"]

def classify(p): return "simple" if len(p.split())<=40 else "medium" if len(p.split())<=80 else "complex"

def guardrail(p):
    for pat in BLOCKED:
        if re.search(pat, p.lower()): return False, "Bloqueado: termo sensivel ou DDL."
    if len(p.split()) < 2: return False, "Pergunta muito curta."
    return True, ""

def get_llm(complexity, model_override=None):
    """Retorna LLM baseado em complexidade ou override manual."""
    if not LANGCHAIN_OK: return None, None
    gk = _get_secret("GROQ_API_KEY")
    ak = _get_secret("ANTHROPIC_API_KEY")
    ok = _get_secret("OPENAI_API_KEY")
    gm = _get_secret("GEMINI_API_KEY")

    # Override manual (usuario escolheu o modelo)
    if model_override and model_override in ALL_MODELS:
        cfg     = ALL_MODELS[model_override]
        key_val = _get_secret(cfg["key"])
        if key_val:
            llm = build_llm(model_override, key_val)
            if llm: return llm, cfg["model"]

    # Routing automatico por complexidade
    if complexity == "simple"  and gk: return ChatGroq(model="llama-3.1-8b-instant",    api_key=gk), "llama-3.1-8b-instant"
    if complexity == "medium"  and gk: return ChatGroq(model="llama-3.3-70b-versatile", api_key=gk), "llama-3.3-70b-versatile"
    if ak: return ChatAnthropic(model="claude-sonnet-4-6", api_key=ak, max_tokens=512), "claude-sonnet-4-6"
    if ok and OPENAI_OK:  return ChatOpenAI(model="gpt-4o", api_key=ok, max_tokens=512), "gpt-4o"
    if gm and GEMINI_OK:  return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gm, max_output_tokens=512), "gemini-2.0-flash"
    if gk: return ChatGroq(model="llama-3.3-70b-versatile", api_key=gk), "llama-3.3-70b-versatile"
    return None, None

def run_agent(pergunta: str, model_override: str = None) -> dict:
    t0 = time.time()
    ok, msg = guardrail(pergunta)
    if not ok: return {"guardrail":True,"resposta":msg,"pergunta":pergunta}

    complexity = classify(pergunta)
    llm, llm_name = get_llm(complexity, model_override)

    if not llm:
        return {"guardrail":False,"pergunta":pergunta,"llm":"-","complexity":complexity,
                "resposta":"⚠️ Configure GROQ_API_KEY (gratuito: console.groq.com) ou ANTHROPIC_API_KEY.",
                "sql_gerado":"-","sql_attempts":0,"resultado":[],"latencia_ms":0,"custo_usd":0.0}

    # Gera SQL
    try:
        resp = llm.invoke([HumanMessage(content=f"Especialista Spark SQL. Retorne APENAS SQL sem markdown.\nSchema:\n{DB_SCHEMA}\nPergunta: {pergunta}\nSQL:")])
        sql  = re.sub(r"```sql|```","",resp.content).strip()
    except Exception as e:
        return {"guardrail":False,"erro":f"Erro LLM: {e}","pergunta":pergunta,"llm":llm_name,"complexity":complexity,"latencia_ms":int((time.time()-t0)*1000)}

    # Executa SQL
    rows, sql_error, attempts = [], "", 0
    for attempt in range(2):
        attempts += 1
        try:
            if SPARK_OK and _spark:
                rows = [r.asDict() for r in _spark.sql(sql).limit(20).collect()]
            else:
                df_g = get_gold_data()
                rows = df_g.head(10).to_dict("records")
            sql_error = ""
            break
        except Exception as e:
            sql_error = str(e)
            if attempt == 0:
                try:
                    fix = llm.invoke([HumanMessage(content=f"Corrija:\n{sql}\nErro:{sql_error}\nSQL corrigido:")])
                    sql = re.sub(r"```sql|```","",fix.content).strip()
                except Exception: break

    # Interpreta
    if rows:
        try:
            r2 = llm.invoke([HumanMessage(content=f"Analista brasileiro. Interprete em português. Máx 2 linhas. Use números.\nContexto: e-commerce + IBGE.\nPergunta: {pergunta}\nResultado:\n{pd.DataFrame(rows).to_string(index=False)}\nResposta:")])
            resposta = r2.content.strip()
            to_extra = getattr(getattr(r2,"usage_metadata",None),"output_tokens",80)
        except Exception:
            resposta = str(rows[0]); to_extra = 50
    else:
        resposta = "Nenhum dado encontrado." + (f" Erro: {sql_error[:60]}" if sql_error else "")
        to_extra = 20

    usage = getattr(resp,"usage_metadata",None)
    ti    = getattr(usage,"input_tokens",300)
    to    = getattr(usage,"output_tokens",100) + to_extra
    lat   = int((time.time()-t0)*1000)
    costs = LLM_COSTS.get(llm_name,{"input":0,"output":0})
    custo = (ti/1e6*costs["input"]) + (to/1e6*costs["output"])

    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=f"{llm_name}_{int(t0)}", nested=True):
                mlflow.log_params({"llm":llm_name,"complexity":complexity,"attempts":attempts})
                mlflow.log_metrics({"tokens_in":ti,"tokens_out":to,"custo_usd":custo,"latencia_ms":lat})
                mlflow.set_tags({"gold_table":"ecommerce_ai.gold_performance_estados_ibge","parte":"4"})
        except Exception: pass

    return {"guardrail":False,"pergunta":pergunta,"resposta":resposta,"sql_gerado":sql,
            "sql_attempts":attempts,"resultado":rows[:5],"llm":llm_name,"complexity":complexity,
            "tokens_input":ti,"tokens_output":to,"custo_usd":round(custo,6),"latencia_ms":lat}

# ─── Paleta ───────────────────────────────────────────────────────────────────
BG=  "#0d0f14"; SURF="#13161e"; SURF2="#1a1e2a"; BORDER="#252a38"
ACCENT="#4f7fe8"; ACCENT2="#38bdf8"; ACCENT3="#a78bfa"
SUCCESS="#34d399"; WARN="#fbbf24"; DANGER="#f87171"
TXT="#e2e8f0"; TXT2="#8892a4"; TXT3="#4a5568"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Figtree:wght@400;500;600;700&display=swap');

/* === BASE === */
html, body {{ background:{BG} !important; color:{TXT} !important; font-family:'Figtree',sans-serif !important; }}
.main, .block-container {{ background:{BG} !important; }}
.block-container {{ padding:1.5rem 2rem 3rem !important; max-width:1400px !important; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.stDeployButton {{ display:none; }}

/* === FORCE WHITE ON EVERYTHING === */
p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6,
[class*="css"], [class*="st-"], [data-testid] {{
    color:{TXT} !important;
}}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {{
    background:{SURF} !important;
    border-right:1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] > div {{
    background:{SURF} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {{
    color:{TXT} !important;
    background:transparent !important;
}}
section[data-testid="stSidebar"] input {{
    background:{SURF2} !important;
    border:1px solid {BORDER} !important;
    color:{TXT} !important;
}}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {{
    background:{SURF} !important;
    border-bottom:1px solid {BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    color:{TXT2} !important;
    background:transparent !important;
    font-size:.82rem !important;
}}
.stTabs [aria-selected="true"] {{
    color:{TXT} !important;
    background:{SURF2} !important;
    border-bottom:2px solid {ACCENT3} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background:{BG} !important;
}}

/* === BUTTONS === */
.stButton button {{
    background:{SURF2} !important;
    border:1px solid {BORDER} !important;
    color:{TXT} !important;
    border-radius:8px !important;
    font-size:.78rem !important;
}}
.stButton button:hover {{
    background:{ACCENT}33 !important;
    border-color:{ACCENT} !important;
    color:{TXT} !important;
}}
[data-testid="baseButton-primary"] {{
    background:{ACCENT} !important;
    border:none !important;
    color:#ffffff !important;
}}

/* === INPUTS === */
.stTextInput input, .stChatInput textarea {{
    background:{SURF2} !important;
    border:1px solid {BORDER} !important;
    color:{TXT} !important;
    border-radius:8px !important;
}}
.stTextInput label, .stSelectbox label, .stTextArea label {{
    color:{TXT2} !important;
    font-size:.72rem !important;
}}
.stChatInput textarea::placeholder {{ color:{TXT3} !important; }}

/* === SELECTBOX === */
[data-baseweb="select"] > div {{
    background:{SURF2} !important;
    border-color:{BORDER} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="select"] div {{
    color:{TXT} !important;
    background:transparent !important;
}}
[data-baseweb="popover"] {{
    background:{SURF2} !important;
}}
[data-baseweb="popover"] li {{
    background:{SURF2} !important;
    color:{TXT} !important;
}}
[data-baseweb="popover"] li:hover {{
    background:{ACCENT}33 !important;
}}
[role="listbox"] > li {{ color:{TXT} !important; background:{SURF2} !important; }}

/* === ALERTS === */
[data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentInfo"],
[data-testid="stNotificationContentSuccess"],
[data-testid="stNotificationContentError"] {{
    color:{TXT} !important;
}}
.stAlert {{ background:{SURF2} !important; border:1px solid {BORDER} !important; }}
.stAlert * {{ color:{TXT} !important; }}

/* === CODE === */
pre, code, .stCodeBlock {{ background:{SURF2} !important; color:{TXT} !important; border:1px solid {BORDER} !important; border-radius:8px !important; }}

/* === EXPANDER === */
.streamlit-expanderHeader {{ background:{SURF2} !important; color:{TXT} !important; border:1px solid {BORDER} !important; }}
.streamlit-expanderContent {{ background:{SURF} !important; border:1px solid {BORDER} !important; }}

/* === SPINNER / DIVIDER === */
.stSpinner > div {{ border-top-color:{ACCENT3} !important; }}
hr {{ border-color:{BORDER} !important; opacity:1 !important; }}

/* === CUSTOM CLASSES === */
.kpi-card {{background:{SURF};border:1px solid {BORDER};border-top:2px solid var(--ac,{ACCENT});border-radius:10px;padding:14px 18px;margin-bottom:10px;}}
.kpi-label {{font-size:.62rem !important;font-weight:600;color:{TXT2} !important;text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;}}
.kpi-value {{font-size:1.5rem;font-weight:700;font-family:'IBM Plex Mono',monospace;line-height:1;color:{TXT} !important;}}
.kpi-sub {{font-size:.68rem !important;color:{TXT3} !important;margin-top:4px;font-family:'IBM Plex Mono',monospace;}}
.msg-user {{background:{SURF2};border:1px solid {BORDER};border-radius:10px 10px 2px 10px;padding:12px 16px;margin:8px 0;}}
.msg-agent {{background:{SURF};border:1px solid {BORDER};border-left:3px solid {ACCENT3};border-radius:2px 10px 10px 10px;padding:12px 16px;margin:8px 0;}}
.msg-err {{background:{SURF};border:1px solid {DANGER}44;border-left:3px solid {DANGER};border-radius:2px 10px 10px 10px;padding:12px 16px;margin:8px 0;}}
.msg-meta {{font-size:.62rem;color:{TXT3} !important;font-family:'IBM Plex Mono',monospace;margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;}}
.msg-user *, .msg-agent *, .msg-err *, .msg-meta * {{ color:{TXT} !important; }}
.pill {{font-size:.58rem;font-weight:600;font-family:'IBM Plex Mono',monospace;padding:2px 9px;border-radius:20px;letter-spacing:.5px;text-transform:uppercase;}}
.pp {{background:{ACCENT3}18;color:{ACCENT3} !important;border:1px solid {ACCENT3}35;}}
.pb {{background:{ACCENT}18;color:{ACCENT} !important;border:1px solid {ACCENT}35;}}
.pc {{background:{ACCENT2}18;color:{ACCENT2} !important;border:1px solid {ACCENT2}35;}}
.pg {{background:{SUCCESS}18;color:{SUCCESS} !important;border:1px solid {SUCCESS}35;}}
.pw {{background:{WARN}18;color:{WARN} !important;border:1px solid {WARN}35;}}
.layer {{display:inline-block;padding:2px 9px;border-radius:4px;font-size:.6rem;font-family:'IBM Plex Mono',monospace;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-right:4px;}}
.br {{background:#cd7f3218;color:#cd7f32 !important;border:1px solid #cd7f3240;}}
.sv {{background:#c0c0c018;color:#c0c0c0 !important;border:1px solid #c0c0c040;}}
.gd {{background:{WARN}18;color:{WARN} !important;border:1px solid {WARN}40;}}
</style>""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
sp_st = "🟢 Spark" if SPARK_OK else "🟡 Estático"
ll_st = "🟢 LLMs" if LANGCHAIN_OK else "🔴 Sem LLMs"
st.markdown(f"""
<div style="background:linear-gradient(135deg,{SURF2},{SURF});border:1px solid {BORDER};
            border-top:2px solid {ACCENT3};border-radius:12px;padding:18px 26px;
            margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:500;color:{TXT};">
      ◈ E-Commerce AI Agent
      <span style="font-size:.56rem;background:{ACCENT3}22;color:{ACCENT3};border:1px solid {ACCENT3}44;
                   border-radius:4px;padding:1px 7px;letter-spacing:1.5px;margin-left:10px;vertical-align:middle;">
        PARTE 4 · DATABRICKS</span>
    </div>
    <div style="font-size:.7rem;color:{TXT3};margin-top:4px;font-family:'IBM Plex Mono',monospace;">
      Medallion · Delta Lake · Spark SQL · MLflow · Olist + IBGE</div>
    <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;">
      <span class="pill pp">LangGraph</span><span class="pill pb">Claude Sonnet</span>
      <span class="pill pc">Llama 3.3 70B</span><span class="pill pg">Delta Lake</span>
      <span class="pill pw">IBGE</span>
      <span class="pill" style="background:#f472b618;color:#f472b6;border:1px solid #f472b635;">Mistral</span>
      <span class="pill" style="background:#fb923c18;color:#fb923c;border:1px solid #fb923c35;">Qwen</span>
    </div>
  </div>
  <div style="text-align:right;font-size:.68rem;color:{TXT3};font-family:'IBM Plex Mono',monospace;">
    <div style="color:{TXT2};">Rafael Reghine Munhoz</div>
    <div style="margin-top:2px;">MBA USP ESALQ · Data Science</div>
    <div style="margin-top:6px;font-size:.6rem;">{sp_st} &nbsp;·&nbsp; {ll_st}</div>
    <div style="margin-top:4px;">
      <a href="https://linkedin.com/in/rafaelreghine" style="color:{ACCENT2};text-decoration:none;">linkedin</a>
      &nbsp;·&nbsp;
      <a href="https://github.com/rreghine" style="color:{ACCENT2};text-decoration:none;">github</a>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:.68rem;color:{TXT2};border-bottom:1px solid {BORDER};padding-bottom:10px;margin-bottom:14px;">STATUS DO SISTEMA</div>', unsafe_allow_html=True)

    groq_ok      = bool(_get_secret("GROQ_API_KEY"))
    anthropic_ok = bool(_get_secret("ANTHROPIC_API_KEY"))
    openai_ok    = bool(_get_secret("OPENAI_API_KEY"))
    llms_ok      = groq_ok or anthropic_ok or openai_ok

    # Status Gold Layer
    c = ACCENT3 if SPARK_OK else WARN
    sub = "Unity Catalog ativo" if SPARK_OK else "27 estados (dados estaticos)"
    st.markdown(f'<div class="kpi-card" style="--ac:{c};padding:10px 14px;margin-bottom:8px;"><div class="kpi-label">Gold Layer</div><div style="font-size:.75rem;color:{c};">{"[OK] Spark conectado" if SPARK_OK else "[OK] Modo estatico"}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    # Status LLMs
    c2 = SUCCESS if llms_ok else WARN
    llm_sub = []
    if groq_ok:      llm_sub.append("Groq: Llama 8B+70B")
    if anthropic_ok: llm_sub.append("Claude Sonnet")
    if openai_ok:    llm_sub.append("GPT-4o")
    if not llms_ok:  llm_sub.append("Configure as keys abaixo")
    st.markdown(f'<div class="kpi-card" style="--ac:{c2};padding:10px 14px;margin-bottom:8px;"><div class="kpi-label">LLMs ({len([x for x in [groq_ok,anthropic_ok,openai_ok] if x])}/3)</div><div style="font-size:.75rem;color:{c2};">{"[OK] Prontos" if llms_ok else "[!] Nao configurado"}</div><div class="kpi-sub">{" | ".join(llm_sub)}</div></div>', unsafe_allow_html=True)

    # Input de API Keys
    st.divider()
    st.markdown(f'<div style="font-size:.63rem;color:{TXT2};font-family:IBM Plex Mono,monospace;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;">API Keys</div>', unsafe_allow_html=True)

    gk = st.text_input("GROQ_API_KEY", type="password",
                       placeholder="gsk_... (gratis: console.groq.com)",
                       value="***" if groq_ok else "")
    ok_inp = st.text_input("OPENAI_API_KEY", type="password",
                       placeholder="sk-... (platform.openai.com)",
                       value="***" if openai_ok else "")
    ak = st.text_input("ANTHROPIC_API_KEY", type="password",
                       placeholder="sk-ant-... (console.anthropic.com)",
                       value="***" if anthropic_ok else "")
    gm = st.text_input("GEMINI_API_KEY", type="password",
                       placeholder="AIza... (aistudio.google.com)",
                       value="***" if bool(_get_secret("GEMINI_API_KEY")) else "")

    if st.button("Aplicar Keys", type="primary", use_container_width=True):
        if gk     and gk     != "***": os.environ["GROQ_API_KEY"]      = gk
        if ok_inp and ok_inp != "***": os.environ["OPENAI_API_KEY"]    = ok_inp
        if ak     and ak     != "***": os.environ["ANTHROPIC_API_KEY"] = ak
        if gm     and gm     != "***": os.environ["GEMINI_API_KEY"]    = gm
        st.success("Keys salvas! Recarregue.")
        st.rerun()

    # Seletor de modelo
    st.divider()
    available = get_available_models()
    st.markdown(f'<div style="font-size:.63rem;color:{TXT2};font-family:IBM Plex Mono,monospace;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;">Modelo do Agente</div>', unsafe_allow_html=True)
    if available:
        modelo_sel = st.selectbox(
            "Escolha o modelo",
            ["Auto (por complexidade)"] + available,
            key="modelo_selecionado",
            label_visibility="collapsed"
        )
    else:
        st.markdown(f'<div style="font-size:.72rem;color:{WARN};">Configure as keys acima</div>', unsafe_allow_html=True)
        modelo_sel = "Auto (por complexidade)"
    st.session_state["model_override"] = None if modelo_sel == "Auto (por complexidade)" else modelo_sel

    st.divider()
    st.markdown(f"""
    <div style="font-size:.68rem;line-height:2.2;color:{TXT2};">
      <span class="layer br">Bronze</span> 7 tabelas Olist + IBGE<br>
      <span class="layer sv">Silver</span> Joins + flags + limpeza<br>
      <span class="layer gd">Gold</span> 27 estados - Olist x IBGE<br>
    </div>
    <div style="margin-top:12px;font-size:.66rem;line-height:2.2;color:{TXT2};font-family:IBM Plex Mono,monospace;">
      simple  -> Llama 3.1 8B (Groq)<br>
      medium  -> Llama 3.3 70B (Groq)<br>
      complex -> Claude Sonnet<br>
      <span style="color:{TXT3};">override -> Mistral / Qwen</span><br>
    </div>""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Agente SQL", "📊 Dashboard", "🏁 Benchmark", "🏗️ Arquitetura"])

# ══ TAB 1 - AGENTE SQL ═══════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div style="font-size:.68rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:14px;">SQL Agent · Gold Layer · ecommerce_ai.gold_performance_estados_ibge · Olist + IBGE</div>', unsafe_allow_html=True)

    SUGESTOES = [
        "Qual região tem maior taxa de atraso?",
        "Estados de baixa renda parcelam mais?",
        "Qual a relação entre PIB e satisfação?",
        "Top 3 estados mais eficientes em entrega",
        "Compare ticket médio por classe econômica",
    ]
    st.markdown(f'<div style="font-size:.6rem;color:{TXT3};margin-bottom:6px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">Sugestões</div>', unsafe_allow_html=True)
    cols_s = st.columns(len(SUGESTOES))
    btn_p  = ""
    for col, s in zip(cols_s, SUGESTOES):
        with col:
            if st.button(s, key=f"btn_{s[:12]}", use_container_width=True): btn_p = s

    if "chat" not in st.session_state: st.session_state.chat = []
    p_input = st.chat_input("Faça uma pergunta sobre o e-commerce brasileiro...")
    p_final = btn_p or p_input

    if p_final:
        st.session_state.chat.append({"role":"user","content":p_final})
        model_override = st.session_state.get("model_override")
        with st.spinner(f"Consultando Gold Layer{' com ' + model_override if model_override else ''}..."):
            result = run_agent(p_final, model_override)
        st.session_state.chat.append({"role":"agent","result":result})

    for msg in reversed(st.session_state.chat):
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user"><span style="font-size:.6rem;color:{TXT3};font-family:IBM Plex Mono,monospace;">👤 VOCÊ</span><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            r = msg["result"]
            if r.get("guardrail") or r.get("erro"):
                txt = r.get("resposta") or r.get("erro","Erro")
                st.markdown(f'<div class="msg-err"><span style="font-size:.6rem;color:{DANGER};font-family:IBM Plex Mono,monospace;">BLOQUEADO</span><br>{txt[:200]}</div>', unsafe_allow_html=True)
            else:
                custo_s = f"${r['custo_usd']:.6f}" if r.get("custo_usd",0)>0 else "Gratuito"
                st.markdown(f"""<div class="msg-agent">
                  <span style="font-size:.6rem;color:{ACCENT3};font-family:IBM Plex Mono,monospace;">AGENTE</span><br>
                  {r["resposta"]}
                  <div class="msg-meta">
                    <span>🤖 {r.get("llm","-")}</span>
                    <span>⚡ {r.get("complexity","-")}</span>
                    <span>⏱️ {r.get("latencia_ms","-")}ms</span>
                    <span>💰 {custo_s}</span>
                  </div></div>""", unsafe_allow_html=True)

    if st.session_state.chat:
        if st.button("🗑️ Limpar"): st.session_state.chat = []; st.rerun()

# ══ TAB 2 - DASHBOARD ════════════════════════════════════════════════════════
with tab2:
    fonte = "🟢 Databricks · Unity Catalog" if SPARK_OK else "🟡 Dados do notebook Databricks (estáticos)"
    st.markdown(f'<div style="font-size:.68rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:4px;">Gold Layer · 27 estados · Olist E-Commerce + IBGE PIB per Capita 2018</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.6rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:14px;">Fonte: {fonte}</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=300)
    def load_gold(): return get_gold_data()
    df = load_gold()

    k1,k2,k3,k4 = st.columns(4)
    for col,color,label,val,sub in [
        (k1,ACCENT3,"Total Pedidos",   f"{int(df['total_pedidos'].sum()):,}","pedidos entregues"),
        (k2,ACCENT, "Ticket Médio BR", f"R${df['ticket_medio'].mean():.2f}", "média nacional"),
        (k3,DANGER, "Taxa Atraso",     f"{df['pct_atrasados'].mean():.1f}%", "média nacional"),
        (k4,SUCCESS,"Satisfação",      f"{df['satisfacao_media'].mean():.2f}/5","nota média"),
    ]:
        with col: st.markdown(f'<div class="kpi-card" style="--ac:{color}"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.divider()
    fig, axes = plt.subplots(2,2,figsize=(14,8)); fig.patch.set_facecolor(SURF)
    for ax in axes.flat:
        ax.set_facecolor(SURF2); ax.tick_params(colors=TXT2,labelsize=8)
        for sp in ax.spines.values(): sp.set_color(BORDER)

    def ttl(ax,t): ax.set_title(t,color=TXT,fontsize=10,fontweight="bold",pad=10)
    cmap_b = LinearSegmentedColormap.from_list("b",[ACCENT+"44",ACCENT])

    top7 = df.nlargest(7,"total_pedidos")
    cb   = [cmap_b(v) for v in np.linspace(.3,1.,7)]
    bars = axes[0,0].barh(top7["customer_state"],top7["total_pedidos"],color=cb[::-1],height=.6)
    for b,v in zip(bars,top7["total_pedidos"]):
        axes[0,0].text(b.get_width()+top7["total_pedidos"].max()*.02,b.get_y()+b.get_height()/2,f"{int(v):,}",va="center",fontsize=8,color=TXT)
    axes[0,0].set_xlim(0,top7["total_pedidos"].max()*1.25); ttl(axes[0,0],"Top 7 Estados - Pedidos")

    sc_c = [ACCENT if c=="Rico" else ACCENT3 if c=="Médio" else DANGER for c in df["classe_economica"]]
    axes[0,1].scatter(df["pib_per_capita"],df["ticket_medio"],c=sc_c,s=70,alpha=.85,edgecolors=BORDER,linewidth=.5)
    for _,row in df.iterrows():
        axes[0,1].annotate(row["customer_state"],(row["pib_per_capita"],row["ticket_medio"]),fontsize=6,color=TXT3,xytext=(3,3),textcoords="offset points")
    axes[0,1].set_xlabel("PIB per capita R$ (IBGE)",color=TXT2,fontsize=8); axes[0,1].set_ylabel("Ticket Médio R$",color=TXT2,fontsize=8)
    axes[0,1].legend(handles=[Patch(color=ACCENT,label="Rico"),Patch(color=ACCENT3,label="Médio"),Patch(color=DANGER,label="Baixa Renda")],fontsize=7,labelcolor=TXT2,facecolor=SURF)
    ttl(axes[0,1],"PIB per Capita × Ticket Médio · Olist + IBGE")

    ordem = ["Baixa Renda","Médio","Rico"]
    bc    = df.groupby("classe_economica")[["media_parcelas","pct_parcelados"]].mean().reindex(ordem)
    x     = np.arange(len(bc))
    axes[1,0].bar(x-.2,bc["media_parcelas"],.35,color=ACCENT,label="Média Parcelas")
    axes[1,0].bar(x+.2,bc["pct_parcelados"]/10,.35,color=ACCENT3,label="% Parcelados÷10")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(ordem,color=TXT2,fontsize=8)
    axes[1,0].legend(fontsize=7,labelcolor=TXT2,facecolor=SURF); ttl(axes[1,0],"Parcelamento por Classe Econômica")

    br  = df.groupby("regiao")["pct_atrasados"].mean().sort_values(ascending=False)
    bc2 = [DANGER if v>12 else WARN if v>8 else SUCCESS for v in br.values]
    b4  = axes[1,1].bar(br.index,br.values,color=bc2,width=.6)
    for b,v in zip(b4,br.values):
        axes[1,1].text(b.get_x()+b.get_width()/2,b.get_height()+.3,f"{v:.1f}%",ha="center",fontsize=8,color=TXT)
    axes[1,1].set_ylabel("Taxa Atraso %",color=TXT2,fontsize=8); ttl(axes[1,1],"Taxa de Atraso por Região")

    fig.text(.5,-.01,"Rafael Reghine Munhoz · github.com/rreghine · MBA USP ESALQ · Dados: Olist + IBGE 2018",ha="center",fontsize=7.5,color=TXT3)
    plt.tight_layout(); st.pyplot(fig)

    st.divider()
    st.markdown(f'<div style="font-size:.6rem;color:{TXT3};margin-bottom:8px;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:1px;">Tabela Gold Completa - 27 estados</div>', unsafe_allow_html=True)

    # Renderiza como HTML para garantir tema escuro
    cols_show = ["customer_state","regiao","classe_economica","pib_per_capita",
                 "total_pedidos","ticket_medio","pct_parcelados","satisfacao_media","pct_atrasados","ticket_por_pib"]
    df_show = df[[c for c in cols_show if c in df.columns]].copy()
    df_show.columns = ["Estado","Região","Classe","PIB p/capita","Pedidos",
                       "Ticket Médio","% Parcelados","Satisfação","% Atraso","Ticket/PIB"]

    def color_row(row):
        if row.get("Classe") == "Rico": return ACCENT+"22"
        if row.get("Classe") == "Médio": return ACCENT3+"22"
        return DANGER+"11"

    headers_html = "".join([f'<th style="padding:6px 10px;background:{SURF2};color:#e2e8f0;font-size:.62rem;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid {BORDER};font-weight:600;white-space:nowrap;">{c}</th>' for c in df_show.columns])

    rows_html = ""
    for _, row in df_show.iterrows():
        bg = color_row(row.to_dict())
        cells = "".join([f'<td style="padding:6px 10px;border-bottom:1px solid {BORDER};font-size:.75rem;color:#e2e8f0;white-space:nowrap;">{v if not isinstance(v,float) else f"{v:.1f}"}</td>' for v in row])
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid {BORDER};border-radius:8px;margin-top:8px;">
      <table style="width:100%;border-collapse:collapse;background:{SURF};">
        <thead><tr>{headers_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

# ══ TAB 3 - BENCHMARK ════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div style="font-size:.68rem;color:{TXT3};font-family:IBM Plex Mono,monospace;margin-bottom:14px;">Benchmark · cada LLM responde as mesmas perguntas · custo x qualidade x latência</div>', unsafe_allow_html=True)

    BENCH_Q = [
        "Qual região tem maior taxa de atraso?",
        "Estados de baixa renda parcelam mais?",
        "Qual estado tem maior ticket médio?",
    ]

    has_keys = bool(_get_secret("GROQ_API_KEY") or _get_secret("ANTHROPIC_API_KEY") or _get_secret("OPENAI_API_KEY"))

    if not LANGCHAIN_OK or not has_keys:
        st.warning("Configure as API Keys na sidebar para rodar o benchmark.")
    else:
        if st.button("▶️ Rodar Benchmark Multi-LLM", type="primary"):
            # Monta lista de LLMs disponíveis
            gk = _get_secret("GROQ_API_KEY")
            ak = _get_secret("ANTHROPIC_API_KEY")
            ok = _get_secret("OPENAI_API_KEY")

            llms_bench = []
            if gk:
                llms_bench.append(("llama-3.1-8b-instant",    ChatGroq(model="llama-3.1-8b-instant",    api_key=gk), "Gratuito"))
                llms_bench.append(("llama-3.3-70b-versatile", ChatGroq(model="llama-3.3-70b-versatile", api_key=gk), "Gratuito"))
                llms_bench.append(("mistral-saba-24b",        ChatGroq(model="mistral-saba-24b",        api_key=gk), "Gratuito"))
                llms_bench.append(("qwen-qwq-32b",            ChatGroq(model="qwen-qwq-32b",            api_key=gk), "Gratuito"))
            if ak:
                llms_bench.append(("claude-sonnet-4-6", ChatAnthropic(model="claude-sonnet-4-6", api_key=ak, max_tokens=512), "$3/1M"))
            if ok and OPENAI_OK:
                llms_bench.append(("gpt-4o-mini", ChatOpenAI(model="gpt-4o-mini", api_key=ok, max_tokens=512), "$0.15/1M"))
                llms_bench.append(("gpt-4o",      ChatOpenAI(model="gpt-4o",      api_key=ok, max_tokens=512), "$2.5/1M"))

            res_b = {}
            prog  = st.progress(0)
            total = len(BENCH_Q) * len(llms_bench)
            idx   = 0

            for q in BENCH_Q:
                res_b[q] = {}
                for llm_name, llm_obj, preco in llms_bench:
                    t0 = time.time()
                    try:
                        # Gera SQL
                        r1 = llm_obj.invoke([HumanMessage(content=f"Especialista Spark SQL. Retorne APENAS SQL sem markdown.\nSchema:\n{DB_SCHEMA}\nPergunta: {q}\nSQL:")])
                        sql = re.sub(r"```sql|```","",r1.content).strip()
                        # Executa
                        if SPARK_OK and _spark:
                            rows = [row.asDict() for row in _spark.sql(sql).limit(10).collect()]
                        else:
                            rows = get_gold_data().head(5).to_dict("records")
                        # Interpreta
                        r2 = llm_obj.invoke([HumanMessage(content=f"Analista brasileiro. Interprete em 1 linha. Use numeros.\nPergunta:{q}\nDados:{pd.DataFrame(rows).to_string(index=False)}\nResposta:")])
                        resposta = r2.content.strip()
                        usage = getattr(r1,"usage_metadata",None)
                        ti = getattr(usage,"input_tokens",300)
                        to = getattr(usage,"output_tokens",100)
                        costs = LLM_COSTS.get(llm_name,{"input":0,"output":0})
                        custo = (ti/1e6*costs["input"])+(to/1e6*costs["output"])
                        res_b[q][llm_name] = {"resposta":resposta,"sql":sql,"latencia":int((time.time()-t0)*1000),"custo":round(custo,6),"preco":preco,"ok":True}
                    except Exception as e:
                        res_b[q][llm_name] = {"resposta":f"Erro: {str(e)[:80]}","sql":"","latencia":0,"custo":0,"preco":preco,"ok":False}
                    idx += 1
                    prog.progress(idx/total)

            st.session_state["bench"] = res_b
            prog.empty()

    if st.session_state.get("bench"):
        res = st.session_state["bench"]
        all_llms = list(list(res.values())[0].keys()) if res else []

        # Métricas por LLM
        st.markdown(f'<div style="font-size:.65rem;color:{TXT2};font-family:IBM Plex Mono,monospace;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px;">Comparativo por LLM</div>', unsafe_allow_html=True)
        cols_llm = st.columns(len(all_llms)) if all_llms else []
        for col, llm_n in zip(cols_llm, all_llms):
            runs   = [v[llm_n] for v in res.values() if llm_n in v]
            t_lat  = sum(r["latencia"] for r in runs)
            t_cost = sum(r["custo"]    for r in runs)
            ok_ct  = sum(1 for r in runs if r["ok"])
            preco  = runs[0]["preco"] if runs else "-"
            color  = ACCENT if "claude" in llm_n else ACCENT2 if "gpt" in llm_n else ACCENT3 if "mistral" in llm_n else WARN if "qwen" in llm_n else SUCCESS
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="--ac:{color}">
                  <div class="kpi-label" style="font-size:.55rem">{llm_n}</div>
                  <div style="font-size:.9rem;font-weight:700;color:{color};font-family:IBM Plex Mono,monospace">{t_lat:,}ms</div>
                  <div class="kpi-sub">${t_cost:.6f} | {preco}<br>{ok_ct}/{len(runs)} OK</div>
                </div>""", unsafe_allow_html=True)

        st.divider()

        # Respostas por pergunta
        for q, llm_results in res.items():
            st.markdown(f'<div style="font-size:.8rem;font-weight:600;color:{TXT};margin:16px 0 8px;padding:8px 12px;background:{SURF2};border-left:3px solid {ACCENT3};border-radius:0 6px 6px 0;">❓ {q}</div>', unsafe_allow_html=True)
            cols_r = st.columns(len(llm_results))
            for col, (llm_n, r) in zip(cols_r, llm_results.items()):
                color = ACCENT if "claude" in llm_n else ACCENT2 if "gpt" in llm_n else ACCENT3 if "mistral" in llm_n else WARN if "qwen" in llm_n else SUCCESS
                custo_s = f"${r['custo']:.6f}" if r["custo"]>0 else "Gratis"
                with col:
                    st.markdown(f"""
                    <div style="background:{SURF};border:1px solid {BORDER};border-top:2px solid {color};
                                border-radius:8px;padding:12px;min-height:120px;">
                      <div style="font-size:.6rem;color:{color};font-family:IBM Plex Mono,monospace;
                                  font-weight:600;margin-bottom:8px;">{llm_n}</div>
                      <div style="font-size:.78rem;color:{TXT if r['ok'] else DANGER};line-height:1.5;">
                        {r['resposta'][:180]}
                      </div>
                      <div style="font-size:.6rem;color:{TXT3};margin-top:8px;font-family:IBM Plex Mono,monospace;">
                        ⏱️ {r['latencia']}ms &nbsp; 💰 {custo_s}
                      </div>
                    </div>""", unsafe_allow_html=True)

# ══ TAB 4 - ARQUITETURA ══════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BORDER};border-radius:12px;padding:24px;
                font-family:'IBM Plex Mono',monospace;font-size:.76rem;line-height:2.2;color:{TXT2};">
      <div style="color:{ACCENT3};font-weight:600;margin-bottom:14px;font-size:.88rem;">Jornada das 4 Partes</div>
      <span style="color:{TXT3}">Parte 1</span> -> <span style="color:{ACCENT2}">RAG Agent</span> · FAISS · Embeddings · CSV Olist<br>
      <span style="color:{TXT3}">Parte 2</span> -> <span style="color:{ACCENT2}">SQL Agent</span> · Text-to-SQL · SQLite · LLM-as-Judge<br>
      <span style="color:{TXT3}">Parte 3</span> -> <span style="color:{ACCENT3}">Orquestrador</span> · LangGraph · Multi-LLM · MLflow · Streamlit<br>
      <span style="color:{TXT3}">Parte 4</span> -> <span style="color:{WARN}">Produção Databricks</span> · Medallion · Delta Lake · Unity Catalog<br>
      <div style="color:{ACCENT3};font-weight:600;margin:18px 0 10px;font-size:.88rem;">Arquitetura Medalão - Databricks Workspace</div>
      <span class="layer br">Bronze</span> 7 CSVs Olist (99k pedidos) + IBGE -> Delta Lake raw<br>
      <span class="layer sv">Silver</span> Joins, tipagem, flags de atraso (96.4k pedidos entregues)<br>
      <span class="layer gd">Gold</span> <b>ecommerce_ai.gold_performance_estados_ibge</b> · 27 estados<br>
      <div style="color:{ACCENT3};font-weight:600;margin:18px 0 10px;font-size:.88rem;">Multi-LLM Routing (cost-aware)</div>
      <span style="color:{SUCCESS}">simple  (≤40 palavras)</span> -> Llama 3.1 8B  · Groq · gratuito · ~900ms<br>
      <span style="color:{WARN}">medium  (41-80 palavras)</span> -> Llama 3.3 70B · Groq · gratuito · ~1.5s<br>
      <span style="color:{ACCENT}">complex (>80 palavras) </span> -> Claude Sonnet · Anthropic · $3/1M tokens<br>
      <span style="color:#f472b6">override: Mistral Saba 24B</span> → Groq · gratuito · rápido em queries simples<br>
      <span style="color:#fb923c">override: Qwen QwQ 32B  </span> → Groq · gratuito · ótimo custo-benefício<br>
      <div style="color:{ACCENT3};font-weight:600;margin:18px 0 10px;font-size:.88rem;">Insights - Gold Layer Olist × IBGE</div>
      <span style="color:{DANGER}">▲</span> Nordeste: 15.2% de atraso vs Sul: 7.3% - reflexo do PIB<br>
      <span style="color:{WARN}">▲</span> Baixa Renda: 59.6% parcelam (3.43x) vs Rico: 50.9% (2.94x)<br>
      <span style="color:{SUCCESS}">▲</span> Paradoxo: estados pobres têm ticket_por_pib 3.6x maior<br>
      <span style="color:{ACCENT2}">▲</span> PR, MG, SP - campeões de eficiência logística<br>
      <div style="margin-top:18px;font-size:.65rem;color:{TXT3};">
        Rafael Reghine Munhoz · MBA USP ESALQ · github.com/rreghine · linkedin.com/in/rafaelreghine
      </div>
    </div>""", unsafe_allow_html=True)
