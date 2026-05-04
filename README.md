# E-Commerce AI Agent — Databricks Edition (Parte 4)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Databricks](https://img.shields.io/badge/Databricks-Medallion_Architecture-red?style=flat-square)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Bronze→Silver→Gold-orange?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-purple?style=flat-square)
![Claude](https://img.shields.io/badge/Anthropic-Claude_Sonnet-black?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.x-green?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

Agente de IA conversacional sobre dados de e-commerce brasileiro (Olist) cruzados com o PIB
per capita estadual do IBGE. Executa sobre Arquitetura Medallion no Databricks — Bronze → Silver
→ Gold em Delta Lake — com Spark SQL direto no Unity Catalog. Multi-LLM routing automático por
complexidade da pergunta: Llama 3.1 8B e Llama 3.3 70B (gratuitos via Groq) para consultas
simples e medianas, Claude Sonnet para consultas complexas. Guardrails integrados, MLflow
nativo e benchmark ao vivo com todos os modelos disponíveis.

---

## Evolucoes em Relacao as Partes Anteriores

| Componente | Parte 1 | Parte 2 | Parte 3 | Parte 4 |
|---|---|---|---|---|
| Dados | CSV + RAG + FAISS | SQLite + Text-to-SQL | SQLite | Delta Lake — Unity Catalog |
| Infraestrutura | Colab | Colab | Colab | Databricks Workspace |
| Armazenamento | CSV | SQLite | SQLite | Delta Lake |
| SQL Engine | — | sqlite3 | sqlite3 | Spark SQL |
| Dados extras | — | — | — | IBGE PIB per Capita |
| Gold Layer | — | — | — | 27 estados · Olist × IBGE |
| LLM principal | Google Gemma 3 | Anthropic Claude Sonnet | Llama 3.3 70B | Multi-LLM routing |
| Orquestrador | LangChain | LangChain | LangGraph | LangGraph |
| Observabilidade | MLflow | MLflow | MLflow | MLflow nativo Databricks |
| Abas | 2 | 4 | 4 | 4 |

---

## Arquitetura

```
Pergunta do usuario
        |
Guardrails — valida escopo, DDL, dados sensiveis e tamanho
        |
Classificador de Complexidade
    |-- simple  (<=40 palavras) -> Llama 3.1 8B  · Groq · gratis
    |-- medium  (41-80 palavras) -> Llama 3.3 70B · Groq · gratis
    |-- complex (>80 palavras)  -> Claude Sonnet  · Anthropic · $3/1M
        |
Text-to-SQL — LLM converte pergunta em Spark SQL
        |
Spark SQL — executa sobre Unity Catalog (com fallback estatico)
    ecommerce_ai.gold_performance_estados_ibge
        |
LLM — interpreta resultado em portugues (max 2 linhas)
        |
MLflow — loga tokens, custo, latencia, complexidade, attempts
        |
Streamlit — 4 abas
    |-- Agente SQL: chat com metadata de execucao
    |-- Dashboard: KPIs e graficos Gold Layer
    |-- Benchmark: todos os LLMs, mesma pergunta, ao vivo
    |-- Arquitetura: jornada das 4 partes + insights Gold Layer
```

---

## Arquitetura Medallion — Databricks Workspace

```
Bronze Layer
    7 CSVs Olist (99.441 pedidos) + IBGE PIB per Capita
    -> Delta Lake raw · sem transformacoes
        |
Silver Layer
    Joins entre tabelas Olist + IBGE
    Tipagem, limpeza, flags de atraso
    96.4k pedidos entregues
        |
Gold Layer
    ecommerce_ai.gold_performance_estados_ibge
    27 estados · 1 linha por estado
    Metricas: pedidos, ticket, parcelas, satisfacao, atraso, PIB, ticket/PIB
```

---

## Dataset

**Brazilian E-Commerce (Kaggle) + IBGE PIB per Capita 2018**

- 99.441 pedidos reais e anonimizados (Olist)
- 7 tabelas relacionadas — pedidos, clientes, produtos, vendedores, pagamentos, avaliacoes
- PIB per capita por estado — IBGE 2018
- Periodo: 2016 a 2018
- Contexto 100% brasileiro — 27 estados

---

## Estrutura do Repositorio

```
ai-agent-databricks/
|
|-- app.py                  # App principal Streamlit — Databricks Apps
|-- app.yaml                # Configuracao Databricks Apps
|-- requirements.txt        # Dependencias Python
|-- streamlit/              # Assets e configuracoes Streamlit
|-- README.md
```

---

## Componentes

### Gold Layer — ecommerce_ai.gold_performance_estados_ibge

Tabela Delta criada no notebook Databricks, disponivel no Unity Catalog:

| Coluna | Descricao |
|---|---|
| customer_state | Sigla do estado |
| regiao | Regiao brasileira |
| pib_per_capita | PIB per capita IBGE 2018 (R$) |
| classe_economica | Rico / Medio / Baixa Renda |
| total_pedidos | Total de pedidos entregues |
| ticket_medio | Ticket medio em reais |
| media_parcelas | Media de parcelas por pedido |
| pct_parcelados | % de pedidos parcelados |
| satisfacao_media | Nota media (0–5) |
| pct_atrasados | % de pedidos com atraso |
| media_dias_atraso | Media de dias de atraso (negativo = adiantado) |
| ticket_por_pib | Ticket medio / PIB per capita × 1000 |

O app tenta conectar via Spark SQL ao Unity Catalog. Se nao estiver no Databricks, usa os dados
estaticos embutidos (espelho exato da tabela Gold).

### Multi-LLM Routing por Complexidade

```
simple  (<=40 palavras) -> Llama 3.1 8B  · Groq · ~900ms  · Gratis
medium  (41-80 palavras) -> Llama 3.3 70B · Groq · ~1.5s   · Gratis
complex (>80 palavras)  -> Claude Sonnet  · Anthropic · $3/1M tokens
```

O usuario pode sobrescrever o routing automatico e escolher o modelo manualmente na sidebar.

### Guardrails

Validacao em dois niveis — entrada:

- Bloqueio de DDL: DROP, DELETE, TRUNCATE, INSERT, UPDATE, ALTER
- Protecao de dados: CPF, CNPJ, senha, password, token, secret
- Controle de tamanho: minimo 2 palavras

### Text-to-SQL com Retry Automatico

```
Pergunta -> LLM gera Spark SQL -> Spark executa
                |
           Erro de SQL?
                |
         LLM corrige -> Spark reexecuta (max 2 tentativas)
```

### Benchmark ao Vivo

Todos os modelos configurados recebem as mesmas perguntas sobre o mesmo Gold Layer. A unica
variavel e o modelo de linguagem.

**Modelos suportados:**

| Modelo | API | Custo |
|---|---|---|
| Llama 3.1 8B | Groq | Gratuito |
| Llama 3.3 70B | Groq | Gratuito |
| Claude Sonnet 4.6 | Anthropic | $3/1M tokens input |
| GPT-4o Mini | OpenAI | $0.15/1M tokens input |
| GPT-4o | OpenAI | $2.50/1M tokens input |
| Gemini 2.0 Flash | Google | $0.10/1M tokens input |
| Gemini 1.5 Pro | Google | $1.25/1M tokens input |

### MLflow — Observabilidade Nativa Databricks

Cada execucao loga automaticamente:

- Parametros: modelo, complexidade, numero de tentativas SQL
- Metricas: tokens input/output, custo USD, latencia ms
- Tags: tabela Gold, numero da parte

---

## Tecnologias Utilizadas

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.12 |
| Plataforma | Databricks Workspace + Databricks Apps |
| Armazenamento | Delta Lake · Unity Catalog |
| SQL Engine | Spark SQL |
| Orquestracao | LangGraph |
| LLMs | Llama 3.1 8B · Llama 3.3 70B (Groq) · Claude Sonnet (Anthropic) |
| Dados | Olist E-Commerce (Kaggle) + IBGE PIB per Capita 2018 |
| Rastreamento | MLflow (nativo Databricks) |
| Interface | Streamlit (Databricks Apps) |

---

## Autor

**Rafael Reghine Munhoz**  
Data Analyst | Data Science & Analytics | MBA USP ESALQ

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rafaelreghine-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/rafaelreghine)
[![GitHub](https://img.shields.io/badge/GitHub-rreghine-black?style=flat-square&logo=github)](https://github.com/rreghine)
