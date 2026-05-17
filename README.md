# E-Commerce AI Agent — Databricks Edition (Parte 4)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Databricks](https://img.shields.io/badge/Databricks-Free_Edition-red?style=flat-square)
![dbt](https://img.shields.io/badge/dbt-Databricks-orange?style=flat-square)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-Bronze→Silver→Gold-orange?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-purple?style=flat-square)
![Claude](https://img.shields.io/badge/Anthropic-Claude_Sonnet-black?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama·Mistral·Qwen-green?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

Agente de IA conversacional sobre dados de e-commerce brasileiro (Olist) cruzados com o PIB
per capita estadual do IBGE. Executa sobre Arquitetura Medalao no Databricks — Bronze → Silver
→ Gold em Delta Lake — com transformacoes orquestradas pelo dbt e consultas via Spark SQL no
Unity Catalog. Multi-LLM routing automatico por complexidade da pergunta: Llama 3.1 8B e
Llama 3.3 70B (gratuitos via Groq) para consultas simples e medianas, Claude Sonnet para
consultas complexas. Benchmark ao vivo com 5 modelos via Groq + Claude Sonnet comparando
latencia, tokens e custo sobre o mesmo Gold Layer.

---

## Evolucoes em Relacao as Partes Anteriores

| Componente | Parte 1 | Parte 2 | Parte 3 | Parte 4 |
| --- | --- | --- | --- | --- |
| LLM principal | Google Gemma 3 | Anthropic Claude Sonnet | Llama 3.3 70B | Multi-LLM routing |
| Dados | CSV + RAG + FAISS | SQLite + Text-to-SQL | SQLite | Delta Lake — Unity Catalog |
| Transformacao | — | — | — | dbt (Silver + Gold) |
| SQL Engine | — | sqlite3 | sqlite3 | Spark SQL |
| Dados extras | — | — | — | IBGE PIB per Capita |
| Infraestrutura | Colab | Colab | Colab | Databricks Free Edition |
| Orquestrador | LangChain | LangChain | LangGraph | LangGraph |
| Observabilidade | MLflow | MLflow | MLflow | MLflow nativo Databricks |
| Modelos benchmark | 1 | 3 | 3 | 5 — Llama · Mistral · Qwen · Claude |
| Abas | 2 | 4 | 4 | 4 |

---

## Arquitetura

```
CSVs Olist (Kaggle) + IBGE (embutido)
        |
  dbt seed → Bronze (workspace.ecommerce_ai_bronze)
        |
  dbt run  → Silver (workspace.ecommerce_ai_silver)
    |-- silver_pedidos_entregues     join + flags de atraso
    |-- silver_itens_por_pedido      valor total + qtd itens
    |-- silver_pagamentos_por_pedido parcelas + flag parcelado
    |-- silver_reviews_por_pedido    nota media + flag satisfeito
        |
  dbt run  → Gold (workspace.ecommerce_ai_gold)
    |-- gold_performance_estados_ibge  27 estados · Olist x IBGE
    |-- gold_parcelamento_por_renda    analise de credito
    |-- gold_kpis_consolidados         visao executiva
        |
  dbt test → 14 testes · PASS=14 · ERROR=0
        |
Pergunta do usuario
        |
Guardrails — valida escopo, DDL, dados sensiveis
        |
Classificador de Complexidade
    |-- simple  (<=40 palavras) -> Llama 3.1 8B  · Groq · gratis
    |-- medium  (41-80 palavras) -> Llama 3.3 70B · Groq · gratis
    |-- complex (>80 palavras)  -> Claude Sonnet  · Anthropic · $3/1M
        |
Text-to-SQL → Spark SQL sobre Gold Layer
        |
MLflow — loga llm, tokens, custo, latencia por run
        |
Streamlit — 4 abas
    |-- Agente SQL · Dashboard · Benchmark · Arquitetura
```

---

## Dataset

**Brazilian E-Commerce (Kaggle)**

- 99.441 pedidos reais e anonimizados
- 7 tabelas relacionadas — pedidos, clientes, produtos, vendedores, pagamentos, avaliacoes
- Periodo: 2016 a 2018

**IBGE PIB per Capita por UF (SIDRA — Tabela 5938)**

- PIB per capita estadual de 2018 em R$
- 27 estados — embutido no projeto como dbt seed

---

## Estrutura do Repositorio

```
ai-agent-databricks/
|
|-- dbt_ecommerce/                     
|   |-- dbt_project.yml
|   |-- profiles.yml
|   |-- seeds/
|   |   |-- ibge_pib_estados.csv       
|   |   |-- schema.yml
|   |   |-- olist_*.csv                
|   |-- models/
|   |   |-- silver/                    
|   |   |-- gold/                      
|   |-- PASSO_A_PASSO_WINDOWS.md
|
|-- app.py                             
|-- app_databricks.py                  
|-- app.yaml                           
|-- requirements.txt
|-- README.md
```

---

## Componentes

### dbt — Bronze, Silver e Gold

O dbt substitui as transformacoes PySpark das camadas Silver e Gold. Os CSVs do
Olist sao carregados via `dbt seed` diretamente para o Unity Catalog como Bronze.
Os models Silver limpam, tipam e enriquecem os dados com joins e flags. O model
Gold principal cruza os pedidos Olist com o PIB per capita do IBGE e gera o
indicador `ticket_por_pib` que revela o paradoxo de consumo relativo: estados mais
pobres compraram proporcionalmente mais.

Os 14 testes de qualidade validam `not_null`, `unique` e `accepted_values` em cada
tabela automaticamente a cada execucao. O lineage completo (Bronze → Silver → Gold)
e visivel via `dbt docs generate`.

### Multi-LLM Routing (cost-aware)

Classificacao automatica por numero de palavras da pergunta. O usuario pode
sobrescrever para qualquer modelo via seletor na sidebar. O agente aceita chaves
de 4 provedores simultaneamente — basta configurar na sidebar.

| Complexidade | Modelo | Provedor | Custo |
| --- | --- | --- | --- |
| simple (<=40 palavras) | Llama 3.1 8B | Groq | Gratuito |
| medium (41-80 palavras) | Llama 3.3 70B | Groq | Gratuito |
| complex (>80 palavras) | Claude Sonnet 4.6 | Anthropic | $3/1M tokens |
| override | GPT-4o Mini | OpenAI | $0.15/1M tokens |
| override | GPT-4o | OpenAI | $2.5/1M tokens |
| override | Gemini 2.0 Flash | Google | $0.10/1M tokens |
| override | Gemini 1.5 Pro | Google | $1.25/1M tokens |

### Benchmark Multi-LLM

5 modelos recebem as mesmas perguntas e consultam o mesmo Gold Layer via Spark SQL.
Metricas comparadas: latencia em ms, tokens de entrada e saida e custo em USD.

### MLflow

Cada chamada ao agente registra automaticamente parametros (llm, complexity,
attempts), metricas (tokens_in, tokens_out, custo_usd, latencia_ms) e tags
(gold_table, parte).

---

## Tecnologias Utilizadas

| Categoria | Ferramentas |
| --- | --- |
| Linguagem | Python 3.12 |
| Plataforma | Databricks Free Edition |
| Armazenamento | Delta Lake · Unity Catalog |
| Transformacao | dbt-databricks · SQL |
| SQL Engine | Databricks SQL Warehouse (Serverless) |
| Orquestracao | LangGraph · LangChain |
| LLMs | Claude Sonnet 4.6 · Llama 3.1 8B · Llama 3.3 70B · GPT-4o · GPT-4o Mini · Gemini 2.0 Flash · Gemini 1.5 Pro |
| Provedores | Anthropic API · Groq API (gratuito) · OpenAI API · Google AI Studio |
| Observabilidade | MLflow |
| Interface | Streamlit |
| Dados | Olist Brazilian E-Commerce · IBGE Tabela 5938 |

---

## Autor

**Rafael Reghine Munhoz**
Data Analyst | Data Science & Analytics | MBA USP ESALQ

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rafaelreghine-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/rafaelreghine)
[![GitHub](https://img.shields.io/badge/GitHub-rreghine-black?style=flat-square&logo=github)](https://github.com/rreghine)
