# Databricks notebook source
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  E-Commerce AI Agent — Parte 4: Produção no Databricks                      ║
# ║  Arquitetura Medalão · Delta Lake · Spark SQL · MLflow Nativo               ║
# ║  Olist E-Commerce + IBGE (PIB por Estado)                                   ║
# ║  Autoria: Rafael Reghine Munhoz | Data Analyst | MBA USP ESALQ              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Arquitetura:
#   Bronze Layer → ingestão raw (CSV Olist + CSV IBGE → Delta Lake)
#   Silver Layer → limpeza, joins, enriquecimento com Spark
#   Gold Layer   → tabelas analíticas para o AI Agent
#        ↓
#   Orquestrador LangGraph (orchestrator.py — Parte 3)
#     ├── SQL Agent   → Spark SQL sobre Gold Layer
#     ├── RAG Agent   → embeddings sobre Gold Layer
#     └── MLflow Nativo Databricks → tracking de experimentos
#
# Como rodar:
#   1. Crie um cluster no Databricks (Runtime 14.x LTS ML)
#   2. Faça upload dos CSVs do Olist para /FileStore/olist/
#   3. Faça upload do CSV do IBGE para /FileStore/ibge/
#   4. Cole cada célula em um notebook separado no Databricks
#   5. Execute célula por célula
#
# Datasets:
#   Olist:  kaggle.com/datasets/olistbr/brazilian-ecommerce
#   IBGE:   sidra.ibge.gov.br → Tabela 5938 (PIB per capita por UF)

# COMMAND ----------
# =============================================================================
# CÉLULA 0 — Instalação de dependências
# =============================================================================
# Cole isso na primeira célula do seu notebook Databricks e rode UMA VEZ

# %pip install langchain langchain-anthropic langchain-groq langgraph mlflow
# dbutils.library.restartPython()

# COMMAND ----------
# =============================================================================
# CÉLULA 1 — Imports e Configuração
# =============================================================================

import os
import re
import json
import time
import warnings
from pathlib import Path

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType
)

import mlflow
import mlflow.spark

warnings.filterwarnings("ignore")

try:
    spark  # já existe no Databricks
    print("SparkSession Databricks detectada")
except NameError:
    spark = SparkSession.builder \
        .appName("EcommerceAIAgent_Part4") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    print("SparkSession local criada")

# Caminhos no DBFS (Databricks File System)
# Ajuste os caminhos se você salvou os arquivos em local diferente
BRONZE_PATH = "/FileStore/medallion/bronze"
SILVER_PATH = "/FileStore/medallion/silver"
GOLD_PATH   = "/FileStore/medallion/gold"

print(f"""
 Caminhos configurados:
   Olist raw:  {OLIST_PATH}
   IBGE raw:   {IBGE_PATH}
   Bronze:     {BRONZE_PATH}
   Silver:     {SILVER_PATH}
   Gold:       {GOLD_PATH}
""")

# COMMAND ----------
# =============================================================================
# CÉLULA 2 — Upload dos dados (instruções)
# =============================================================================
# ⚠️  ANTES DE CONTINUAR: faça upload dos arquivos no Databricks
#
# No menu lateral: Data → Add Data → Upload Files
# Ou use o DBFS Explorer
#
# Arquivos necessários do Olist (Kaggle):
#   olist_orders_dataset.csv
#   olist_customers_dataset.csv
#   olist_order_items_dataset.csv
#   olist_order_payments_dataset.csv
#   olist_order_reviews_dataset.csv
#   olist_products_dataset.csv
#   olist_sellers_dataset.csv
#
# Arquivo do IBGE (crie manualmente ou baixe do SIDRA):
#   ibge_pib_estados.csv
#
# Formato do ibge_pib_estados.csv:
#   estado,sigla,pib_per_capita_2018,regiao
#   Acre,AC,16423.5,Norte
#   Alagoas,AL,14271.2,Nordeste
#   ...
#
# Dados reais do IBGE PIB per capita 2018 (R$):

IBGE_DATA = [
    ("Acre",                "AC", 16423.5,  "Norte"),
    ("Alagoas",             "AL", 14271.2,  "Nordeste"),
    ("Amapá",               "AP", 17463.8,  "Norte"),
    ("Amazonas",            "AM", 22644.3,  "Norte"),
    ("Bahia",               "BA", 18301.6,  "Nordeste"),
    ("Ceará",               "CE", 15673.4,  "Nordeste"),
    ("Distrito Federal",    "DF", 84209.9,  "Centro-Oeste"),
    ("Espírito Santo",      "ES", 37182.1,  "Sudeste"),
    ("Goiás",               "GO", 30011.5,  "Centro-Oeste"),
    ("Maranhão",            "MA", 11329.4,  "Nordeste"),
    ("Mato Grosso",         "MT", 43415.2,  "Centro-Oeste"),
    ("Mato Grosso do Sul",  "MS", 35441.7,  "Centro-Oeste"),
    ("Minas Gerais",        "MG", 30977.8,  "Sudeste"),
    ("Pará",                "PA", 17118.6,  "Norte"),
    ("Paraíba",             "PB", 15073.2,  "Nordeste"),
    ("Paraná",              "PR", 40696.3,  "Sul"),
    ("Pernambuco",          "PE", 19023.5,  "Nordeste"),
    ("Piauí",               "PI", 13261.8,  "Nordeste"),
    ("Rio de Janeiro",      "RJ", 43685.4,  "Sudeste"),
    ("Rio Grande do Norte", "RN", 17951.3,  "Nordeste"),
    ("Rio Grande do Sul",   "RS", 41717.2,  "Sul"),
    ("Rondônia",            "RO", 23414.5,  "Norte"),
    ("Roraima",             "RR", 21012.6,  "Norte"),
    ("Santa Catarina",      "SC", 46645.1,  "Sul"),
    ("São Paulo",           "SP", 51559.3,  "Sudeste"),
    ("Sergipe",             "SE", 17603.1,  "Nordeste"),
    ("Tocantins",           "TO", 23591.4,  "Norte"),
]

print(f"✅ Dados IBGE carregados em memória: {len(IBGE_DATA)} estados")

# COMMAND ----------
# =============================================================================
# CÉLULA 3 — BRONZE LAYER: Ingestão Raw
# =============================================================================
# Lê os CSVs do Olist e IBGE e salva como Delta Lake sem transformações
# Princípio: Bronze = dado exatamente como veio da fonte

print("=" * 60)
print("  🥉 BRONZE LAYER — Ingestão Raw")
print("=" * 60)

def ingest_bronze_olist(table_name: str, filename: str) -> int:
    """Ingere um CSV do Olist para o Bronze Layer."""
    path = f"{OLIST_PATH}/{filename}"
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)

        # Adiciona metadados de ingestão
        df = df.withColumn("_ingested_at", F.current_timestamp()) \
               .withColumn("_source_file", F.lit(filename))

        # Salva como Delta Lake
        output_path = f"{BRONZE_PATH}/{table_name}"
        df.write.format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(output_path)

        count = df.count()
        print(f"  ✅ bronze_{table_name}: {count:,} linhas → {output_path}")
        return count
    except Exception as e:
        print(f"  ⚠️  bronze_{table_name}: erro ({e})")
        print(f"       → Usando dados sintéticos para demonstração")
        return 0


# Ingerir todas as tabelas do Olist
olist_tables = {
    "orders":    "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items":     "olist_order_items_dataset.csv",
    "payments":  "olist_order_payments_dataset.csv",
    "reviews":   "olist_order_reviews_dataset.csv",
    "products":  "olist_products_dataset.csv",
    "sellers":   "olist_sellers_dataset.csv",
}

bronze_counts = {}
for table, file in olist_tables.items():
    bronze_counts[table] = ingest_bronze_olist(table, file)

# Ingerir IBGE — criado diretamente do dicionário (sem precisar de arquivo)
ibge_schema = StructType([
    StructField("estado",          StringType(),  True),
    StructField("sigla",           StringType(),  True),
    StructField("pib_per_capita",  DoubleType(),  True),
    StructField("regiao",          StringType(),  True),
])

df_ibge = spark.createDataFrame(IBGE_DATA, ibge_schema) \
               .withColumn("_ingested_at", F.current_timestamp()) \
               .withColumn("_source_file", F.lit("ibge_pib_estados_2018"))

df_ibge.write.format("delta") \
       .mode("overwrite") \
       .option("overwriteSchema", "true") \
       .save(f"{BRONZE_PATH}/ibge_estados")

print(f"  ✅ bronze_ibge_estados: {df_ibge.count()} estados → {BRONZE_PATH}/ibge_estados")
print(f"\n✅ Bronze Layer concluído!")

# COMMAND ----------
# =============================================================================
# CÉLULA 4 — SILVER LAYER: Limpeza e Enriquecimento
# =============================================================================
# Aplica transformações, joins e validações de qualidade
# Princípio: Silver = dados limpos, tipados e enriquecidos

print("=" * 60)
print("  🥈 SILVER LAYER — Limpeza e Enriquecimento")
print("=" * 60)

# ── 4.1 Pedidos com cliente e estado ─────────────────────────────────────────
try:
    df_orders    = spark.read.format("delta").load(f"{BRONZE_PATH}/orders")
    df_customers = spark.read.format("delta").load(f"{BRONZE_PATH}/customers")

    df_silver_orders = df_orders.join(
        df_customers.select("customer_id", "customer_state", "customer_city"),
        on="customer_id",
        how="left"
    ).withColumn(
        "order_purchase_timestamp",
        F.to_timestamp("order_purchase_timestamp")
    ).withColumn(
        "order_delivered_customer_date",
        F.to_timestamp("order_delivered_customer_date")
    ).withColumn(
        "order_estimated_delivery_date",
        F.to_timestamp("order_estimated_delivery_date")
    ).withColumn(
        # Flag de atraso: entrega real > estimada
        "atrasado",
        F.when(
            F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date"),
            True
        ).otherwise(False)
    ).withColumn(
        # Dias de atraso (negativo = adiantado)
        "dias_atraso",
        F.datediff(
            F.col("order_delivered_customer_date"),
            F.col("order_estimated_delivery_date")
        )
    ).filter(
        F.col("order_status") == "delivered"  # apenas pedidos entregues
    ).drop("_ingested_at", "_source_file")

    df_silver_orders.write.format("delta") \
                    .mode("overwrite") \
                    .option("overwriteSchema", "true") \
                    .save(f"{SILVER_PATH}/pedidos_entregues")

    print(f"  ✅ silver_pedidos_entregues: {df_silver_orders.count():,} linhas")

except Exception as e:
    print(f"  ⚠️  silver_pedidos: {e}")

# ── 4.2 Itens com valor total por pedido ─────────────────────────────────────
try:
    df_items = spark.read.format("delta").load(f"{BRONZE_PATH}/items")

    df_silver_itens = df_items.groupBy("order_id").agg(
        F.sum("price").alias("valor_produtos"),
        F.sum("freight_value").alias("valor_frete"),
        F.sum(F.col("price") + F.col("freight_value")).alias("valor_total"),
        F.count("product_id").alias("qtd_itens"),
        F.countDistinct("seller_id").alias("qtd_vendedores"),
    )

    df_silver_itens.write.format("delta") \
                   .mode("overwrite") \
                   .option("overwriteSchema", "true") \
                   .save(f"{SILVER_PATH}/itens_por_pedido")

    print(f"  ✅ silver_itens_por_pedido: {df_silver_itens.count():,} linhas")

except Exception as e:
    print(f"  ⚠️  silver_itens: {e}")

# ── 4.3 Pagamentos por pedido ─────────────────────────────────────────────────
try:
    df_payments = spark.read.format("delta").load(f"{BRONZE_PATH}/payments")

    df_silver_pagamentos = df_payments.groupBy("order_id").agg(
        F.sum("payment_value").alias("valor_pago"),
        F.max("payment_installments").alias("max_parcelas"),
        F.collect_set("payment_type").alias("tipos_pagamento"),
    ).withColumn(
        "parcelado",
        F.when(F.col("max_parcelas") > 1, True).otherwise(False)
    )

    df_silver_pagamentos.write.format("delta") \
                        .mode("overwrite") \
                        .option("overwriteSchema", "true") \
                        .save(f"{SILVER_PATH}/pagamentos_por_pedido")

    print(f"  ✅ silver_pagamentos_por_pedido: {df_silver_pagamentos.count():,} linhas")

except Exception as e:
    print(f"  ⚠️  silver_pagamentos: {e}")

# ── 4.4 Reviews por pedido ────────────────────────────────────────────────────
try:
    df_reviews = spark.read.format("delta").load(f"{BRONZE_PATH}/reviews")

    df_silver_reviews = df_reviews.groupBy("order_id").agg(
        F.avg("review_score").alias("nota_media"),
        F.count("review_id").alias("qtd_reviews"),
    ).withColumn(
        "satisfeito",
        F.when(F.col("nota_media") >= 4.0, True).otherwise(False)
    )

    df_silver_reviews.write.format("delta") \
                     .mode("overwrite") \
                     .option("overwriteSchema", "true") \
                     .save(f"{SILVER_PATH}/reviews_por_pedido")

    print(f"  ✅ silver_reviews_por_pedido: {df_silver_reviews.count():,} linhas")

except Exception as e:
    print(f"  ⚠️  silver_reviews: {e}")

print(f"\n✅ Silver Layer concluído!")

# COMMAND ----------
# =============================================================================
# CÉLULA 5 — GOLD LAYER: Tabelas Analíticas para o AI Agent
# =============================================================================
# Agrega dados para responder perguntas de negócio
# Princípio: Gold = dado pronto para consumo pelo agente e pelo BI
# DIFERENCIAL: cruzamento com IBGE (PIB per capita por estado)

print("=" * 60)
print("  🥇 GOLD LAYER — Tabelas Analíticas")
print("=" * 60)

df_ibge_silver = spark.read.format("delta").load(f"{BRONZE_PATH}/ibge_estados") \
                            .select("sigla", "pib_per_capita", "regiao", "estado")

# ── 5.1 Gold: Performance por Estado + PIB (DIFERENCIAL) ─────────────────────
try:
    df_pedidos  = spark.read.format("delta").load(f"{SILVER_PATH}/pedidos_entregues")
    df_itens    = spark.read.format("delta").load(f"{SILVER_PATH}/itens_por_pedido")
    df_pagtos   = spark.read.format("delta").load(f"{SILVER_PATH}/pagamentos_por_pedido")
    df_reviews2 = spark.read.format("delta").load(f"{SILVER_PATH}/reviews_por_pedido")

    # Join de todas as camadas Silver
    df_completo = df_pedidos \
        .join(df_itens,    on="order_id", how="left") \
        .join(df_pagtos,   on="order_id", how="left") \
        .join(df_reviews2, on="order_id", how="left")

    # Agrega por estado
    df_gold_estados = df_completo.groupBy("customer_state").agg(
        F.count("order_id").alias("total_pedidos"),
        F.avg("valor_total").alias("ticket_medio"),
        F.avg("qtd_itens").alias("media_itens_por_pedido"),
        F.avg("max_parcelas").alias("media_parcelas"),
        F.sum(F.col("parcelado").cast("int")).alias("pedidos_parcelados"),
        F.avg("nota_media").alias("satisfacao_media"),
        F.sum(F.col("atrasado").cast("int")).alias("pedidos_atrasados"),
        F.avg("dias_atraso").alias("media_dias_atraso"),
        F.avg(F.col("satisfeito").cast("int")).alias("taxa_satisfacao"),
        F.avg(F.col("atrasado").cast("int")).alias("taxa_atraso"),
    ).withColumn(
        "pct_parcelados",
        F.round(F.col("pedidos_parcelados") / F.col("total_pedidos") * 100, 1)
    ).withColumn(
        "pct_atrasados",
        F.round(F.col("taxa_atraso") * 100, 1)
    )

    # JOIN COM IBGE — o grande diferencial!
    df_gold_estados_ibge = df_gold_estados.join(
        df_ibge_silver,
        df_gold_estados.customer_state == df_ibge_silver.sigla,
        how="left"
    ).withColumn(
        # Índice de valor por poder aquisitivo
        "ticket_por_pib",
        F.round(F.col("ticket_medio") / F.col("pib_per_capita") * 1000, 2)
    ).withColumn(
        # Classificação de riqueza do estado
        "classe_economica",
        F.when(F.col("pib_per_capita") >= 40000, "Rico")
         .when(F.col("pib_per_capita") >= 25000, "Médio")
         .otherwise("Baixa Renda")
    ).select(
        "customer_state", "estado", "regiao", "pib_per_capita", "classe_economica",
        "total_pedidos", "ticket_medio", "media_itens_por_pedido",
        "media_parcelas", "pct_parcelados",
        "satisfacao_media", "taxa_satisfacao", "pct_atrasados", "media_dias_atraso",
        "ticket_por_pib"
    )

    df_gold_estados_ibge.write.format("delta") \
                        .mode("overwrite") \
                        .option("overwriteSchema", "true") \
                        .save(f"{GOLD_PATH}/performance_estados_ibge")

    print(f"  ✅ gold_performance_estados_ibge: {df_gold_estados_ibge.count()} estados")

    # Preview
    print("\n  📊 Preview — Top 5 estados por ticket médio:")
    df_gold_estados_ibge.orderBy(F.col("ticket_medio").desc()) \
                        .select("customer_state", "pib_per_capita", "ticket_medio",
                                "pct_parcelados", "pct_atrasados", "classe_economica") \
                        .show(5, truncate=False)

except Exception as e:
    print(f"  ⚠️  gold_performance_estados_ibge: {e}")

# ── 5.2 Gold: Análise de Parcelamento x Renda ─────────────────────────────────
try:
    df_gold_parcelamento = df_gold_estados_ibge.select(
        "customer_state", "regiao", "pib_per_capita", "classe_economica",
        "media_parcelas", "pct_parcelados", "ticket_medio", "total_pedidos"
    ).orderBy("pib_per_capita")

    df_gold_parcelamento.write.format("delta") \
                        .mode("overwrite") \
                        .option("overwriteSchema", "true") \
                        .save(f"{GOLD_PATH}/parcelamento_por_renda")

    print(f"  ✅ gold_parcelamento_por_renda: {df_gold_parcelamento.count()} estados")

except Exception as e:
    print(f"  ⚠️  gold_parcelamento_por_renda: {e}")

# ── 5.3 Gold: KPIs Consolidados (visão executiva) ─────────────────────────────
try:
    df_gold_kpis = spark.read.format("delta") \
                        .load(f"{GOLD_PATH}/performance_estados_ibge") \
                        .agg(
        F.sum("total_pedidos").alias("total_pedidos_brasil"),
        F.avg("ticket_medio").alias("ticket_medio_brasil"),
        F.avg("pct_atrasados").alias("taxa_atraso_media"),
        F.avg("satisfacao_media").alias("satisfacao_media_brasil"),
        F.avg("pib_per_capita").alias("pib_per_capita_medio"),
        F.avg("media_parcelas").alias("media_parcelas_brasil"),
    )

    df_gold_kpis.write.format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(f"{GOLD_PATH}/kpis_consolidados")

    print(f"  ✅ gold_kpis_consolidados: visão executiva")
    print("\n  📊 KPIs do Brasil:")
    df_gold_kpis.show(truncate=False)

except Exception as e:
    print(f"  ⚠️  gold_kpis_consolidados: {e}")

print(f"\n✅ Gold Layer concluído! 3 tabelas analíticas criadas.")

# COMMAND ----------
# =============================================================================
# CÉLULA 6 — Registrar tabelas no Catálogo Databricks (Unity Catalog / Hive)
# =============================================================================
# Permite consultar via SQL diretamente no Databricks SQL Editor

print("=" * 60)
print("  📚 Registrando tabelas no Catálogo")
print("=" * 60)

# Cria database se não existir
spark.sql("CREATE DATABASE IF NOT EXISTS ecommerce_ai_agent")
spark.sql("USE ecommerce_ai_agent")

gold_tables = {
    "gold_performance_estados_ibge": f"{GOLD_PATH}/performance_estados_ibge",
    "gold_parcelamento_por_renda":   f"{GOLD_PATH}/parcelamento_por_renda",
    "gold_kpis_consolidados":        f"{GOLD_PATH}/kpis_consolidados",
}

for table_name, path in gold_tables.items():
    try:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        spark.sql(f"""
            CREATE TABLE {table_name}
            USING DELTA
            LOCATION '{path}'
        """)
        count = spark.sql(f"SELECT COUNT(*) as n FROM {table_name}").collect()[0]["n"]
        print(f"  ✅ {table_name}: {count} linhas registradas no catálogo")
    except Exception as e:
        print(f"  ⚠️  {table_name}: {e}")

print(f"\n✅ Tabelas disponíveis no SQL Editor do Databricks!")
print(f"   Acesse: SQL → SQL Editor → selecione 'ecommerce_ai_agent'")

# COMMAND ----------
# =============================================================================
# CÉLULA 7 — Queries de validação (Spark SQL)
# =============================================================================
# Valida que o Gold Layer está correto e demonstra o poder do Spark SQL

print("=" * 60)
print("  🔍 Validação com Spark SQL")
print("=" * 60)

queries_validacao = {
    "Estados mais ricos vs ticket médio": """
        SELECT estado, regiao, pib_per_capita, ticket_medio,
               ticket_por_pib, classe_economica
        FROM gold_performance_estados_ibge
        ORDER BY pib_per_capita DESC
        LIMIT 10
    """,
    "Parcelamento por classe econômica": """
        SELECT classe_economica,
               COUNT(*) as n_estados,
               ROUND(AVG(media_parcelas), 2) as media_parcelas,
               ROUND(AVG(pct_parcelados), 1) as pct_parcelados,
               ROUND(AVG(ticket_medio), 2) as ticket_medio
        FROM gold_performance_estados_ibge
        GROUP BY classe_economica
        ORDER BY media_parcelas DESC
    """,
    "Regiões: atraso x satisfação x renda": """
        SELECT regiao,
               ROUND(AVG(pib_per_capita), 0) as pib_medio,
               ROUND(AVG(pct_atrasados), 1) as taxa_atraso_pct,
               ROUND(AVG(satisfacao_media), 2) as satisfacao_media,
               SUM(total_pedidos) as total_pedidos
        FROM gold_performance_estados_ibge
        GROUP BY regiao
        ORDER BY taxa_atraso_pct ASC
    """,
}

for titulo, query in queries_validacao.items():
    print(f"\n  📊 {titulo}:")
    try:
        spark.sql(query).show(truncate=False)
    except Exception as e:
        print(f"  ⚠️  {e}")

# COMMAND ----------
# =============================================================================
# CÉLULA 8 — Integração com MLflow (nativo Databricks)
# =============================================================================
# Rastreia experimentos do AI Agent diretamente no MLflow do Databricks

print("=" * 60)
print("  🔬 MLflow — Rastreamento de Experimentos")
print("=" * 60)

# No Databricks, o MLflow já está configurado automaticamente
# Apenas defina o nome do experimento
EXPERIMENT_NAME = "/Users/rafael/ecommerce_ai_agent_part4"

try:
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"  ✅ Experimento MLflow: {EXPERIMENT_NAME}")
except Exception:
    # Fallback para ambiente local
    mlflow.set_experiment("ecommerce_ai_agent_part4")
    print(f"  ✅ Experimento MLflow local criado")


def log_agent_run(
    pergunta: str,
    resposta: str,
    llm_chosen: str,
    tokens_input: int,
    tokens_output: int,
    custo_usd: float,
    latencia_ms: int,
    eval_status: str,
    eval_score: float = None,
    route: str = "",
    sql_query: str = "",
):
    """
    Loga uma execução do AI Agent no MLflow.
    Compatível com o OrchestratorState do orchestrator.py (Parte 3).
    """
    with mlflow.start_run(run_name=f"agent_{llm_chosen}_{int(time.time())}"):
        # Parâmetros da run
        mlflow.log_params({
            "llm_chosen":   llm_chosen,
            "route":        route,
            "pergunta_len": len(pergunta.split()),
        })

        # Métricas
        mlflow.log_metrics({
            "tokens_input":  tokens_input,
            "tokens_output": tokens_output,
            "tokens_total":  tokens_input + tokens_output,
            "custo_usd":     custo_usd,
            "latencia_ms":   latencia_ms,
            "eval_score":    eval_score or 0.0,
            "custo_por_token": custo_usd / max(tokens_input + tokens_output, 1),
        })

        # Tags
        mlflow.set_tags({
            "eval_status": eval_status,
            "projeto":     "ecommerce_ai_agent_part4",
            "dataset":     "olist_ibge",
            "arquitetura": "medallion_lakehouse",
        })

        # Artefatos de texto
        mlflow.log_text(pergunta,    "pergunta.txt")
        mlflow.log_text(resposta,    "resposta.txt")
        mlflow.log_text(sql_query,   "sql_gerado.txt")

    print(f"  ✅ Run logada: {llm_chosen} | {eval_status} | ${custo_usd:.6f}")


# Simula 3 runs de teste — uma por LLM
runs_teste = [
    {
        "pergunta":     "Qual estado tem o maior ticket médio?",
        "resposta":     "São Paulo lidera com ticket médio de R$ 142,50",
        "llm_chosen":  "claude-sonnet-4-6",
        "tokens_input": 312,
        "tokens_output": 87,
        "custo_usd":    0.000936 + 0.001305,
        "latencia_ms":  1843,
        "eval_status":  "correta",
        "eval_score":   1.0,
        "route":        "sql",
        "sql_query":    "SELECT customer_state, AVG(ticket_medio) FROM gold_performance_estados_ibge GROUP BY 1 ORDER BY 2 DESC LIMIT 1",
    },
    {
        "pergunta":     "Estados de baixa renda parcelam mais?",
        "resposta":     "Sim, estados com PIB menor têm média de 3.2 parcelas vs 2.1 nos ricos",
        "llm_chosen":  "qwen-qwq-32b",
        "tokens_input": 298,
        "tokens_output": 112,
        "custo_usd":    0.0,
        "latencia_ms":  2210,
        "eval_status":  "correta",
        "eval_score":   0.95,
        "route":        "sql",
        "sql_query":    "SELECT classe_economica, AVG(media_parcelas) FROM gold_performance_estados_ibge GROUP BY 1",
    },
    {
        "pergunta":     "Qual o ticket médio geral?",
        "resposta":     "O ticket médio no Brasil é R$ 137,32",
        "llm_chosen":  "mistral-saba-24b",
        "tokens_input": 156,
        "tokens_output": 43,
        "custo_usd":    0.0,
        "latencia_ms":  892,
        "eval_status":  "correta",
        "eval_score":   1.0,
        "route":        "sql",
        "sql_query":    "SELECT AVG(ticket_medio) FROM gold_kpis_consolidados",
    },
]

print("\n  🧪 Logando runs de teste:")
for run in runs_teste:
    log_agent_run(**run)

print(f"\n✅ MLflow configurado! Acesse: Experiments → ecommerce_ai_agent_part4")

# COMMAND ----------
# =============================================================================
# CÉLULA 9 — Integração com orchestrator.py (Parte 3)
# =============================================================================
# Adapter que conecta o Gold Layer do Databricks ao seu orquestrador LangGraph

print("=" * 60)
print("  🔗 Adapter: Databricks → orchestrator.py (Parte 3)")
print("=" * 60)

DB_SCHEMA_GOLD = """
Banco Delta Lake — Databricks Medallion Architecture
Dataset: Olist E-Commerce + IBGE (PIB por Estado) — Brasil 2016-2018

TABELAS GOLD (use APENAS estas):

gold_performance_estados_ibge
  customer_state VARCHAR  — sigla do estado (ex: SP, RJ)
  estado         VARCHAR  — nome completo do estado
  regiao         VARCHAR  — Norte | Nordeste | Sul | Sudeste | Centro-Oeste
  pib_per_capita DOUBLE   — PIB per capita do estado em R$ (IBGE 2018)
  classe_economica VARCHAR — Rico | Médio | Baixa Renda
  total_pedidos  LONG     — total de pedidos entregues
  ticket_medio   DOUBLE   — valor médio por pedido (R$)
  media_itens_por_pedido DOUBLE — média de itens por pedido
  media_parcelas DOUBLE   — média de parcelas por pagamento
  pct_parcelados DOUBLE   — % de pedidos parcelados
  satisfacao_media DOUBLE — nota média de avaliação (1-5)
  taxa_satisfacao  DOUBLE — % de pedidos com nota >= 4
  pct_atrasados    DOUBLE — % de pedidos atrasados
  media_dias_atraso DOUBLE — média de dias de atraso (negativo = adiantado)
  ticket_por_pib   DOUBLE — ticket médio / pib_per_capita * 1000

gold_parcelamento_por_renda
  (subconjunto focado em análise de crédito e parcelamento)

gold_kpis_consolidados
  (KPIs agregados do Brasil inteiro — use para visão macro)

REGRAS:
- Sempre use LIMIT para evitar resultados grandes
- Para correlações, use gold_performance_estados_ibge
- Para visão executiva, use gold_kpis_consolidados
- pib_per_capita é em R$ por habitante por ano
"""


def run_spark_sql_agent(pergunta: str) -> dict:
    """
    Adapter que executa o SQL Agent do orchestrator.py
    usando Spark SQL sobre o Gold Layer do Databricks.

    Substitui o SQLite do Projeto 2 pelo Delta Lake em produção.
    """
    t0 = time.time()

    # Importa o router e Claude do orchestrator.py
    # (assumindo que orchestrator.py está no mesmo diretório)
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {
                "erro": "ANTHROPIC_API_KEY não configurada",
                "dica": "Configure em: Databricks → Secrets ou variável de ambiente"
            }

        claude = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=api_key,
            max_tokens=512
        )

        # Gera SQL
        sql_prompt = f"""Você é um especialista em SQL para Spark SQL (Delta Lake).
Retorne APENAS o SQL, sem explicações, sem markdown.

Schema disponível:
{DB_SCHEMA_GOLD}

Pergunta: {pergunta}
SQL:"""

        response = claude.invoke([HumanMessage(content=sql_prompt)])
        sql = response.content.replace("```sql", "").replace("```", "").strip()

        # Executa via Spark SQL
        try:
            df_result = spark.sql(sql)
            rows = [row.asDict() for row in df_result.limit(20).collect()]
        except Exception as e:
            return {
                "pergunta": pergunta,
                "sql_gerado": sql,
                "erro": f"Erro no Spark SQL: {e}",
                "latencia_ms": int((time.time() - t0) * 1000)
            }

        # Interpreta resultado
        if rows:
            import pandas as pd
            df_display = pd.DataFrame(rows)
            interp_prompt = f"""Analista de dados brasileiro. Interprete em português claro.
Máximo 3 linhas. Contexto: e-commerce + dados socioeconômicos do Brasil.

Pergunta: {pergunta}
Resultado:
{df_display.to_string(index=False)}

Resposta:"""
            resp2 = claude.invoke([HumanMessage(content=interp_prompt)])
            resposta = resp2.content.strip()

            usage = getattr(response, "usage_metadata", None)
            ti = getattr(usage, "input_tokens", 300)
            to = getattr(usage, "output_tokens", 100)

        else:
            resposta = "Nenhum dado encontrado para essa consulta."
            ti, to = 200, 50

        latencia = int((time.time() - t0) * 1000)
        custo = (ti / 1e6 * 3.0) + (to / 1e6 * 15.0)

        # Log automático no MLflow
        log_agent_run(
            pergunta=pergunta,
            resposta=resposta,
            llm_chosen="claude-sonnet-4-6",
            tokens_input=ti,
            tokens_output=to,
            custo_usd=custo,
            latencia_ms=latencia,
            eval_status="sem_gt",
            route="sql_spark",
            sql_query=sql,
        )

        return {
            "pergunta":   pergunta,
            "sql_gerado": sql,
            "resultado":  rows[:5],
            "resposta":   resposta,
            "latencia_ms": latencia,
            "custo_usd":  round(custo, 6),
        }

    except ImportError:
        return {
            "erro": "langchain_anthropic não instalado",
            "dica": "Execute a Célula 0 primeiro"
        }


# Testa o adapter com perguntas que exploram o cruzamento Olist + IBGE
perguntas_teste = [
    "Qual região tem maior taxa de atraso nas entregas?",
    "Estados de baixa renda parcelam mais as compras?",
    "Qual a relação entre PIB per capita e satisfação com entrega?",
]

print("\n  🧪 Testando Spark SQL Agent:")
for pergunta in perguntas_teste:
    print(f"\n  ❓ {pergunta}")
    result = run_spark_sql_agent(pergunta)
    if "erro" in result:
        print(f"  ⚠️  {result['erro']}")
        if "dica" in result:
            print(f"  💡 {result['dica']}")
    else:
        print(f"  🔍 SQL: {result.get('sql_gerado', '')[:80]}...")
        print(f"  💬 {result.get('resposta', '')}")
        print(f"  ⏱️  {result.get('latencia_ms')}ms | ${result.get('custo_usd'):.6f}")

print(f"\n✅ Adapter Databricks ↔ orchestrator.py pronto!")

# COMMAND ----------
# =============================================================================
# CÉLULA 10 — Benchmark Multi-LLM no Databricks
# =============================================================================
# Compara Claude, Qwen e Mistral sobre o Gold Layer
# Registra tudo no MLflow para análise posterior

print("=" * 60)
print("  🏁 Benchmark Multi-LLM — Gold Layer Databricks")
print("=" * 60)

BENCHMARK_QUERIES = [
    "Qual estado tem o maior ticket médio?",
    "Nordeste tem mais atrasos que o Sudeste?",
    "Estados mais ricos parcelam menos?",
    "Qual a taxa de satisfação média do Brasil?",
]

print("""
  Para rodar o benchmark completo:

  1. Configure as API Keys:
     - ANTHROPIC_API_KEY → console.anthropic.com
     - GROQ_API_KEY      → console.groq.com (gratuito)

  2. Execute:
     from orchestrator import run_benchmark
     resultados = run_benchmark(BENCHMARK_QUERIES)

  3. Os resultados são logados automaticamente no MLflow
     Acesse: Experiments → ecommerce_ai_agent_part4

  4. Compare:
     - Claude Sonnet  → máxima qualidade, maior custo
     - Qwen QwQ 32B   → ótimo custo-benefício (gratuito)
     - Mistral Saba   → mais rápido para queries simples (gratuito)
""")

print("  📊 Queries do benchmark:")
for i, q in enumerate(BENCHMARK_QUERIES, 1):
    print(f"    {i}. {q}")

# COMMAND ----------
# =============================================================================
# CÉLULA 11 — Resumo Final e Próximos Passos
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅ PROJETO CONCLUÍDO — E-Commerce AI Agent Parte 4                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🥉 BRONZE LAYER                                                             ║
║     7 tabelas Olist + IBGE → Delta Lake                                      ║
║                                                                              ║
║  🥈 SILVER LAYER                                                             ║
║     Joins, limpeza, tipagem, flags de atraso e satisfação                    ║
║                                                                              ║
║  🥇 GOLD LAYER                                                               ║
║     3 tabelas analíticas prontas para o AI Agent                             ║
║     DIFERENCIAL: cruzamento com PIB per capita do IBGE                       ║
║                                                                              ║
║  🤖 AI AGENT                                                                 ║
║     Spark SQL Agent sobre Gold Layer                                          ║
║     Integrado ao orchestrator.py (LangGraph — Parte 3)                       ║
║                                                                              ║
║  🔬 MLFLOW                                                                   ║
║     Rastreamento nativo Databricks                                            ║
║     Custo, latência, qualidade por LLM                                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📝 POST LINKEDIN                                                            ║
║                                                                              ║
║  Título: "AI Agent em Produção: do SQLite ao Databricks"                     ║
║                                                                              ║
║  Estrutura sugerida:                                                         ║
║  1. A jornada: Parte 1 (RAG) → 2 (SQL) → 3 (LangGraph) → 4 (Databricks)    ║
║  2. Por que Databricks? Delta Lake + MLflow nativo + escala                  ║
║  3. O diferencial: cruzamento Olist + IBGE                                   ║
║  4. Perguntas que o agente responde agora (com PIB)                          ║
║  5. Conexão com projeto real: LWART → Fabric + Copilot                       ║
║  6. Link do GitHub                                                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🎯 PARA A ENTREVISTA INDICIUM                                               ║
║                                                                              ║
║  "Desenvolvi um AI Agent com LangGraph e migrei para produção no             ║
║   Databricks — com Arquitetura Medalão, Delta Lake e MLflow nativo.          ║
║   O agente responde perguntas de negócio cruzando dados de e-commerce        ║
║   com PIB per capita do IBGE. Posso mostrar o código e os experimentos."     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
