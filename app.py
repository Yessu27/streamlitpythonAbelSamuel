import glob
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from preparacao_dados import preparar_dados

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(page_title="Avaliação de Risco de Crédito", page_icon="💳", layout="centered")

PASTA_APP = os.path.dirname(os.path.abspath(__file__))

# Valores usados apenas como fallback, caso o arquivo
# 'metadados_modelo.json' não esteja na pasta do app.
COLUNAS_ESPERADAS_PADRAO = [
    "idade", "renda_mensal", "dependentes", "uso_limite_rotativo", "razao_divida",
    "linhas_credito_abertas", "financiamentos_imobiliarios", "atrasos_30_59_dias",
    "atrasos_60_89_dias", "atrasos_90_mais_dias", "renda_ausente", "dependentes_ausente",
    "atraso_sentinela", "renda_por_dependente", "sobra_caixa",
]
LIMIAR_DECISAO_PADRAO = 0.59


# ============================================================
# Carregamento do modelo e dos metadados (cacheados)
# ============================================================
@st.cache_resource
def carregar_modelo():
    candidatos = sorted(glob.glob(os.path.join(PASTA_APP, "modelo_xgboost_credito*.pkl")))
    if not candidatos:
        raise FileNotFoundError(
            "Nenhum arquivo 'modelo_xgboost_credito*.pkl' encontrado na pasta do app. "
            "Coloque o .pkl do modelo na mesma pasta do app.py."
        )
    caminho_modelo = candidatos[0]
    modelo = joblib.load(caminho_modelo)
    return modelo, caminho_modelo


@st.cache_resource
def carregar_metadados(_modelo):
    # Valores de referência: sempre calculados a partir do próprio modelo,
    # usados como fallback campo a campo (nunca "tudo ou nada").
    imputador = _modelo.named_steps["imputacao"]
    colunas_modelo = list(imputador.feature_names_in_)
    estatisticas = dict(zip(colunas_modelo, imputador.statistics_))
    valores_padrao = {
        "limiar_decisao": LIMIAR_DECISAO_PADRAO,
        "mediana_renda_treino": float(estatisticas.get("renda_mensal", 5400.0)),
        "colunas_esperadas": colunas_modelo,
    }

    caminho_json = os.path.join(PASTA_APP, "metadados_modelo.json")
    meta = dict(valores_padrao)
    campos_ausentes = []

    if os.path.exists(caminho_json):
        with open(caminho_json, "r", encoding="utf-8") as arquivo:
            meta_arquivo = json.load(arquivo)
        for chave in valores_padrao:
            valor = meta_arquivo.get(chave)
            if valor not in (None, "", []):
                meta[chave] = valor
            else:
                campos_ausentes.append(chave)
    else:
        campos_ausentes = list(valores_padrao.keys())

    # Validação: as colunas do metadados precisam bater exatamente com as
    # que o modelo espera (mesmo conjunto e mesma ordem). Se não baterem,
    # é sinal de que o .json e o .pkl não são do mesmo treino — melhor
    # parar o app com uma mensagem clara do que gerar previsão errada.
    if list(meta["colunas_esperadas"]) != colunas_modelo:
        st.error(
            "As `colunas_esperadas` do metadados_modelo.json não coincidem com as "
            "colunas que este modelo (.pkl) realmente espera. Verifique se os dois "
            "arquivos vêm do mesmo treino."
        )
        st.stop()

    origem = "completo" if not campos_ausentes else ("padrao" if len(campos_ausentes) == len(valores_padrao) else "parcial")
    return meta, origem, campos_ausentes


modelo, caminho_modelo = carregar_modelo()
metadados, origem_metadados, campos_ausentes_metadados = carregar_metadados(modelo)

LIMIAR_DECISAO = float(metadados["limiar_decisao"])
MEDIANA_RENDA_TREINO = float(metadados["mediana_renda_treino"])
COLUNAS_ESPERADAS = metadados["colunas_esperadas"]

# ============================================================
# Cabeçalho
# ============================================================
st.title("💳 Avaliação de Risco de Crédito")
st.caption(
    f"Modelo carregado de `{os.path.basename(caminho_modelo)}` · "
    f"Limiar de decisão: {LIMIAR_DECISAO:.2f}"
)

if origem_metadados == "padrao":
    st.warning(
        "Arquivo `metadados_modelo.json` não encontrado nesta pasta. O app está usando "
        "um limiar padrão (0.59) e uma mediana de renda estimada a partir do próprio modelo. "
        "Para reproduzir exatamente os valores do treino, gere e inclua o metadados_modelo.json "
        "junto com o .pkl."
    )
elif origem_metadados == "parcial":
    st.warning(
        "Os seguintes campos não foram encontrados (ou estavam vazios) no "
        "`metadados_modelo.json` e usaram um valor padrão estimado a partir do modelo: "
        f"{', '.join(campos_ausentes_metadados)}."
    )

st.write("Preencha as informações do cliente para obter a previsão de inadimplência.")

# ============================================================
# Formulário de entrada
# ============================================================
with st.form("form_credito"):
    st.subheader("Dados do cliente")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=110, value=40, step=1)

        renda_nao_informada = st.checkbox("Renda mensal não informada")
        renda_mensal = st.number_input(
            "Renda mensal (R$)", min_value=0.0, value=5000.0, step=100.0,
            disabled=renda_nao_informada,
        )

        dependentes_nao_informado = st.checkbox("Nº de dependentes não informado")
        dependentes = st.number_input(
            "Número de dependentes", min_value=0, max_value=20, value=0, step=1,
            disabled=dependentes_nao_informado,
        )

        linhas_credito_abertas = st.number_input(
            "Linhas de crédito abertas", min_value=0, max_value=100, value=5, step=1
        )
        financiamentos_imobiliarios = st.number_input(
            "Financiamentos imobiliários", min_value=0, max_value=20, value=1, step=1
        )

    with col2:
        uso_limite_rotativo = st.number_input(
            "Uso do limite rotativo (proporção do limite utilizada)",
            min_value=0.0, max_value=5.0, value=0.30, step=0.01, format="%.4f",
        )
        razao_divida = st.number_input(
            "Razão dívida/renda", min_value=0.0, max_value=10.0, value=0.35, step=0.01, format="%.4f"
        )
        atrasos_30_59_dias = st.number_input(
            "Nº de atrasos de 30-59 dias (últimos 2 anos)", min_value=0, max_value=98, value=0, step=1
        )
        atrasos_60_89_dias = st.number_input(
            "Nº de atrasos de 60-89 dias (últimos 2 anos)", min_value=0, max_value=98, value=0, step=1
        )
        atrasos_90_mais_dias = st.number_input(
            "Nº de atrasos de 90+ dias (últimos 2 anos)", min_value=0, max_value=98, value=0, step=1
        )

    enviado = st.form_submit_button("Avaliar crédito", use_container_width=True)

# ============================================================
# Previsão
# ============================================================
if enviado:
    dados_brutos = pd.DataFrame([{
        "idade": idade,
        "renda_mensal": (np.nan if renda_nao_informada else renda_mensal),
        "dependentes": (np.nan if dependentes_nao_informado else dependentes),
        "uso_limite_rotativo": uso_limite_rotativo,
        "razao_divida": razao_divida,
        "linhas_credito_abertas": linhas_credito_abertas,
        "financiamentos_imobiliarios": financiamentos_imobiliarios,
        "atrasos_30_59_dias": atrasos_30_59_dias,
        "atrasos_60_89_dias": atrasos_60_89_dias,
        "atrasos_90_mais_dias": atrasos_90_mais_dias,
    }])

    # Mesma função usada no treino, garantindo consistência
    # (sem vazamento de dados: mediana vem do treino).
    dados_tratados, _ = preparar_dados(dados_brutos, mediana_renda=MEDIANA_RENDA_TREINO)
    dados_tratados = dados_tratados[COLUNAS_ESPERADAS]

    # 1) Probabilidade de inadimplência
    probabilidade = float(modelo.predict_proba(dados_tratados)[0, 1])

    # 2) Decisão com base no limiar
    recusar = probabilidade >= LIMIAR_DECISAO

    st.subheader("Resultado da avaliação")

    c1, c2 = st.columns(2)
    c1.metric("Probabilidade de inadimplência", f"{probabilidade:.1%}")
    c2.metric("Decisão do modelo", "❌ Recusar crédito" if recusar else "✅ Conceder crédito")

    st.caption(
        f"Decisão baseada no limiar de {LIMIAR_DECISAO:.2f}: "
        f"probabilidade {'≥' if recusar else '<'} limiar ⇒ "
        f"{'recusar' if recusar else 'conceder'} o crédito."
    )

    # 3) SHAP — 5 features mais importantes para este cliente
    st.subheader("Principais fatores da decisão (SHAP)")

    imputador = modelo.named_steps["imputacao"]
    modelo_xgb = modelo.named_steps["modelo"]

    dados_imputados = pd.DataFrame(
        imputador.transform(dados_tratados), columns=COLUNAS_ESPERADAS
    )

    explainer = shap.TreeExplainer(modelo_xgb)
    valores_shap = np.array(explainer.shap_values(dados_imputados)).reshape(-1)

    shap_df = pd.DataFrame({
        "feature": COLUNAS_ESPERADAS,
        "valor_cliente": dados_imputados.iloc[0].values,
        "shap": valores_shap,
    })
    shap_df["impacto_abs"] = shap_df["shap"].abs()
    top5 = shap_df.sort_values("impacto_abs", ascending=False).head(5).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    cores = ["#d62728" if v > 0 else "#2ca02c" for v in top5["shap"]]
    ax.barh(top5["feature"], top5["shap"], color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impacto no risco (SHAP, escala log-odds)")
    ax.set_title("Top 5 features mais importantes para este cliente")
    st.pyplot(fig)

    st.caption("🔴 aumenta o risco de inadimplência · 🟢 reduz o risco de inadimplência")

    with st.expander("Ver valores detalhados"):
        st.dataframe(
            top5[["feature", "valor_cliente", "shap"]]
            .rename(columns={
                "feature": "Variável", "valor_cliente": "Valor do cliente", "shap": "Impacto SHAP",
            })
            .sort_values("Impacto SHAP", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True)
        )