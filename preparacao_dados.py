import pandas as pd

SEED = 42


def preparar_dados(dados, mediana_renda=None):
    """
    Funcao unica de preparacao de dados para o modelo de risco de credito.

    Esta funcao e reutilizavel tanto na fase de treino quanto em producao
    (Streamlit). Na base de treino, mediana_renda deve ser None (a funcao
    calcula e devolve a mediana). Na base de teste ou em novos dados de
    producao, mediana_renda deve ser o valor ja calculado no treino, para
    evitar vazamento de dados (data leakage).

    Parametros
    ----------
    dados : pd.DataFrame
        Base de dados original, com as colunas brutas do credito_tratado.csv
    mediana_renda : float ou None
        Mediana da renda mensal calculada no treino. Se None, calcula a
        partir dos proprios dados recebidos (usar apenas no treino).

    Retorna
    -------
    dados_tratados : pd.DataFrame
        Base de dados apos todos os tratamentos e novas features
    mediana_renda : float
        Mediana da renda utilizada (para reutilizar depois no teste/producao)
    """
    dados_tratados = dados.copy()

    # ============================================
    # 1 e 2 - Sinalizar e preencher renda ausente
    # ============================================
    dados_tratados['renda_ausente'] = dados_tratados['renda_mensal'].isnull().astype(int)

    if mediana_renda is None:
        mediana_renda = dados_tratados['renda_mensal'].median()

    dados_tratados['renda_mensal'] = dados_tratados['renda_mensal'].fillna(mediana_renda)

    # ============================================
    # 3 - Sinalizar dependentes ausentes e preencher com 0
    # ============================================
    dados_tratados['dependentes_ausente'] = dados_tratados['dependentes'].isnull().astype(int)
    dados_tratados['dependentes'] = dados_tratados['dependentes'].fillna(0)

    # ============================================
    # 4 - Corte (clip) das colunas de atraso em 20 + sinalizador de sentinela (96/98)
    # ============================================
    colunas_atraso = ['atrasos_30_59_dias', 'atrasos_60_89_dias', 'atrasos_90_mais_dias']

    dados_tratados['atraso_sentinela'] = (
        (dados_tratados['atrasos_30_59_dias'] >= 96) |
        (dados_tratados['atrasos_60_89_dias'] >= 96) |
        (dados_tratados['atrasos_90_mais_dias'] >= 96)
    ).astype(int)

    for coluna in colunas_atraso:
        dados_tratados[coluna] = dados_tratados[coluna].clip(upper=20)

    # ============================================
    # 5 - Corte (clip) da renda mensal em 50000
    # ============================================
    dados_tratados['renda_mensal'] = dados_tratados['renda_mensal'].clip(upper=50000)

    # ============================================
    # Novas features
    # ============================================
    dados_tratados['renda_por_dependente'] = (
        dados_tratados['renda_mensal'] / (dados_tratados['dependentes'] + 1)
    )
    dados_tratados['sobra_caixa'] = (
        dados_tratados['renda_mensal'] * (1 - dados_tratados['razao_divida'])
    )

    return dados_tratados, mediana_renda


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    base = pd.read_csv('credito_tratado.csv')

    treino, teste = train_test_split(
        base, test_size=0.25, random_state=SEED, stratify=base['inadimplente_2anos']
    )

    treino_tratado, mediana_renda_treino = preparar_dados(treino, mediana_renda=None)

    print('Mediana de renda usada (calculada no treino):', mediana_renda_treino)
    print()
    print('Dimensoes da base de treino tratada:', treino_tratado.shape)
    print()
    print('Colunas e tipos de dados:')
    print(treino_tratado.dtypes)
    print()
    print(treino_tratado.head())

    treino_tratado.to_csv('treino_tratado.csv', index=False)
