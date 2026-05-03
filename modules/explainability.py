"""
Camada 7 — Explicabilidade via SHAP.

Calcula os valores SHAP usando shap.LinearExplainer aplicado ao modelo OLS.
Fundamentação: Corolário 1 de Lundberg e Lee (2017) — para modelos lineares
com features independentes, os valores SHAP equivalem aos coeficientes
ponderados pela diferença em relação à média das variáveis, o que confere
rigor matemático à interpretação gerada automaticamente.
"""
from typing import List

import numpy as np
import pandas as pd
import shap
from statsmodels.regression.linear_model import RegressionResultsWrapper


def compute_shap(
    result: RegressionResultsWrapper,
    df: pd.DataFrame,
    x_vars: List[str],
) -> np.ndarray:
    """
    Calcula os valores SHAP via LinearExplainer sobre o modelo OLS ajustado.

    Parameters
    ----------
    result : RegressionResultsWrapper
        Resultado do OLS (statsmodels).
    df : pd.DataFrame
        DataFrame com os dados de entrada (pré-processado).
    x_vars : list[str]
        Variáveis independentes usadas no modelo.

    Returns
    -------
    np.ndarray
        Array (n_obs × n_features) de valores SHAP.
    """
    X = df[x_vars].dropna()
    explainer = shap.LinearExplainer(result, X)
    return explainer.shap_values(X)


def get_shap_summary(
    shap_values: np.ndarray,
    x_vars: List[str],
) -> pd.DataFrame:
    """
    Retorna o impacto médio absoluto de cada variável (base do beeswarm/bar plot).

    Returns
    -------
    pd.DataFrame
        Colunas: variavel, shap_mean_abs — ordenado do maior para o menor impacto.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"variavel": x_vars, "shap_mean_abs": mean_abs})
        .sort_values("shap_mean_abs", ascending=False)
        .reset_index(drop=True)
    )


def get_top_factors(shap_summary: pd.DataFrame, n: int = 3) -> dict:
    """
    Retorna os top-N fatores de maior impacto positivo e negativo.
    Usado para geração de texto interpretativo automático no relatório.
    """
    top_positive = shap_summary.head(n)["variavel"].tolist()
    top_negative = shap_summary.tail(n)["variavel"].tolist()
    return {"top_positive": top_positive, "top_negative": top_negative}
