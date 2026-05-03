"""
Camada 5 — Diagnóstico de multicolinearidade (VIF).

Calcula o Fator de Inflação da Variância (VIF) para cada preditor do modelo.
Emite alertas em dois níveis: VIF > 5 (moderado) e VIF > 10 (severo).
Permite remoção interativa da variável com maior VIF e reexecução automática.
"""
from typing import List

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_MODERADO = 5.0
VIF_SEVERO = 10.0


def compute_vif(df: pd.DataFrame, x_vars: List[str]) -> pd.DataFrame:
    """
    Calcula o VIF para cada variável preditora.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as colunas de preditores (pré-processado).
    x_vars : list[str]
        Nomes das variáveis independentes usadas no modelo.

    Returns
    -------
    pd.DataFrame
        Colunas: variavel, vif, alerta
        alerta: 'ok' | 'moderado (VIF > 5)' | 'severo (VIF > 10)'
    """
    subset = df[x_vars].dropna()
    vif_values = [
        variance_inflation_factor(subset.values, i)
        for i in range(subset.shape[1])
    ]

    def _label(v: float) -> str:
        if v > VIF_SEVERO:
            return "severo (VIF > 10)"
        if v > VIF_MODERADO:
            return "moderado (VIF > 5)"
        return "ok"

    return pd.DataFrame({
        "variavel": x_vars,
        "vif":      [round(v, 3) for v in vif_values],
        "alerta":   [_label(v) for v in vif_values],
    })


def get_variable_to_remove(vif_table: pd.DataFrame) -> str:
    """
    Retorna o nome da variável com maior VIF acima do limiar moderado.
    Retorna None se todas estiverem dentro do limiar.
    """
    problematic = vif_table[vif_table["vif"] > VIF_MODERADO]
    if problematic.empty:
        return None
    return problematic.sort_values("vif", ascending=False).iloc[0]["variavel"]


def has_multicollinearity(vif_table: pd.DataFrame) -> bool:
    """Retorna True se alguma variável tem VIF > limiar moderado."""
    return (vif_table["vif"] > VIF_MODERADO).any()
