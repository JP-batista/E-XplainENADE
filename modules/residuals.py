"""
Camada 6 — Diagnóstico de resíduos.

Executa os testes de pressupostos do modelo OLS:
  - Normalidade dos resíduos: Shapiro-Wilk (scipy.stats)
  - Homocedasticidade: Breusch-Pagan (statsmodels)

Também prepara os dados para o gráfico interativo de resíduos vs. fitted.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan


def run_diagnostics(result: RegressionResultsWrapper) -> dict:
    """
    Executa Shapiro-Wilk e Breusch-Pagan sobre os resíduos do modelo ajustado.

    Returns
    -------
    dict com chaves:
      shapiro_stat, shapiro_pvalue, shapiro_normal (bool)
      bp_stat, bp_pvalue, bp_homocedastic (bool)
    """
    residuals = result.resid.values

    sw_stat, sw_pvalue = stats.shapiro(residuals)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, result.model.exog)

    return {
        "shapiro_stat":     round(float(sw_stat), 6),
        "shapiro_pvalue":   round(float(sw_pvalue), 6),
        "shapiro_normal":   float(sw_pvalue) >= 0.05,
        "bp_stat":          round(float(bp_stat), 6),
        "bp_pvalue":        round(float(bp_pvalue), 6),
        "bp_homocedastic":  float(bp_pvalue) >= 0.05,
    }


def get_residuals_plot_data(result: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Retorna DataFrame com resíduos e valores preditos para plotagem.

    Colunas: fitted (valores preditos), residual, residual_std (padronizado)
    """
    fitted = result.fittedvalues
    residuals = result.resid
    std_resid = residuals / residuals.std()

    return pd.DataFrame({
        "fitted":       fitted.values,
        "residual":     residuals.values,
        "residual_std": std_resid.values,
    })
