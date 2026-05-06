"""
Camada 4 — Engine de modelagem estatística (OLS).

Ajusta o modelo de regressão linear múltipla via statsmodels.OLS,
extrai coeficientes, p-valores e IC 95%, e gera interpretação textual.

Escolha de statsmodels (não scikit-learn): o E-XplainENADE é confirmatório,
não preditivo. O objetivo é testar hipóteses sobre relações entre variáveis,
o que requer p-valores, IC e testes de hipótese — ausentes no scikit-learn.
"""
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from modules.hypothesis import HypothesisConfig


def fit_ols(
    hypothesis: HypothesisConfig,
    df: pd.DataFrame,
) -> RegressionResultsWrapper:
    """
    Ajusta o modelo OLS a partir de uma hipótese e um DataFrame.

    Parameters
    ----------
    hypothesis : HypothesisConfig
        Hipótese com Y, X e interações; formula gerada automaticamente.
    df : pd.DataFrame
        DataFrame pré-processado (saída de loader.get_dataset()).

    Returns
    -------
    RegressionResultsWrapper
        Objeto de resultado do statsmodels com coeficientes, p-valores e IC.
    """
    formula = hypothesis.to_formula()
    model = smf.ols(formula=formula, data=df)
    return model.fit()


def get_coefficients_table(result: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Extrai tabela de coeficientes com p-valores e IC 95%.

    Returns
    -------
    pd.DataFrame
        Colunas: variável, coeficiente, erro_padrao, t_stat, p_valor, ic_inf, ic_sup
    """
    summary = result.summary2().tables[1].reset_index()
    summary.columns = ["variavel", "coeficiente", "erro_padrao", "t_stat",
                        "p_valor", "ic_inf_95", "ic_sup_95"]
    return summary


def get_model_metrics(result: RegressionResultsWrapper) -> dict:
    """Retorna R², R² ajustado, AIC, BIC e n observações."""
    return {
        "r2":          round(float(result.rsquared), 4),
        "r2_adj":      round(float(result.rsquared_adj), 4),
        "aic":         round(float(result.aic), 2),
        "bic":         round(float(result.bic), 2),
        "n_obs":       int(result.nobs),
        "f_stat":      round(float(result.fvalue), 4),
        "f_pvalue":    round(float(result.f_pvalue), 6),
    }
