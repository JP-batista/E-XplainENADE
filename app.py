"""
Camada 9 — Interface e orquestração (Streamlit).

Entry point da aplicação. Gerencia a navegação sequencial entre as etapas
analíticas e o estado da sessão via st.session_state.

Estado da sessão (st.session_state):
  df              pd.DataFrame         — dataset filtrado e pré-processado
  filtros         dict                 — filtros aplicados (grupos, regioes)
  hypothesis      HypothesisConfig     — hipótese definida pelo usuário
  formula         str                  — fórmula OLS gerada
  model_result    RegressionResults    — resultado do ajuste OLS
  vif_table       pd.DataFrame         — tabela VIF com alertas
  residuals_diag  dict                 — resultados Shapiro-Wilk + Breusch-Pagan
  shap_values     np.ndarray           — valores SHAP calculados
  shap_summary    pd.DataFrame         — impacto médio por variável
  exec_id         str (UUID4)          — identificador único da execução
  config          dict                 — configuração completa da sessão (JSON)
"""
import uuid

import streamlit as st

from config.variable_map import VARS_SUGERIDAS, get_label
from modules import etl, loader
from modules.hypothesis import build_hypothesis
from modules.modeling import fit_ols, get_coefficients_table, get_model_metrics
from modules.multicollinearity import compute_vif, get_variable_to_remove, has_multicollinearity
from modules.residuals import get_residuals_plot_data, run_diagnostics
from modules.explainability import compute_shap, get_shap_summary
from modules.report import generate_report

st.set_page_config(
    page_title="E-XplainENADE",
    page_icon="📊",
    layout="wide",
)

# ── Identificador único da execução ───────────────────────────────────────────
if "exec_id" not in st.session_state:
    st.session_state.exec_id = str(uuid.uuid4())

st.title("E-XplainENADE")
st.caption(f"Exec-ID: `{st.session_state.exec_id}`")

# ── Navegação ─────────────────────────────────────────────────────────────────
etapas = [
    "1. Carregar Dados",
    "2. Definir Hipótese",
    "3. Modelagem OLS",
    "4. Multicolinearidade (VIF)",
    "5. Diagnóstico de Resíduos",
    "6. Explicabilidade (SHAP)",
    "7. Relatório",
]
etapa = st.sidebar.radio("Etapa analítica", etapas)

# ── Etapa 1: Carregar Dados ───────────────────────────────────────────────────
if etapa == etapas[0]:
    st.header("1. Carregar Dados")
    st.info(
        "Carrega os microdados ENADE 2021 filtrados para Norte + Nordeste, "
        "Ciência da Computação e Sistemas de Informação."
    )

    if st.button("Carregar dataset"):
        with st.spinner("Lendo 43 arquivos (~30 segundos)..."):
            df = loader.get_dataset()
            st.session_state.df = df
            st.session_state.filtros = {
                "grupos": [4004, 4006],
                "regioes": [1, 2],
                "apenas_presentes": True,
            }
        st.success(f"Dataset carregado: {df.shape[0]} estudantes, {df.shape[1]} variáveis.")
        st.dataframe(df.head(10))

# ── Etapa 2: Definir Hipótese ─────────────────────────────────────────────────
elif etapa == etapas[1]:
    st.header("2. Definir Hipótese")

    if "df" not in st.session_state:
        st.warning("Carregue o dataset primeiro (Etapa 1).")
        st.stop()

    df = st.session_state.df

    y_options = ["NT_GER", "NT_FG", "NT_CE"]
    y = st.selectbox("Variável dependente (Y)", y_options)

    x_options = [c for c in df.columns if c not in y_options and not c.startswith("DS_")]
    x_labels = {c: f"{c} — {get_label(c)}" for c in x_options}

    x_sugeridas = [v for v in VARS_SUGERIDAS if v in df.columns]
    x_vars = st.multiselect(
        "Variáveis independentes (X)",
        options=x_options,
        default=x_sugeridas,
        format_func=lambda c: x_labels.get(c, c),
    )

    st.markdown("**Termos de interação** (opcional):")
    inter_a = st.selectbox("Variável A", [""] + x_vars, key="inter_a")
    inter_b = st.selectbox("Variável B", [""] + x_vars, key="inter_b")
    interactions = []
    if inter_a and inter_b and inter_a != inter_b:
        interactions = [(inter_a, inter_b)]
        st.info(f"Interação adicionada: {inter_a} × {inter_b}")

    if st.button("Confirmar hipótese") and x_vars:
        hyp = build_hypothesis(y, x_vars, interactions, df)
        st.session_state.hypothesis = hyp
        st.session_state.formula = hyp.to_formula()
        st.success(f"Fórmula: `{st.session_state.formula}`")

# ── Etapa 3: Modelagem OLS ────────────────────────────────────────────────────
elif etapa == etapas[2]:
    st.header("3. Modelagem OLS")

    if "hypothesis" not in st.session_state:
        st.warning("Defina a hipótese primeiro (Etapa 2).")
        st.stop()

    if st.button("Executar OLS"):
        result = fit_ols(st.session_state.hypothesis, st.session_state.df)
        st.session_state.model_result = result
        st.session_state.config = {
            "exec_id": st.session_state.exec_id,
            "formula": st.session_state.formula,
            "filtros": st.session_state.filtros,
        }

        metrics = get_model_metrics(result)
        st.session_state.model_metrics = metrics
        st.metric("R²", metrics["r2"])
        st.metric("R² ajustado", metrics["r2_adj"])
        st.metric("n observações", metrics["n_obs"])

        coef_table = get_coefficients_table(result)
        st.dataframe(coef_table)

# ── Etapa 4: VIF ──────────────────────────────────────────────────────────────
elif etapa == etapas[3]:
    st.header("4. Multicolinearidade (VIF)")

    if "model_result" not in st.session_state:
        st.warning("Execute o modelo OLS primeiro (Etapa 3).")
        st.stop()

    x_vars = st.session_state.hypothesis.x_vars
    vif_table = compute_vif(st.session_state.df, x_vars)
    st.session_state.vif_table = vif_table
    st.dataframe(vif_table)

    if has_multicollinearity(vif_table):
        remove_var = get_variable_to_remove(vif_table)
        st.warning(f"Multicolinearidade detectada. Variável sugerida para remoção: **{remove_var}**")
        if st.button(f"Remover {remove_var} e reajustar modelo"):
            new_x = [v for v in x_vars if v != remove_var]
            new_hyp = build_hypothesis(
                st.session_state.hypothesis.y,
                new_x,
                st.session_state.hypothesis.interactions,
                st.session_state.df,
            )
            new_result = fit_ols(new_hyp, st.session_state.df)
            st.session_state.hypothesis = new_hyp
            st.session_state.formula = new_hyp.to_formula()
            st.session_state.model_result = new_result
            st.session_state.model_metrics = get_model_metrics(new_result)
            st.success(f"Modelo reajustado sem '{remove_var}'. Fórmula: `{st.session_state.formula}`")
            st.rerun()
    else:
        st.success("Nenhuma multicolinearidade severa detectada (todos VIF ≤ 5).")

# ── Etapa 5: Diagnóstico de Resíduos ─────────────────────────────────────────
elif etapa == etapas[4]:
    st.header("5. Diagnóstico de Resíduos")

    if "model_result" not in st.session_state:
        st.warning("Execute o modelo OLS primeiro (Etapa 3).")
        st.stop()

    import plotly.express as px

    diag = run_diagnostics(st.session_state.model_result)
    st.session_state.residuals_diag = diag

    col1, col2 = st.columns(2)
    col1.metric("Shapiro-Wilk p-valor", diag["shapiro_pvalue"],
                delta="Normal ✓" if diag["shapiro_normal"] else "Não-normal ✗")
    col2.metric("Breusch-Pagan p-valor", diag["bp_pvalue"],
                delta="Homocedástico ✓" if diag["bp_homocedastic"] else "Heterocedástico ✗")

    plot_data = get_residuals_plot_data(st.session_state.model_result)
    fig = px.scatter(
        plot_data, x="fitted", y="residual",
        labels={"fitted": "Valores Preditos", "residual": "Resíduos"},
        title="Resíduos vs. Valores Preditos",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# ── Etapa 6: SHAP ─────────────────────────────────────────────────────────────
elif etapa == etapas[5]:
    st.header("6. Explicabilidade (SHAP)")

    if "model_result" not in st.session_state:
        st.warning("Execute o modelo OLS primeiro (Etapa 3).")
        st.stop()

    import plotly.express as px

    if st.button("Calcular SHAP"):
        x_vars = st.session_state.hypothesis.x_vars
        shap_values = compute_shap(
            st.session_state.model_result,
            st.session_state.df,
            x_vars,
        )
        st.session_state.shap_values = shap_values
        shap_summary = get_shap_summary(shap_values, x_vars)
        st.session_state.shap_summary = shap_summary

        fig = px.bar(
            shap_summary,
            x="shap_mean_abs", y="variavel",
            orientation="h",
            labels={"shap_mean_abs": "Impacto Médio |SHAP|", "variavel": "Variável"},
            title="Importância das Variáveis (SHAP)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Etapa 7: Relatório ────────────────────────────────────────────────────────
elif etapa == etapas[6]:
    st.header("7. Relatório Técnico")

    required = ["model_result", "vif_table", "residuals_diag", "shap_summary"]
    missing = [k for k in required if k not in st.session_state]
    if missing:
        st.warning(f"Conclua as etapas anteriores antes de gerar o relatório. Faltam: {missing}")
        st.stop()

    if st.button("Gerar PDF"):
        path = generate_report(
            session={
                "exec_id":        st.session_state.exec_id,
                "config":         st.session_state.get("config", {}),
                "formula":        st.session_state.formula,
                "model_metrics":  st.session_state.model_metrics,
                "vif_table":      st.session_state.vif_table,
                "residuals_diag": st.session_state.residuals_diag,
                "shap_summary":   st.session_state.shap_summary,
            }
        )
        st.success(f"Relatório gerado: `{path}`")
        with open(path, "rb") as f:
            st.download_button("Baixar PDF", f, file_name=path.name, mime="application/pdf")
