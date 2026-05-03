"""
Camada 9 — Interface e orquestração (Streamlit).

Implementa as 7 etapas do Comportamento do Sistema definido no TCC:
  Etapa 1: Seleção de Base
  Etapa 2: Definição Formal da Hipótese
  Etapa 3: Modelagem + Diagnóstico Automático
  Etapa 4: VIF + Remoção Interativa
  Etapa 5: Diagnóstico de Resíduos
  Etapa 6: Explicabilidade SHAP
  Etapa 7: Relatório PDF
"""
import uuid
import itertools
from typing import List, Tuple

import pandas as pd
import streamlit as st

from config.variable_map import CODE_TO_LABEL
from modules import etl, loader
from modules.hypothesis import build_hypothesis
from modules.modeling import fit_ols, get_coefficients_table, get_model_metrics
from modules.multicollinearity import (
    compute_vif, has_multicollinearity, get_variable_to_remove,
    VIF_MODERADO, VIF_SEVERO,
)
from modules.residuals import get_residuals_plot_data, run_diagnostics
from modules.explainability import compute_shap, get_shap_summary
from modules.report import generate_report

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-XplainENADE",
    page_icon="📊",
    layout="wide",
)

# ── Constantes de UI ──────────────────────────────────────────────────────────

Y_OPTS = {
    "Nota Geral":            "NT_GER",
    "Formação Geral":        "NT_FG",
    "Componente Específico": "NT_CE",
}

X_OPTS = {
    "Renda Familiar":               "QE_RENDA",
    "Escolaridade da Mãe":          "QE_ESC_MAE",
    "Escolaridade do Pai":          "QE_ESC_PAI",
    "Tipo de Escola (EM)":          "QE_TIPO_EM",
    "Horas de Estudo por Semana":   "QE_HORAS_ESTUDO",
    "Situação de Trabalho":         "QE_TRABALHO",
    "Familiar com Ens. Superior":   "QE_FAM_SUPERIOR",
    "Sexo":                         "TP_SEXO",
    "Turno da Graduação":           "CO_TURNO",
    "Tipo de IES":                  "CO_CATEGAD",
    "Ingresso por Ação Afirmativa": "QE_ACAO_AFIRM",
}
CODE_TO_UI_LABEL = {v: k for k, v in X_OPTS.items()}
CODE_TO_UI_LABEL.update({v: k for k, v in Y_OPTS.items()})

# Apenas 2021 disponível. Outros anos serão integrados via ENADE-Time (Lucas).
ANOS_DISPONIVEIS = ["2021"]
CURSO_OPTS   = {"Todos": [4004, 4006], "Ciência da Computação": [4004], "Sistemas de Informação": [4006]}
IES_OPTS     = {"Todas": None, "Pública": [1, 2, 3], "Privada": [4, 5]}

DEFAULT_X    = ["Renda Familiar", "Escolaridade da Mãe", "Horas de Estudo por Semana"]

# ── Exec-ID da sessão ─────────────────────────────────────────────────────────
if "exec_id" not in st.session_state:
    st.session_state.exec_id = str(uuid.uuid4())


# ── Funções de interpretação automática ──────────────────────────────────────

def _label(code: str) -> str:
    return CODE_TO_UI_LABEL.get(code, CODE_TO_LABEL.get(code, code))


def _p_str(p: float) -> str:
    if p < 0.001:
        return "p < 0,001"
    return f"p = {p:.3f}".replace(".", ",")


def _interpretar_modelo(coef_table: pd.DataFrame, metrics: dict, y_label: str) -> str:
    r2, n = metrics["r2"], metrics["n_obs"]
    lines = [f"O modelo explica {r2*100:.1f}% da variância de {y_label} (R² = {r2:.4f}, n = {n})."]

    sig, not_sig = [], []
    for _, row in coef_table.iterrows():
        var = str(row["variavel"])
        if var == "Intercept":
            continue
        (sig if row["p_valor"] < 0.05 else not_sig).append(row)

    for row in sig:
        var = str(row["variavel"])
        coef, p = row["coeficiente"], row["p_valor"]
        if ":" in var:
            parts = var.split(":")
            lines.append(
                f"A interação entre {_label(parts[0])} e {_label(parts[1])} "
                f"é estatisticamente significativa ({_p_str(p)}, β = {coef:.2f}), "
                f"indicando efeito moderador entre as variáveis."
            )
        else:
            direcao = "positivo" if coef > 0 else "negativo"
            lines.append(
                f"{_label(var)} apresenta efeito {direcao} significativo "
                f"({_p_str(p)}, β = {coef:.2f})."
            )

    for row in not_sig:
        var = str(row["variavel"])
        if ":" in var:
            continue
        lines.append(f"{_label(var)} não apresenta efeito estatisticamente significativo ({_p_str(row['p_valor'])}).")

    return " ".join(lines)


def _interpretar_vif(vif_table: pd.DataFrame) -> Tuple[str, str]:
    problematic = vif_table[vif_table["vif"] > VIF_MODERADO]
    if problematic.empty:
        return "ok", "Nenhuma variável com multicolinearidade detectada. Todos os VIF estão abaixo de 5."
    worst = problematic.sort_values("vif", ascending=False).iloc[0]
    nivel = "severo" if worst["vif"] > VIF_SEVERO else "moderado"
    return "warning", (
        f'"{_label(worst["variavel"])}" apresenta VIF {nivel} ({worst["vif"]:.1f}), '
        f'indicando possível multicolinearidade. '
        f'Considere remover esta variável e reexecutar o modelo.'
    )


def _interpretar_residuos(diag: dict) -> str:
    msgs = []
    if not diag["bp_homocedastic"]:
        msgs.append(
            f"Indícios de heterocedasticidade detectados "
            f"(Breusch-Pagan p = {diag['bp_pvalue']:.4f}). "
            f"Considere uso de erros-padrão robustos (HC3) ou transformação da variável dependente."
        )
    else:
        msgs.append("Resíduos apresentam variância homocedástica (Breusch-Pagan não rejeita H0).")

    if not diag["shapiro_normal"]:
        nota = " (teste aplicado em amostra de 5.000)" if not diag.get("shapiro_confiavel", True) else ""
        msgs.append(
            f"Resíduos não seguem distribuição normal "
            f"(Shapiro-Wilk p = {diag['shapiro_pvalue']:.4f}{nota}). "
            f"Com n > 5.000, o Teorema Central do Limite garante validade assintótica dos testes t."
        )
    else:
        msgs.append("Resíduos seguem distribuição aproximadamente normal (Shapiro-Wilk não rejeita H0).")

    return " ".join(msgs)


def _interpretar_shap(shap_summary: pd.DataFrame, y_label: str) -> str:
    if shap_summary.empty:
        return ""
    top = shap_summary.iloc[0]
    return (
        f'"{_label(top["variavel"])}" é o fator com maior impacto médio absoluto sobre '
        f'{y_label} (|SHAP| médio = {top["shap_mean_abs"]:.2f}). '
        f'O gráfico apresenta a importância relativa de cada variável calculada via '
        f'SHAP LinearExplainer (Lundberg & Lee, 2017).'
    )


# ── Estilo da tabela VIF com cores ────────────────────────────────────────────

def _style_vif(df: pd.DataFrame):
    def color_row(row):
        v = row["vif"]
        bg = "#ffcccc" if v > VIF_SEVERO else ("#fff3cd" if v > VIF_MODERADO else "#d4edda")
        return [f"background-color: {bg}"] * len(row)
    return df.style.apply(color_row, axis=1).format({"vif": "{:.3f}"})


def _style_coef(df: pd.DataFrame):
    def color_p(row):
        p = row["p_valor"]
        bg = "#d4edda" if p < 0.05 else "#f8f9fa"
        return [f"background-color: {bg}"] * len(row)
    return df.style.apply(color_p, axis=1).format({
        "coeficiente": "{:.4f}", "erro_padrao": "{:.4f}",
        "t_stat": "{:.4f}", "p_valor": "{:.4f}",
        "ic_inf_95": "{:.4f}", "ic_sup_95": "{:.4f}",
    })


# ── Navegação ─────────────────────────────────────────────────────────────────
ETAPAS = [
    "1. Seleção de Base",
    "2. Definição da Hipótese",
    "3. Modelagem e Diagnóstico",
    "4. Multicolinearidade (VIF)",
    "5. Diagnóstico de Resíduos",
    "6. Explicabilidade (SHAP)",
    "7. Relatório",
]

st.sidebar.title("E-XplainENADE")
st.sidebar.caption(f"Exec-ID: `{st.session_state.exec_id[:8]}...`")
etapa = st.sidebar.radio("Etapa", ETAPAS, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Seleção de Base
# ══════════════════════════════════════════════════════════════════════════════
if etapa == ETAPAS[0]:
    st.header("Etapa 1 — Seleção de Base")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1.1 Ano")
        st.markdown("**2021**")
        st.caption("Outros anos disponíveis após integração com ENADE-Time.")

    with col2:
        st.subheader("1.2 Curso")
        curso_sel = st.radio(
            "Curso",
            options=list(CURSO_OPTS.keys()),
            label_visibility="collapsed",
        )

    with col3:
        st.subheader("1.3 Tipo de IES")
        ies_sel = st.radio(
            "Tipo de IES",
            options=list(IES_OPTS.keys()),
            label_visibility="collapsed",
        )

    st.divider()

    if st.button("Carregar Base", type="primary"):
        grupos = CURSO_OPTS[curso_sel]
        with st.spinner("Carregando base de dados (~30 segundos)..."):
            df = loader.get_dataset(grupos=grupos)

        # Filtro de tipo de IES (pós-carga, CO_CATEGAD já está tipado)
        ies_filter = IES_OPTS[ies_sel]
        if ies_filter:
            df = df[df["CO_CATEGAD"].isin(ies_filter)].reset_index(drop=True)

        st.session_state.df = df
        st.session_state.filtros = {
            "ano": "2021", "curso": curso_sel, "tipo_ies": ies_sel,
            "grupos": grupos, "ies_filter": ies_filter,
        }

    if "df" in st.session_state:
        df = st.session_state.df
        st.success(f"Base carregada com {len(df):,} registros. Variáveis disponíveis: {len(df.columns)}.")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Média Nota Geral",    f"{df['NT_GER'].mean():.2f}")
        col_b.metric("Desvio Padrão",       f"{df['NT_GER'].std():.2f}")
        ausentes = df.isnull().mean().mul(100).round(1)
        col_c.metric("Máx. % Dados Ausentes", f"{ausentes.max():.1f}%")

        with st.expander("Ver amostra dos dados (10 primeiras linhas)"):
            st.dataframe(df.head(10))

        with st.expander("% de dados ausentes por variável (top 10)"):
            st.dataframe(
                ausentes[ausentes > 0].sort_values(ascending=False).head(10)
                .rename("% ausente").to_frame()
            )

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Definição Formal da Hipótese
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[1]:
    st.header("Etapa 2 — Definição Formal da Hipótese")

    if "df" not in st.session_state:
        st.warning("Carregue a base primeiro (Etapa 1).")
        st.stop()

    df = st.session_state.df

    # 2.1 Variável dependente
    st.subheader("2.1 Variável Dependente (Y)")
    y_label = st.selectbox("Selecione a nota a ser modelada", options=list(Y_OPTS.keys()))
    y_var = Y_OPTS[y_label]

    st.divider()

    # 2.2 Variáveis independentes
    st.subheader("2.2 Variáveis Independentes (X)")
    st.caption("Selecione as variáveis preditoras a incluir no modelo.")

    # Filtra variáveis que existem no DataFrame
    x_disponiveis = {lbl: cod for lbl, cod in X_OPTS.items() if cod in df.columns}

    x_selecionados_labels = []
    cols = st.columns(2)
    for i, (lbl, _) in enumerate(x_disponiveis.items()):
        default = lbl in DEFAULT_X
        if cols[i % 2].checkbox(lbl, value=default, key=f"x_{lbl}"):
            x_selecionados_labels.append(lbl)

    x_vars = [x_disponiveis[lbl] for lbl in x_selecionados_labels]

    st.divider()

    # 2.3 Interações
    st.subheader("2.3 Testar Interações?")
    interactions = []
    if len(x_vars) >= 2:
        pairs = list(itertools.combinations(x_selecionados_labels, 2))
        inter_cols = st.columns(min(len(pairs), 3))
        for i, (la, lb) in enumerate(pairs):
            ca, cb = x_disponiveis[la], x_disponiveis[lb]
            if inter_cols[i % len(inter_cols)].checkbox(f"{la} × {lb}", key=f"inter_{la}_{lb}"):
                interactions.append((ca, cb))
    else:
        st.info("Selecione ao menos 2 variáveis independentes para habilitar interações.")

    st.divider()

    if x_vars:
        st.info(f"**Fórmula prévia:** `{y_var} ~ {' + '.join(x_vars)}"
                + (f" + {' + '.join(f'{a}:{b}' for a,b in interactions)}`" if interactions else "`"))

    if st.button("Executar Modelagem", type="primary", disabled=(len(x_vars) == 0)):
        hyp = build_hypothesis(y_var, x_vars, interactions, df)
        with st.spinner("Executando regressão OLS, VIF, resíduos e SHAP..."):
            result     = fit_ols(hyp, df)
            metrics    = get_model_metrics(result)
            coef_table = get_coefficients_table(result)
            vif_table  = compute_vif(df, x_vars)
            diag       = run_diagnostics(result)
            shap_vals, feat_names = compute_shap(result, df, x_vars)
            shap_sum   = get_shap_summary(shap_vals, feat_names)

        st.session_state.update({
            "hypothesis":    hyp,
            "formula":       hyp.to_formula(),
            "y_label":       y_label,
            "model_result":  result,
            "model_metrics": metrics,
            "coef_table":    coef_table,
            "vif_table":     vif_table,
            "residuals_diag": diag,
            "shap_values":   shap_vals,
            "shap_summary":  shap_sum,
            "config": {
                "exec_id": st.session_state.exec_id,
                "formula": hyp.to_formula(),
                "filtros": st.session_state.get("filtros", {}),
            },
        })
        st.success("Modelagem concluída. Navegue pelas Etapas 3 a 6 para ver os resultados.")

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — Modelagem e Diagnóstico Automático
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[2]:
    st.header("Etapa 3 — Modelagem e Diagnóstico")

    if "model_result" not in st.session_state:
        st.warning("Execute a modelagem primeiro (Etapa 2).")
        st.stop()

    metrics    = st.session_state.model_metrics
    coef_table = st.session_state.coef_table
    y_label    = st.session_state.get("y_label", "Nota Geral")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²",             metrics["r2"])
    col2.metric("R² Ajustado",    metrics["r2_adj"])
    col3.metric("n Observações",  metrics["n_obs"])
    col4.metric("F p-valor",      f"{metrics['f_pvalue']:.2e}")

    st.divider()
    st.subheader("3.1 Tabela de Coeficientes")
    st.caption("Verde = efeito significativo (p < 0,05)")

    # Renomeia a coluna 'variavel' para nomes legíveis na exibição
    display_coef = coef_table.copy()
    display_coef["variavel"] = display_coef["variavel"].apply(
        lambda v: " × ".join(_label(p) for p in v.split(":")) if ":" in v else _label(v)
    )
    st.dataframe(_style_coef(display_coef), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3.2 Interpretação Automática")
    interp = _interpretar_modelo(coef_table, metrics, y_label)
    st.info(interp)

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — Multicolinearidade (VIF)
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[3]:
    st.header("Etapa 4 — Diagnóstico de Multicolinearidade (VIF)")

    if "vif_table" not in st.session_state:
        st.warning("Execute a modelagem primeiro (Etapa 2).")
        st.stop()

    vif   = st.session_state.vif_table
    df    = st.session_state.df

    st.caption("Verde = VIF ≤ 5 (ok) | Amarelo = VIF > 5 (moderado) | Vermelho = VIF > 10 (severo)")

    display_vif = vif.copy()
    display_vif["variavel"] = display_vif["variavel"].apply(_label)
    st.dataframe(_style_vif(display_vif), use_container_width=True, hide_index=True)

    st.divider()
    status, msg = _interpretar_vif(vif)
    if status == "ok":
        st.success(msg)
    else:
        st.warning(msg)
        remove_var = get_variable_to_remove(vif)
        if st.button(f"Remover '{_label(remove_var)}' e Reexecutar", type="primary"):
            hyp = st.session_state.hypothesis
            new_x = [v for v in hyp.x_vars if v != remove_var]
            if not new_x:
                st.error("Não é possível remover — nenhuma variável restaria no modelo.")
                st.stop()

            new_hyp = build_hypothesis(hyp.y, new_x, hyp.interactions, df)
            with st.spinner("Reexecutando modelo..."):
                result     = fit_ols(new_hyp, df)
                metrics    = get_model_metrics(result)
                coef_table = get_coefficients_table(result)
                vif_table  = compute_vif(df, new_x)
                diag       = run_diagnostics(result)
                shap_vals, feat_names = compute_shap(result, df, new_x)
                shap_sum   = get_shap_summary(shap_vals, feat_names)

            st.session_state.update({
                "hypothesis":    new_hyp,
                "formula":       new_hyp.to_formula(),
                "model_result":  result,
                "model_metrics": metrics,
                "coef_table":    coef_table,
                "vif_table":     vif_table,
                "residuals_diag": diag,
                "shap_values":   shap_vals,
                "shap_summary":  shap_sum,
            })
            st.success(f"Variável '{_label(remove_var)}' removida. Modelo reexecutado.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — Diagnóstico de Resíduos
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[4]:
    st.header("Etapa 5 — Diagnóstico de Resíduos")

    if "model_result" not in st.session_state:
        st.warning("Execute a modelagem primeiro (Etapa 2).")
        st.stop()

    import plotly.express as px

    diag = st.session_state.residuals_diag

    col1, col2 = st.columns(2)
    sw_status = "Normal" if diag["shapiro_normal"] else "Nao-normal"
    bp_status = "Homocedastico" if diag["bp_homocedastic"] else "Heterocedastico"
    col1.metric("Shapiro-Wilk p-valor", diag["shapiro_pvalue"],
                delta=sw_status, delta_color="normal" if diag["shapiro_normal"] else "inverse")
    col2.metric("Breusch-Pagan p-valor", diag["bp_pvalue"],
                delta=bp_status, delta_color="normal" if diag["bp_homocedastic"] else "inverse")

    st.divider()
    plot_data = get_residuals_plot_data(st.session_state.model_result)
    fig = px.scatter(
        plot_data, x="fitted", y="residual",
        labels={"fitted": "Valores Preditos", "residual": "Residuos"},
        title="Grafico de Residuos vs. Valores Preditos",
        opacity=0.4,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="y = 0")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Interpretação Automática")
    st.warning(_interpretar_residuos(diag))

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — Explicabilidade (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[5]:
    st.header("Etapa 6 — Explicabilidade (SHAP)")

    if "shap_summary" not in st.session_state:
        st.warning("Execute a modelagem primeiro (Etapa 2).")
        st.stop()

    import plotly.express as px

    shap_sum = st.session_state.shap_summary
    y_label  = st.session_state.get("y_label", "Nota Geral")

    display_shap = shap_sum.copy()
    display_shap["variavel"] = display_shap["variavel"].apply(
        lambda v: " × ".join(_label(p) for p in v.split(":")) if ":" in v else _label(v)
    )

    fig = px.bar(
        display_shap,
        x="shap_mean_abs", y="variavel",
        orientation="h",
        labels={"shap_mean_abs": "Impacto Medio |SHAP|", "variavel": "Variavel"},
        title=f"Importancia das Variaveis para {y_label} (SHAP Global)",
        color="shap_mean_abs",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Interpretação Automática")
    st.info(_interpretar_shap(st.session_state.shap_summary, y_label))

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 7 — Relatório PDF
# ══════════════════════════════════════════════════════════════════════════════
elif etapa == ETAPAS[6]:
    st.header("Etapa 7 — Relatório Técnico")

    required = ["model_result", "vif_table", "residuals_diag", "shap_summary", "coef_table"]
    missing  = [k for k in required if k not in st.session_state]

    if missing:
        st.warning(f"Conclua as etapas anteriores antes de gerar o relatório.")
        for k in missing:
            st.write(f"- {k} não encontrado na sessão.")
        st.stop()

    col1, col2 = st.columns(2)
    col1.metric("Exec-ID", st.session_state.exec_id[:16] + "...")
    col2.metric("Fórmula", st.session_state.get("formula", "N/A"))

    if st.button("Gerar Relatório PDF", type="primary"):
        with st.spinner("Gerando PDF..."):
            path = generate_report(
                session={
                    "exec_id":        st.session_state.exec_id,
                    "config":         st.session_state.get("config", {}),
                    "formula":        st.session_state.get("formula", ""),
                    "model_metrics":  st.session_state.model_metrics,
                    "vif_table":      st.session_state.vif_table,
                    "residuals_diag": st.session_state.residuals_diag,
                    "shap_summary":   st.session_state.shap_summary,
                }
            )
        st.success(f"Relatório gerado: `{path.name}`")
        with open(path, "rb") as f:
            st.download_button(
                label="Baixar PDF",
                data=f,
                file_name=path.name,
                mime="application/pdf",
                type="primary",
            )
