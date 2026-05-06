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

from config.variable_map import CODE_TO_LABEL, VALUE_LABELS
from modules import etl, loader
from modules.hypothesis import build_hypothesis
from modules.modeling import fit_ols, get_coefficients_table, get_model_metrics
from modules.multicollinearity import (
    compute_vif, has_multicollinearity, get_variable_to_remove,
    VIF_MODERADO, VIF_SEVERO,
)
from modules.residuals import get_residuals_plot_data, get_qqplot_data, run_diagnostics
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

# Anos: 2021 disponível. Os demais serão integrados via ENADE-Time (Lucas).
ANOS_OPTS = {"2021": True, "2019": False, "2018": False, "2017": False}

CURSO_OPTS = {
    "Todos":                    None,
    "Ciência da Computação":    [4004],
    "Engenharia de Software":   [4005],
    "Sistemas de Informação":   [4006],
}
IES_OPTS = {"Todas": None, "Pública": [1, 2, 3], "Privada": [4, 5]}

# Prefixos de colunas não-analíticas (excluídas da contagem "Variáveis disponíveis")
_PREFIXOS_NAO_ANALITICOS = ("DS_", "CO_RS_I", "NU_ITEM_", "TP_PR_", "TP_SFG_", "TP_SCE_")

def _n_vars_analiticas(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns
               if not any(c.startswith(p) for p in _PREFIXOS_NAO_ANALITICOS))

DEFAULT_X    = ["Renda Familiar", "Escolaridade da Mãe", "Horas de Estudo por Semana"]

# ── Cache de dados ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _carregar_dados(grupos_tuple, ies_filter_tuple):
    grupos = list(grupos_tuple) if grupos_tuple is not None else None
    df = loader.get_dataset(grupos=grupos)
    if ies_filter_tuple:
        df = df[df["CO_CATEGAD"].isin(list(ies_filter_tuple))].reset_index(drop=True)
    return df


# ── Exec-ID da sessão ─────────────────────────────────────────────────────────
if "exec_id" not in st.session_state:
    st.session_state.exec_id = str(uuid.uuid4())


# ── Funções auxiliares ────────────────────────────────────────────────────────

def _label(code: str) -> str:
    return CODE_TO_UI_LABEL.get(code, CODE_TO_LABEL.get(code, code))


def _p_fmt(p: float) -> str:
    """Formata p-valor em linguagem acessível."""
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


def _coef_table_display(coef_table: pd.DataFrame) -> pd.DataFrame:
    """Constrói tabela simplificada e legível para exibição na interface."""
    rows = []
    for _, row in coef_table.iterrows():
        var = str(row["variavel"])
        if var == "Intercept":
            continue
        label = " × ".join(_label(p) for p in var.split(":")) if ":" in var else _label(var)
        coef, p = row["coeficiente"], row["p_valor"]
        rows.append({
            "Fator":           label,
            "Efeito na Nota":  f"{coef:+.2f}",
            "Direção":         "↑ Aumenta" if coef > 0 else "↓ Diminui",
            "Confiável?":      "✅ Sim" if p < 0.05 else "❌ Não",
            "P-valor":         _p_fmt(p),
            "IC 95%":          f"[{row['ic_inf_95']:.2f} ; {row['ic_sup_95']:.2f}]",
        })
    return pd.DataFrame(rows)




# ── Funções de interpretação automática ──────────────────────────────────────

def _interpretar_modelo(coef_table: pd.DataFrame, metrics: dict, y_label: str) -> str:
    r2, n = metrics["r2"], metrics["n_obs"]
    linhas = [
        f"Com base em {n:,} estudantes do ENADE 2021, "
        f"o modelo consegue explicar **{r2*100:.1f}%** da variação nas notas de {y_label}.\n"
    ]

    sig, nao_sig = [], []
    for _, row in coef_table.iterrows():
        var = str(row["variavel"])
        if var == "Intercept":
            continue
        (sig if row["p_valor"] < 0.05 else nao_sig).append(row)

    if sig:
        linhas.append("**Fatores com efeito comprovado:**")
        for row in sig:
            var = str(row["variavel"])
            coef, p = row["coeficiente"], row["p_valor"]
            if ":" in var:
                parts = var.split(":")
                la, lb = _label(parts[0]), _label(parts[1])
                direcao = "moderam o efeito um do outro" if coef < 0 else "se reforçam mutuamente"
                linhas.append(
                    f"• **{la} × {lb}** (interação): os dois fatores {direcao}. "
                    f"Efeito combinado: {coef:+.2f} pontos (p {_p_fmt(p)})."
                )
            else:
                label = _label(var)
                if coef > 0:
                    direcao = f"quanto maior, maior a nota (+{coef:.2f} pontos por nível)"
                else:
                    direcao = f"quanto maior, menor a nota ({coef:.2f} pontos por nível)"
                linhas.append(f"• **{label}**: {direcao}. (p {_p_fmt(p)})")

    if nao_sig:
        nomes = ", ".join(_label(str(r["variavel"])) for r in nao_sig if ":" not in str(r["variavel"]))
        if nomes:
            linhas.append(f"\n**Sem efeito comprovado:** {nomes}.")

    return "\n".join(linhas)


def _interpretar_vif(vif_table: pd.DataFrame) -> Tuple[str, str]:
    prob = vif_table[vif_table["vif"] > VIF_MODERADO]
    if prob.empty:
        return "ok", (
            "✅ Nenhum problema de correlação entre variáveis detectado. "
            "Os coeficientes do modelo são confiáveis."
        )
    worst = prob.sort_values("vif", ascending=False).iloc[0]
    nivel = "alto" if worst["vif"] > VIF_SEVERO else "moderado"
    return "warning", (
        f"⚠️ **{_label(worst['variavel'])}** tem correlação {nivel} com as outras variáveis "
        f"(VIF = {worst['vif']:.1f}). "
        f"Isso pode tornar o coeficiente desta variável pouco confiável. "
        f"Recomenda-se removê-la do modelo."
    )


def _interpretar_residuos(diag: dict) -> str:
    partes = []

    # Normalidade
    if diag["shapiro_normal"]:
        partes.append(
            "✅ **Distribuição dos erros:** Os erros do modelo seguem um padrão normal — "
            "erros pequenos são mais comuns e erros grandes são raros. Isso é o esperado."
        )
    else:
        nota = " (avaliado em amostra de 5.000 estudantes)" if not diag.get("shapiro_confiavel", True) else ""
        partes.append(
            f"ℹ️ **Distribuição dos erros:** Os erros não seguem exatamente um padrão normal{nota}. "
            f"Com mais de 5.000 observações, isso tem impacto pequeno na validade dos resultados — "
            f"os testes estatísticos continuam sendo válidos."
        )

    # Homocedasticidade
    if diag["bp_homocedastic"]:
        partes.append(
            "✅ **Consistência dos erros:** O modelo erra de forma parecida para todos os grupos. "
            "Isso indica que os resultados são igualmente confiáveis para estudantes de diferentes perfis."
        )
    else:
        partes.append(
            f"⚠️ **Consistência dos erros:** O modelo erra mais para alguns grupos do que para outros "
            f"(p = {diag['bp_pvalue']:.4f}). "
            f"Os intervalos de confiança podem ser menos precisos para grupos extremos "
            f"(ex: estudantes de renda muito alta ou muito baixa)."
        )

    return "\n\n".join(partes)


def _interpretar_shap(shap_summary: pd.DataFrame, y_label: str) -> str:
    if shap_summary.empty:
        return ""
    top = shap_summary.iloc[0]
    second = shap_summary.iloc[1] if len(shap_summary) > 1 else None
    label_top = " × ".join(_label(p) for p in top["variavel"].split(":")) \
                if ":" in top["variavel"] else _label(top["variavel"])

    msg = (
        f"O fator que **mais influencia** a {y_label} é **{label_top}** "
        f"— a diferença entre estudantes com valores altos e baixos neste fator "
        f"representa, em média, {top['shap_mean_abs']:.1f} pontos na nota."
    )
    if second is not None:
        label_2 = " × ".join(_label(p) for p in second["variavel"].split(":")) \
                  if ":" in second["variavel"] else _label(second["variavel"])
        msg += (
            f" Em segundo lugar aparece **{label_2}** "
            f"(impacto médio de {second['shap_mean_abs']:.1f} pontos)."
        )
    msg += " O gráfico mostra todos os fatores ordenados do mais para o menos influente."
    return msg


# Estilo base: texto sempre escuro para garantir contraste em modo claro e escuro
_CELL = "color: #212529; font-size: 13px"


def _style_vif(df: pd.DataFrame):
    vif_col = "VIF" if "VIF" in df.columns else "vif"
    def color_row(row):
        v = row[vif_col]
        if v > VIF_SEVERO:
            bg, txt = "#c0392b", "#ffffff"   # vermelho forte, texto branco
        elif v > VIF_MODERADO:
            bg, txt = "#f39c12", "#212529"   # laranja, texto escuro
        else:
            bg, txt = "#27ae60", "#ffffff"   # verde forte, texto branco
        return [f"background-color: {bg}; color: {txt}; font-weight: bold"] * len(row)
    return df.style.apply(color_row, axis=1).format({vif_col: "{:.2f}"})


def _style_coef_display(df: pd.DataFrame):
    def color(row):
        ok = row["Confiável?"] == "✅ Sim"
        bg  = "#1e8449" if ok else "#616a6b"   # verde escuro / cinza
        return [f"background-color: {bg}; color: #ffffff; font-size: 13px"] * len(row)
    return df.style.apply(color, axis=1)


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

# Indicadores de progresso
_PROG = {
    "df":           "1. Seleção de Base",
    "model_result": "2-6. Modelagem executada",
    "shap_summary": "6. SHAP calculado",
}
for key, label in _PROG.items():
    icon = "✅" if key in st.session_state else "⬜"
    st.sidebar.caption(f"{icon} {label}")

st.sidebar.divider()
etapa = st.sidebar.radio("Navegar", ETAPAS, label_visibility="collapsed")
st.sidebar.divider()

if st.sidebar.button("🔄 Nova Análise", help="Limpa a sessão e reinicia do zero"):
    for key in list(st.session_state.keys()):
        if key != "exec_id":
            del st.session_state[key]
    st.session_state.exec_id = str(uuid.uuid4())
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Seleção de Base
# ══════════════════════════════════════════════════════════════════════════════
if etapa == ETAPAS[0]:
    st.header("Etapa 1 — Seleção de Base")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1.1 Ano")
        ano_sel = st.radio(
            "Ano",
            options=list(ANOS_OPTS.keys()),
            label_visibility="collapsed",
        )
        if not ANOS_OPTS[ano_sel]:
            st.caption(f"⚠️ {ano_sel} disponível após integração com ENADE-Time.")

    with col2:
        st.subheader("1.2 Curso")
        curso_sel = st.radio(
            "Curso",
            options=list(CURSO_OPTS.keys()),
            label_visibility="collapsed",
        )
        if curso_sel == "Engenharia de Software":
            st.caption("⚠️ Engenharia de Software tem poucos registros no ENADE 2021.")

    with col3:
        st.subheader("1.3 Tipo de IES")
        ies_sel = st.radio(
            "Tipo de IES",
            options=list(IES_OPTS.keys()),
            label_visibility="collapsed",
        )

    st.divider()

    ano_ok = ANOS_OPTS[ano_sel]
    if st.button("Carregar Base", type="primary", disabled=(not ano_ok)):
        grupos = CURSO_OPTS[curso_sel]
        ies_filter = IES_OPTS[ies_sel]
        with st.spinner("Carregando base de dados (primeira vez ~30s, depois instantâneo)..."):
            df = _carregar_dados(
                tuple(grupos) if grupos is not None else None,
                tuple(ies_filter) if ies_filter else (),
            )

        st.session_state.df = df
        st.session_state.filtros = {
            "ano": ano_sel, "curso": curso_sel, "tipo_ies": ies_sel,
            "grupos": grupos, "ies_filter": ies_filter,
        }

    if not ano_ok:
        st.info(f"Selecione o ano **2021** para carregar os dados disponíveis.")

    _MIN_REGISTROS = 50  # mínimo para OLS com múltiplas variáveis ser confiável

    if "df" in st.session_state:
        df = st.session_state.df
        n_analiticas = _n_vars_analiticas(df)

        if len(df) == 0:
            st.error("❌ Nenhum registro encontrado com os filtros selecionados. "
                     "Tente outra combinação de Curso e Tipo de IES.")
            st.stop()
        elif len(df) < _MIN_REGISTROS:
            st.warning(
                f"⚠️ Base carregada com apenas **{len(df)} registros**. "
                f"Isso é insuficiente para uma análise estatística confiável (mínimo recomendado: {_MIN_REGISTROS}). "
                f"Tente ampliar os filtros (ex: selecionar 'Todas' no Tipo de IES)."
            )
        else:
            st.success(f"Base carregada com **{len(df):,} registros**. Variáveis disponíveis: **{n_analiticas}**.")

        nt_vals = df["NT_GER"].dropna()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Média Nota Geral",  f"{nt_vals.mean():.2f}" if len(nt_vals) > 0 else "N/A")
        col_b.metric("Desvio Padrão",     f"{nt_vals.std():.2f}"  if len(nt_vals) > 1 else "N/A")
        col_c.metric("Mín / Máx",         f"{nt_vals.min():.1f} / {nt_vals.max():.1f}" if len(nt_vals) > 0 else "N/A")
        ausentes = df.isnull().mean().mul(100).round(1)
        col_d.metric("Máx. % Ausentes",   f"{ausentes.max():.1f}%")

        col_e, col_f = st.columns(2)
        turno_counts = df["CO_TURNO"].value_counts().rename(VALUE_LABELS.get("CO_TURNO", {}))
        cat_counts   = df["CO_CATEGAD"].value_counts().rename(VALUE_LABELS.get("CO_CATEGAD", {}))
        with col_e.expander("Distribuição por Turno"):
            st.dataframe(turno_counts.rename("n"))
        with col_f.expander("Distribuição por Tipo de IES"):
            st.dataframe(cat_counts.rename("n"))

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

    _MIN_REGISTROS = 50
    if len(df) < _MIN_REGISTROS:
        st.error(
            f"❌ A base tem apenas {len(df)} registro(s), insuficiente para modelagem. "
            f"Volte à **Etapa 1** e amplie os filtros."
        )
        st.stop()

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
        _cod_to_lbl = {v: k for k, v in x_disponiveis.items()}
        x_legivel = " + ".join(x_selecionados_labels)
        preview = f"{y_label}  ←  {x_legivel}"
        if interactions:
            inter_legivel = " + ".join(
                f"{_cod_to_lbl.get(a, a)} × {_cod_to_lbl.get(b, b)}"
                for a, b in interactions
            )
            preview += f"  +  Interação: {inter_legivel}"
        st.info(f"**Modelo que será executado:** {preview}")

    if st.button("Executar Modelagem", type="primary", disabled=(len(x_vars) == 0)):
        hyp = build_hypothesis(y_var, x_vars, interactions, df)
        try:
            with st.spinner("Executando: divisão treino/teste → regressão OLS → VIF → resíduos → SHAP..."):
                # ── Divisão treino/teste (80/20, seed fixo para reprodutibilidade) ──
                df_train = df.sample(frac=0.8, random_state=42)
                df_test  = df.drop(df_train.index)

                # ── Regressão OLS no conjunto completo (confirmatório) ────────────
                result     = fit_ols(hyp, df)
                metrics    = get_model_metrics(result)
                coef_table = get_coefficients_table(result)

                # ── R² no conjunto de teste (validação preditiva) ─────────────────
                try:
                    y_pred  = result.predict(df_test)
                    y_real  = df_test[hyp.y].dropna()
                    y_pred  = y_pred.reindex(y_real.index).dropna()
                    y_real  = y_real.reindex(y_pred.index)
                    ss_res  = ((y_real - y_pred) ** 2).sum()
                    ss_tot  = ((y_real - y_real.mean()) ** 2).sum()
                    r2_test = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                except Exception:
                    r2_test = float("nan")

                vif_table  = compute_vif(df, x_vars)
                diag       = run_diagnostics(result)
                shap_vals, feat_names = compute_shap(result, df, x_vars)
                shap_sum   = get_shap_summary(shap_vals, feat_names)

        except Exception as e:
            st.error(
                f"❌ Não foi possível executar a modelagem com os dados e variáveis selecionados.\n\n"
                f"**Causa provável:** poucos registros após os filtros aplicados, ou variáveis com "
                f"valores insuficientes para o modelo OLS.\n\n"
                f"**Sugestão:** volte à Etapa 1 e amplie os filtros, ou remova algumas variáveis.\n\n"
                f"Detalhe técnico: `{type(e).__name__}: {e}`"
            )
            st.stop()

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
            "r2_test":       round(r2_test, 4),
            "n_treino":      len(df_train),
            "n_teste":       len(df_test),
            "config": {
                "exec_id": st.session_state.exec_id,
                "formula": hyp.to_formula(),
                "filtros": st.session_state.get("filtros", {}),
            },
        })
        st.success("Modelagem concluída!")
        st.info("➡️ **Etapa 3** (coeficientes) → **Etapa 4** (VIF) → **Etapa 5** (resíduos) → **Etapa 6** (SHAP) → **Etapa 7** (relatório).")

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

    # Passos executados
    st.caption("✅ Divisão treino/teste  ✅ Regressão OLS  ✅ VIF  ✅ Normalidade/Homocedasticidade  ✅ SHAP")
    st.divider()

    # Métricas principais
    r2_pct    = metrics["r2"] * 100
    r2_test   = st.session_state.get("r2_test", None)
    n_treino  = st.session_state.get("n_treino", metrics["n_obs"])
    n_teste   = st.session_state.get("n_teste", 0)

    col0, col1, col2, col3 = st.columns(4)
    col0.metric(
        label=f"Poder Explicativo (R²) — {y_label}",
        value=f"{r2_pct:.1f}%",
        help="Percentual da variação nas notas explicado pelo modelo. Calculado sobre o conjunto completo.",
    )
    col1.metric("R² Ajustado", f"{metrics['r2_adj']:.4f}",
                help="Penaliza variáveis desnecessárias. Próximo do R² = modelo parcimonioso.")
    col2.metric(
        "R² no Conjunto de Teste",
        f"{r2_test*100:.1f}%" if r2_test is not None and r2_test == r2_test else "N/A",
        help=f"Poder preditivo estimado em {n_teste:,} estudantes não usados no ajuste (20% da base).",
    )
    col3.metric("Estudantes no Treino / Teste", f"{n_treino:,} / {n_teste:,}",
                help="Divisão 80/20 com seed fixo (42) para reprodutibilidade.")

    st.divider()
    st.subheader("3.1 Efeito de cada fator na nota")
    st.caption(
        "**Verde** = efeito comprovado estatisticamente (p < 0,05)  |  "
        "**↑ Aumenta** = fator está associado a notas mais altas  |  "
        "**IC 95%** = intervalo onde o efeito verdadeiro está com 95% de confiança"
    )

    st.caption(
        "🟢 Verde escuro = efeito comprovado (p < 0,05)  |  "
        "⬛ Cinza = sem evidência suficiente"
    )
    st.dataframe(
        _style_coef_display(_coef_table_display(coef_table)),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("ℹ️ Como ler esta tabela?"):
        st.markdown("""
- **Efeito na Nota**: quanto a nota muda, em média, a cada nível a mais nesta variável
- **Direção**: se o fator aumenta (↑) ou diminui (↓) a nota
- **Confiável?**: ✅ = efeito improvável de ser acaso | ❌ = sem evidência suficiente
- **P-valor**: probabilidade de o efeito ser pura coincidência. Abaixo de 0,05 é considerado confiável
- **IC 95%**: intervalo onde o efeito verdadeiro está em 95% das análises repetidas com os mesmos dados
        """)

    st.divider()
    st.subheader("3.2 O que os resultados dizem?")
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

    with st.expander("ℹ️ O que é esta análise?"):
        st.markdown("""
**Análise de Correlação entre Variáveis (VIF)**

Quando duas variáveis do modelo são muito parecidas entre si (ex: renda e escolaridade da mãe andam juntas),
os coeficientes ficam instáveis e difíceis de confiar. Esta análise detecta isso.

| Cor | VIF | Significado |
|---|---|---|
| 🟢 Verde | ≤ 5 | Sem problema |
| 🟡 Amarelo | 5 a 10 | Atenção — correlação moderada |
| 🔴 Vermelho | > 10 | Problema — considere remover a variável |
        """)

    st.caption("🟢 Verde = sem problema (VIF ≤ 5)  |  🟠 Laranja = atenção (VIF 5–10)  |  🔴 Vermelho = problema (VIF > 10)")
    display_vif = vif.copy()
    display_vif["variavel"] = display_vif["variavel"].apply(_label)
    display_vif = display_vif.rename(columns={"variavel": "Fator", "vif": "VIF", "alerta": "Situacao"})
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
    st.header("Etapa 5 — Qualidade do Modelo (Diagnóstico de Resíduos)")

    if "model_result" not in st.session_state:
        st.warning("Execute a modelagem primeiro (Etapa 2).")
        st.stop()

    import plotly.express as px

    diag = st.session_state.residuals_diag

    with st.expander("ℹ️ O que é esta análise?"):
        st.markdown("""
Esta etapa verifica se o modelo se comporta de forma correta e honesta.

**Distribuição dos erros** (Shapiro-Wilk): O modelo erra de forma equilibrada?
Erros pequenos deveriam ser mais comuns que erros grandes. Se não for assim, os intervalos de confiança podem estar incorretos.

**Consistência dos erros** (Breusch-Pagan): O modelo erra de forma parecida para todos os perfis de estudante?
Se errar muito para uns e pouco para outros, os resultados são menos confiáveis para grupos extremos.

Os **gráficos** ajudam a visualizar: no gráfico da esquerda, os pontos deveriam ficar espalhados aleatoriamente em volta da linha vermelha. No gráfico da direita (QQ-plot), os pontos deveriam seguir a linha diagonal.
        """)

    col1, col2 = st.columns(2)
    sw_label = "✅ Normal" if diag["shapiro_normal"] else "⚠️ Não-normal"
    bp_label = "✅ Consistente" if diag["bp_homocedastic"] else "⚠️ Inconsistente"
    col1.metric("Distribuição dos Erros", sw_label,
                help="Teste de Shapiro-Wilk. ✅ = erros seguem padrão normal.")
    col2.metric("Consistência dos Erros", bp_label,
                help="Teste de Breusch-Pagan. ✅ = erros são homogêneos entre grupos.")

    st.divider()
    col_g1, col_g2 = st.columns(2)

    plot_data = get_residuals_plot_data(st.session_state.model_result)
    fig_res = px.scatter(
        plot_data, x="fitted", y="residual",
        labels={"fitted": "Valores Preditos", "residual": "Residuos"},
        title="Residuos vs. Valores Preditos",
        opacity=0.4,
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    col_g1.plotly_chart(fig_res, use_container_width=True)

    qq_data = get_qqplot_data(st.session_state.model_result)
    fig_qq = px.scatter(
        qq_data, x="theoretical", y="sample",
        labels={"theoretical": "Quantis Teóricos (Normal)", "sample": "Quantis Amostrais"},
        title="QQ-Plot dos Resíduos",
        opacity=0.4,
    )
    fig_qq.add_scatter(
        x=qq_data["theoretical"], y=qq_data["line_y"],
        mode="lines", line=dict(color="red", dash="dash"), name="Referência Normal",
    )
    col_g2.plotly_chart(fig_qq, use_container_width=True)

    st.divider()
    st.subheader("O que os resultados indicam?")
    interp_res = _interpretar_residuos(diag)
    # Usa verde se ambos os testes passaram, amarelo se algum falhou
    if diag["shapiro_normal"] and diag["bp_homocedastic"]:
        st.success(interp_res)
    elif diag["shapiro_normal"] or diag["bp_homocedastic"]:
        st.warning(interp_res)
    else:
        st.info(interp_res)

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

    st.markdown(
        "A análise SHAP revela **quais fatores mais influenciam** a nota e o **quanto** "
        "cada um contribui para as diferenças entre os estudantes."
    )

    if st.button("Ver Explicabilidade", type="primary"):
        st.session_state["shap_visivel"] = True

    if not st.session_state.get("shap_visivel", False):
        st.info("Clique em **Ver Explicabilidade** para exibir o gráfico e a interpretação.")
        st.stop()

    st.info(
        "O gráfico abaixo mostra **quanto cada fator influencia** a nota. "
        "Quanto maior a barra, maior o peso desse fator para explicar as diferenças entre estudantes."
    )

    display_shap = shap_sum.copy()
    display_shap["variavel"] = display_shap["variavel"].apply(
        lambda v: " × ".join(_label(p) for p in v.split(":")) if ":" in v else _label(v)
    )
    display_shap = display_shap.rename(columns={"shap_mean_abs": "Influência média na nota"})

    fig = px.bar(
        display_shap,
        x="Influência média na nota", y="variavel",
        orientation="h",
        labels={"variavel": "Fator"},
        title=f"Importância dos Fatores para a {y_label}",
        color="Influência média na nota",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Como ler este gráfico?"):
        st.markdown("""
- **Barra maior** = esse fator faz mais diferença na nota
- O valor indica, em média, **quantos pontos** de variação na nota são atribuíveis a esse fator
- A ordem vai do **mais influente** (topo) ao **menos influente** (base)
- Isso não diz a direção do efeito (se aumenta ou diminui) — para isso, veja a Etapa 3
        """)

    st.divider()
    st.subheader("O que os números dizem?")
    st.success(_interpretar_shap(st.session_state.shap_summary, y_label))

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
                    "exec_id":               st.session_state.exec_id,
                    "config":                st.session_state.get("config", {}),
                    "formula":               st.session_state.get("formula", ""),
                    "model_metrics":         st.session_state.model_metrics,
                    "coef_table":            st.session_state.coef_table,
                    "vif_table":             st.session_state.vif_table,
                    "residuals_diag":        st.session_state.residuals_diag,
                    "shap_summary":          st.session_state.shap_summary,
                    "y_label":               st.session_state.get("y_label", "Nota Geral"),
                    "interpretacao_modelo":  _interpretar_modelo(
                                                st.session_state.coef_table,
                                                st.session_state.model_metrics,
                                                st.session_state.get("y_label", "Nota Geral"),
                                             ),
                    "interpretacao_vif":     _interpretar_vif(st.session_state.vif_table)[1],
                    "interpretacao_residuos": _interpretar_residuos(st.session_state.residuals_diag),
                    "interpretacao_shap":    _interpretar_shap(
                                                st.session_state.shap_summary,
                                                st.session_state.get("y_label", "Nota Geral"),
                                             ),
                    "plot_data":             get_residuals_plot_data(st.session_state.model_result),
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
