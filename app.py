"""
Camada 9 — Interface e orquestração (Streamlit).

Layout linear: todas as 7 etapas numa única página vertical.
Cada seção aparece progressivamente conforme os pré-requisitos são atendidos:
  - Etapa 1: sempre visível
  - Etapa 2: aparece após base carregada
  - Etapas 3–7: aparecem após modelagem executada
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
    initial_sidebar_state="collapsed",
)

# ── CSS Global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Layout base ─────────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1160px !important;
}

/* ── Sistema de sombras (variáveis reutilizáveis via inline style) ───── */
/* shadow-sm  : 0 1px 3px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.04)  */
/* shadow-md  : 0 2px 6px rgba(15,23,42,.07), 0 8px 24px rgba(15,23,42,.06)  */

/* ── Cards Streamlit (st.container border=True) ──────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.04) !important;
    background: #ffffff !important;
    padding: 4px 8px !important;
}

/* ── Hero ────────────────────────────────────────────────────────────── */
.exn-hero {
    border-radius: 24px;
    border: 1px solid #1e293b;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #2d1b69 100%);
    color: white;
    padding: 30px 36px;
    margin-bottom: 8px;
    box-shadow: 0 8px 32px rgba(15,23,42,.18);
}
.exn-hero h1 {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;
    margin: 0 0 8px 0; color: white; line-height: 1.2;
}
.exn-hero p {
    color: #cbd5e1; font-size: 14px; line-height: 1.65;
    margin: 0 0 22px 0; max-width: 640px;
}
.exn-meta-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
}
.exn-meta-card {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.07);
    padding: 14px 16px;
}
.exn-meta-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-weight: 600; }
.exn-meta-value { font-size: 15px; font-weight: 700; margin-top: 5px; color: white; }
.exn-meta-sub   { font-size: 11px; color: #64748b; margin-top: 3px; }

/* ── Etapa header ────────────────────────────────────────────────────── */
.exn-step {
    display: flex; align-items: center; gap: 16px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #7c3aed;
    background: #ffffff;
    padding: 16px 22px;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 4px 16px rgba(15,23,42,.04);
    margin-bottom: 22px;
}
.exn-step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: #7c3aed; color: white; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 15px;
    box-shadow: 0 2px 8px rgba(124,58,237,.35);
}
.exn-step-title { font-size: 17px; font-weight: 700; color: #1e1b4b; line-height: 1.2; }
.exn-step-desc  { font-size: 13px; color: #7c3aed; margin-top: 3px; font-weight: 500; }

/* ── Pills ───────────────────────────────────────────────────────────── */
.exn-pill {
    display: inline-flex; align-items: center; border-radius: 9999px;
    padding: 3px 11px; font-size: 12px; font-weight: 600;
    letter-spacing: 0.01em; white-space: nowrap;
}
.pill-violet { border: 1px solid #ddd6fe; background: #f5f3ff; color: #6d28d9; }
.pill-blue   { border: 1px solid #bfdbfe; background: #eff6ff; color: #1e40af; }
.pill-green  { border: 1px solid #bbf7d0; background: #f0fdf4; color: #166534; }
.pill-amber  { border: 1px solid #fde68a; background: #fffbeb; color: #92400e; }
.pill-fuchsia{ border: 1px solid #f5d0fe; background: #fdf4ff; color: #86198f; }
.pill-slate  { border: 1px solid #cbd5e1; background: #f1f5f9; color: #475569; }

/* ── Filter label (dentro de st.container) ───────────────────────────── */
.exn-filter-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #7c3aed; margin-bottom: 10px;
    padding-bottom: 8px; border-bottom: 1px solid #ede9fe;
}

/* ── Var group header ────────────────────────────────────────────────── */
.exn-var-group-header {
    font-size: 13px; font-weight: 600; color: #475569;
    margin-bottom: 8px; margin-top: 4px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Section divider ─────────────────────────────────────────────────── */
.exn-section-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 40px 0 36px 0;
}

/* ── Expanders ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.04) !important;
}

/* ── Streamlit alerts ────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ── Streamlit dataframe ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Oculta sidebar e seta ───────────────────────────────────────────── */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Remove padding extra do block-container interno ─────────────────── */
.main .block-container > div:first-child { padding-top: 0 !important; }

</style>
""", unsafe_allow_html=True)

# ── Constantes de UI ──────────────────────────────────────────────────────────

Y_OPTS = {
    "Nota Geral":            "NT_GER",
    "Formação Geral":        "NT_FG",
    "Componente Específico": "NT_CE",
}

X_OPTS = {
    "Renda Familiar":                "QE_RENDA",
    "Escolaridade da Mãe":           "QE_ESC_MAE",
    "Escolaridade do Pai":           "QE_ESC_PAI",
    "Tipo de Escola (EM)":           "QE_TIPO_EM",
    "Horas de Estudo por Semana":    "QE_HORAS_ESTUDO",
    "Situação de Trabalho":          "QE_TRABALHO",
    "Familiar com Ens. Superior":    "QE_FAM_SUPERIOR",
    "Sexo":                          "TP_SEXO",
    "Turno da Graduação":            "CO_TURNO",
    "Tipo de IES (Pública/Privada)": "TP_CATEGAD_BIN",
    "Ingresso por Ação Afirmativa":  "QE_ACAO_AFIRM",
}
CODE_TO_UI_LABEL = {v: k for k, v in X_OPTS.items()}
CODE_TO_UI_LABEL.update({v: k for k, v in Y_OPTS.items()})

ANOS_OPTS = {"2021": True, "2019": False, "2018": False, "2017": False}

CURSO_OPTS = {
    "Todos":                    None,
    "Ciência da Computação":    [4004],
    "Engenharia de Software":   [4005],
    "Sistemas de Informação":   [4006],
}
IES_OPTS = {"Todas": None, "Pública": [1, 2, 3], "Privada": [4, 5]}

X_CATEGORIAS = {
    "Socioeconômicas": {
        "pill": "pill-blue",
        "vars": [
            "Renda Familiar", "Escolaridade da Mãe", "Escolaridade do Pai",
            "Familiar com Ens. Superior", "Situação de Trabalho",
            "Ingresso por Ação Afirmativa",
        ],
    },
    "Acadêmicas": {
        "pill": "pill-green",
        "vars": ["Horas de Estudo por Semana", "Tipo de Escola (EM)"],
    },
    "Demográficas": {
        "pill": "pill-fuchsia",
        "vars": ["Sexo"],
    },
    "Institucionais": {
        "pill": "pill-amber",
        "vars": ["Turno da Graduação", "Tipo de IES (Pública/Privada)"],
    },
}

_PREFIXOS_NAO_ANALITICOS = ("DS_", "CO_RS_I", "NU_ITEM_", "TP_PR_", "TP_SFG_", "TP_SCE_")
_MIN_REGISTROS = 50

def _n_vars_analiticas(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns
               if not any(c.startswith(p) for p in _PREFIXOS_NAO_ANALITICOS))

DEFAULT_X = ["Renda Familiar", "Escolaridade da Mãe", "Horas de Estudo por Semana"]

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
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")

def _coef_table_display(coef_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coef_table.iterrows():
        var = str(row["variavel"])
        if var == "Intercept":
            continue
        lbl = " × ".join(_label(p) for p in var.split(":")) if ":" in var else _label(var)
        coef, p = row["coeficiente"], row["p_valor"]
        rows.append({
            "Fator":          lbl,
            "Efeito na Nota": f"{coef:+.2f}",
            "Direção":        "↑ Aumenta" if coef > 0 else "↓ Diminui",
            "Confiável?":     "✅ Sim" if p < 0.05 else "❌ Não",
            "P-valor":        _p_fmt(p),
            "IC 95%":         f"[{row['ic_inf_95']:.2f} ; {row['ic_sup_95']:.2f}]",
        })
    return pd.DataFrame(rows)


# ── Funções de interpretação ──────────────────────────────────────────────────

def _interpretar_modelo(coef_table: pd.DataFrame, metrics: dict, y_label: str) -> str:
    r2, r2_adj, n = metrics["r2"], metrics["r2_adj"], metrics["n_obs"]
    f_p = metrics.get("f_pvalue", 0)

    linhas = [
        f"O modelo de regressão linear múltipla ajustado com {n:,} estudantes do ENADE 2021 "
        f"apresenta R² = {r2:.4f} (R² ajustado = {r2_adj:.4f}), explicando "
        f"**{r2*100:.1f}%** da variância nas notas de {y_label}. "
        f"O modelo é globalmente significativo (F, p {_p_fmt(f_p)}).\n"
    ]

    sig_main, sig_inter, nao_sig = [], [], []
    for _, row in coef_table.iterrows():
        var = str(row["variavel"])
        if var == "Intercept":
            continue
        if row["p_valor"] < 0.05:
            (sig_inter if ":" in var else sig_main).append(row)
        else:
            nao_sig.append(row)

    if sig_main:
        linhas.append("**Efeitos principais estatisticamente significativos (p < 0,05):**")
        for row in sig_main:
            coef, p = row["coeficiente"], row["p_valor"]
            lbl = _label(str(row["variavel"]))
            sinal = "positivo" if coef > 0 else "negativo"
            movimento = "aumento" if coef > 0 else "redução"
            linhas.append(
                f"• **{lbl}** apresenta efeito {sinal} significativo "
                f"(β = {coef:+.2f}, p {_p_fmt(p)}). "
                f"Cada nível adicional está associado a um {movimento} médio de "
                f"{abs(coef):.2f} pontos na {y_label}."
            )

    if sig_inter:
        linhas.append("\n**Interações estatisticamente significativas:**")
        for row in sig_inter:
            coef, p = row["coeficiente"], row["p_valor"]
            parts = str(row["variavel"]).split(":")
            la, lb = _label(parts[0]), _label(parts[1])
            if coef < 0:
                linhas.append(
                    f"• O efeito de **{la}** é moderado por **{lb}** "
                    f"(interação significativa, β = {coef:+.2f}, p {_p_fmt(p)}). "
                    f"O sinal negativo indica atenuação mútua dos efeitos individuais."
                )
            else:
                linhas.append(
                    f"• O efeito de **{la}** é amplificado por **{lb}** "
                    f"(interação significativa, β = {coef:+.2f}, p {_p_fmt(p)}). "
                    f"Os dois fatores se reforçam mutuamente."
                )

    if nao_sig:
        nomes = ", ".join(
            _label(str(r["variavel"])) for r in nao_sig
            if ":" not in str(r["variavel"])
        )
        if nomes:
            linhas.append(
                f"\n**Sem efeito estatisticamente significativo (p ≥ 0,05):** {nomes}. "
                f"Esses fatores não apresentaram associação comprovada com a {y_label} "
                f"nesta amostra."
            )

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
    from modules.residuals import _SHAPIRO_MAX_N
    partes = []
    if diag["shapiro_normal"]:
        partes.append(
            "✅ **Distribuição dos erros:** Os erros do modelo seguem um padrão normal — "
            "erros pequenos são mais comuns e erros grandes são raros. Isso é o esperado."
        )
    else:
        nota = (f" (avaliado em amostra de {_SHAPIRO_MAX_N:,} estudantes)"
                if not diag.get("shapiro_confiavel", True) else "")
        partes.append(
            f"ℹ️ **Distribuição dos erros:** Os erros não seguem exatamente um padrão normal{nota}. "
            f"Com mais de {_SHAPIRO_MAX_N:,} observações, isso tem impacto pequeno na validade dos resultados."
        )
    if diag["bp_homocedastic"]:
        partes.append(
            "✅ **Consistência dos erros:** O modelo erra de forma parecida para todos os grupos. "
            "Isso indica que os resultados são igualmente confiáveis para diferentes perfis."
        )
    else:
        partes.append(
            f"⚠️ **Consistência dos erros:** O modelo erra mais para alguns grupos do que para outros "
            f"(p {_p_fmt(diag['bp_pvalue'])}). "
            f"Os intervalos de confiança podem ser menos precisos para grupos extremos."
        )
    return "\n\n".join(partes)


def _interpretar_shap(shap_summary: pd.DataFrame, y_label: str) -> str:
    if shap_summary.empty:
        return ""
    top = shap_summary.iloc[0]
    second = shap_summary.iloc[1] if len(shap_summary) > 1 else None
    label_top = (" × ".join(_label(p) for p in top["variavel"].split(":"))
                 if ":" in top["variavel"] else _label(top["variavel"]))
    msg = (
        f"O fator que **mais influencia** a {y_label} é **{label_top}** "
        f"— a diferença entre estudantes com valores altos e baixos neste fator "
        f"representa, em média, {top['shap_mean_abs']:.1f} pontos na nota."
    )
    if second is not None:
        label_2 = (" × ".join(_label(p) for p in second["variavel"].split(":"))
                   if ":" in second["variavel"] else _label(second["variavel"]))
        msg += (f" Em segundo lugar aparece **{label_2}** "
                f"(impacto médio de {second['shap_mean_abs']:.1f} pontos).")
    msg += " O gráfico mostra todos os fatores ordenados do mais para o menos influente."
    return msg


# ── Estilos de tabela ─────────────────────────────────────────────────────────

def _style_vif(df: pd.DataFrame):
    vif_col = "VIF" if "VIF" in df.columns else "vif"
    def color_row(row):
        v = row[vif_col]
        if v > VIF_SEVERO:
            bg, txt = "#c0392b", "#ffffff"
        elif v > VIF_MODERADO:
            bg, txt = "#f39c12", "#212529"
        else:
            bg, txt = "#27ae60", "#ffffff"
        return [f"background-color: {bg}; color: {txt}; font-weight: bold"] * len(row)
    return df.style.apply(color_row, axis=1).format({vif_col: "{:.2f}"})


def _style_coef_display(df: pd.DataFrame):
    def color(row):
        ok = row["Confiável?"] == "✅ Sim"
        bg = "#1e8449" if ok else "#616a6b"
        return [f"background-color: {bg}; color: #ffffff; font-size: 13px"] * len(row)
    return df.style.apply(color, axis=1)


# ── Componentes visuais ───────────────────────────────────────────────────────

def _hero():
    exec_id_short = st.session_state.exec_id[:8]
    df_loaded  = "df" in st.session_state
    model_done = "model_result" in st.session_state

    status_badge = (
        '<span class="exn-pill" style="background:rgba(16,185,129,0.18);'
        'border:1px solid rgba(16,185,129,0.35);color:#6ee7b7;">✓ Base carregada</span>'
        if df_loaded else
        '<span class="exn-pill" style="background:rgba(255,255,255,0.08);'
        'border:1px solid rgba(255,255,255,0.15);color:#94a3b8;">Aguardando dados</span>'
    )
    model_badge = (
        '<span class="exn-pill" style="background:rgba(139,92,246,0.22);'
        'border:1px solid rgba(139,92,246,0.38);color:#c4b5fd;">✓ Modelo pronto</span>'
        if model_done else ""
    )
    st.markdown(f"""
    <div class="exn-hero">
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
        <span class="exn-pill" style="background:rgba(255,255,255,0.1);
              border:1px solid rgba(255,255,255,0.15);color:#e2e8f0;">
          Hipóteses e Modelagem Explicável
        </span>
        {status_badge}
        {model_badge}
      </div>
      <h1>E-XplainENADE</h1>
      <p>Modelagem estatística multivariada explicável dos microdados ENADE.
         Defina hipóteses, ajuste modelos OLS, diagnostique pressupostos
         e interprete resultados com SHAP — reprodutível e rastreável.</p>
      <div class="exn-meta-grid">
        <div class="exn-meta-card">
          <div class="exn-meta-label">Método</div>
          <div class="exn-meta-value">Regressão OLS</div>
          <div class="exn-meta-sub">confirmatório · statsmodels</div>
        </div>
        <div class="exn-meta-card">
          <div class="exn-meta-label">Explicabilidade</div>
          <div class="exn-meta-value">SHAP Linear</div>
          <div class="exn-meta-sub">Lundberg &amp; Lee, 2017</div>
        </div>
        <div class="exn-meta-card">
          <div class="exn-meta-label">Exec-ID</div>
          <div class="exn-meta-value">{exec_id_short}…</div>
          <div class="exn-meta-sub">seed 42 · reprodutível</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _etapa_header(numero: int, titulo: str, descricao: str = ""):
    desc_html = f"<div class='exn-step-desc'>{descricao}</div>" if descricao else ""
    st.markdown(f"""
    <div class="exn-step">
      <div class="exn-step-num">{numero}</div>
      <div style="flex:1;">
        <div class="exn-step-title">{titulo}</div>
        {desc_html}
      </div>
      <span class="exn-pill pill-violet">E-XplainENADE</span>
    </div>
    """, unsafe_allow_html=True)


def _section_divider():
    st.markdown('<hr class="exn-section-divider">', unsafe_allow_html=True)


def _metric_row(items: list):
    """items: [(valor, label, tipo)] — tipo in {dark, violet, blue, slate, green}"""
    _shadow = "0 1px 3px rgba(15,23,42,.07), 0 6px 20px rgba(15,23,42,.05)"
    color_map = {
        "dark":   ("#0f172a", "white",   "#64748b", ""),
        "violet": ("#f5f3ff", "#3b0764", "#7c3aed", "border:1px solid #ddd6fe;"),
        "blue":   ("#eff6ff", "#1e3a8a", "#3b82f6", "border:1px solid #bfdbfe;"),
        "slate":  ("#ffffff", "#1e293b", "#64748b", "border:1px solid #e2e8f0;"),
        "green":  ("#f0fdf4", "#14532d", "#16a34a", "border:1px solid #bbf7d0;"),
    }
    cols = st.columns(len(items), gap="small")
    for col, (valor, lbl, tipo) in zip(cols, items):
        bg, txt, lbl_c, border = color_map.get(tipo, color_map["slate"])
        col.markdown(f"""
        <div style="border-radius:16px;background:{bg};padding:18px 20px;
                    {border}box-shadow:{_shadow};">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:.09em;
                      font-weight:700;color:{lbl_c};margin-bottom:8px;">{lbl}</div>
          <div style="font-size:22px;font-weight:800;color:{txt};
                      letter-spacing:-0.01em;line-height:1;">{valor}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO + botão Nova Análise
# ══════════════════════════════════════════════════════════════════════════════
_hero()

_, _col_reset = st.columns([5, 1])
with _col_reset:
    if st.button("🔄 Nova Análise", use_container_width=True,
                 help="Limpa a sessão e reinicia do zero"):
        for key in list(st.session_state.keys()):
            if key != "exec_id":
                del st.session_state[key]
        st.session_state.exec_id = str(uuid.uuid4())
        st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Seleção de Base  (sempre visível)
# ══════════════════════════════════════════════════════════════════════════════
_etapa_header(1, "Seleção de Base",
              "Escolha o recorte dos microdados ENADE 2021 a ser analisado")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<div class="exn-filter-label">📅 Ano</div>', unsafe_allow_html=True)
        ano_sel = st.radio("Ano", options=list(ANOS_OPTS.keys()),
                           label_visibility="collapsed")
        if not ANOS_OPTS[ano_sel]:
            st.caption(f"⚠️ {ano_sel} disponível após integração com ENADE-Time.")

with col2:
    with st.container(border=True):
        st.markdown('<div class="exn-filter-label">🎓 Curso</div>', unsafe_allow_html=True)
        curso_sel = st.radio("Curso", options=list(CURSO_OPTS.keys()),
                             label_visibility="collapsed")
        if curso_sel == "Engenharia de Software":
            st.caption("⚠️ Poucos registros no ENADE 2021.")

with col3:
    with st.container(border=True):
        st.markdown('<div class="exn-filter-label">🏛️ Tipo de IES</div>', unsafe_allow_html=True)
        ies_sel = st.radio("Tipo de IES", options=list(IES_OPTS.keys()),
                           label_visibility="collapsed")

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

ano_ok = ANOS_OPTS[ano_sel]
if st.button("Carregar Base", type="primary", disabled=(not ano_ok)):
    grupos     = CURSO_OPTS[curso_sel]
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
    for k in ["model_result", "model_metrics", "coef_table", "vif_table",
              "residuals_diag", "shap_values", "shap_summary", "hypothesis",
              "formula", "y_label", "r2_test", "n_treino", "n_teste",
              "config", "shap_visivel"]:
        st.session_state.pop(k, None)

if not ano_ok:
    st.info("Selecione o ano **2021** para carregar os dados disponíveis.")

if "df" in st.session_state:
    df = st.session_state.df
    n_analiticas = _n_vars_analiticas(df)

    if len(df) == 0:
        st.error("❌ Nenhum registro encontrado com os filtros selecionados.")
        st.stop()
    elif len(df) < _MIN_REGISTROS:
        st.warning(
            f"⚠️ Base carregada com apenas **{len(df)} registros** — insuficiente para "
            f"análise estatística confiável (mínimo: {_MIN_REGISTROS}). Amplie os filtros."
        )
    else:
        st.success(f"✅ Base carregada com **{len(df):,} registros** · "
                   f"**{n_analiticas}** variáveis analíticas disponíveis.")

    nt_vals  = df["NT_GER"].dropna()
    ausentes = df.isnull().mean().mul(100).round(1)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _metric_row([
        (f"{len(df):,}",
         "Registros na base", "dark"),
        (f"{nt_vals.mean():.2f}" if len(nt_vals) > 0 else "N/A",
         "Média Nota Geral", "violet"),
        (f"{nt_vals.std():.2f}" if len(nt_vals) > 1 else "N/A",
         "Desvio Padrão", "blue"),
        (f"{ausentes.max():.1f}%",
         "Máx. % Ausentes", "slate"),
    ])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_e, col_f = st.columns(2, gap="medium")
    turno_counts = df["CO_TURNO"].value_counts().rename(VALUE_LABELS.get("CO_TURNO", {}))
    cat_counts   = df["CO_CATEGAD"].value_counts().rename(VALUE_LABELS.get("CO_CATEGAD", {}))
    with col_e.expander("📊 Distribuição por Turno"):
        st.dataframe(turno_counts.rename("n"), use_container_width=True)
    with col_f.expander("🏛️ Distribuição por Tipo de IES"):
        st.dataframe(cat_counts.rename("n"), use_container_width=True)
    with st.expander("🔍 % de dados ausentes por variável (top 10)"):
        st.dataframe(
            ausentes[ausentes > 0].sort_values(ascending=False).head(10)
            .rename("% ausente").to_frame(),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Definição da Hipótese  (aparece após base carregada)
# ══════════════════════════════════════════════════════════════════════════════
if "df" not in st.session_state or len(st.session_state.df) < _MIN_REGISTROS:
    st.stop()

df = st.session_state.df

_section_divider()
_etapa_header(2, "Definição da Hipótese",
              "Selecione Y (nota a modelar), X (preditores) e interações a testar")

# 2.1 Variável dependente
with st.container(border=True):
    st.markdown('<div class="exn-filter-label">🎯 Variável Dependente (Y) — nota a modelar</div>',
                unsafe_allow_html=True)
    y_label = st.radio("Y", options=list(Y_OPTS.keys()),
                       horizontal=True, label_visibility="collapsed")
    y_var = Y_OPTS[y_label]

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# 2.2 Variáveis independentes
st.markdown('<div style="font-size:13px;font-weight:700;color:#374151;'
            'margin-bottom:10px;">📐 Variáveis Independentes (X)</div>',
            unsafe_allow_html=True)

x_disponiveis = {lbl: cod for lbl, cod in X_OPTS.items() if cod in df.columns}
x_selecionados_labels = []
cat_items = list(X_CATEGORIAS.items())

col_L, col_R = st.columns(2, gap="medium")

for col, grupo_items in [(col_L, cat_items[:1]), (col_R, cat_items[1:])]:
    with col:
        for nome_grupo, info in grupo_items:
            with st.container(border=True):
                st.markdown(
                    f'<div class="exn-var-group-header">'
                    f'<span class="exn-pill {info["pill"]}">{nome_grupo}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for lbl in info["vars"]:
                    if lbl not in x_disponiveis:
                        continue
                    if st.checkbox(lbl, value=(lbl in DEFAULT_X), key=f"x_{lbl}"):
                        x_selecionados_labels.append(lbl)

x_vars = [x_disponiveis[lbl] for lbl in x_selecionados_labels if lbl in x_disponiveis]

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# 2.3 Interações
with st.container(border=True):
    st.markdown('<div class="exn-filter-label">🔗 Interações entre variáveis (opcional)</div>',
                unsafe_allow_html=True)
    interactions = []
    if len(x_vars) >= 2:
        pairs = list(itertools.combinations(x_selecionados_labels, 2))
        inter_cols = st.columns(min(len(pairs), 3), gap="small")
        for i, (la, lb) in enumerate(pairs):
            ca, cb = x_disponiveis[la], x_disponiveis[lb]
            if inter_cols[i % len(inter_cols)].checkbox(f"{la} × {lb}",
                                                         key=f"inter_{la}_{lb}"):
                interactions.append((ca, cb))
    else:
        st.caption("Selecione ao menos 2 variáveis independentes para habilitar interações.")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if x_vars:
    _cod_to_lbl = {v: k for k, v in x_disponiveis.items()}
    preview = f"{y_label}  ←  {' + '.join(x_selecionados_labels)}"
    if interactions:
        inter_legivel = " + ".join(
            f"{_cod_to_lbl.get(a, a)} × {_cod_to_lbl.get(b, b)}"
            for a, b in interactions
        )
        preview += f"  +  [{inter_legivel}]"
    st.info(f"**Modelo a executar:** {preview}")

if st.button("▶ Executar Modelagem", type="primary", disabled=(len(x_vars) == 0)):
    hyp = build_hypothesis(y_var, x_vars, interactions, df)
    try:
        with st.spinner("Divisão treino/teste → OLS → VIF → resíduos → SHAP…"):
            df_train = df.sample(frac=0.8, random_state=42)
            df_test  = df.drop(df_train.index)

            result     = fit_ols(hyp, df)
            metrics    = get_model_metrics(result)
            coef_table = get_coefficients_table(result)

            try:
                y_pred = result.predict(df_test)
                y_real = df_test[hyp.y].dropna()
                y_pred = y_pred.reindex(y_real.index).dropna()
                y_real = y_real.reindex(y_pred.index)
                ss_res = ((y_real - y_pred) ** 2).sum()
                ss_tot = ((y_real - y_real.mean()) ** 2).sum()
                r2_test = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            except Exception:
                r2_test = float("nan")

            vif_table = compute_vif(df, x_vars)
            diag      = run_diagnostics(result)
            shap_vals, feat_names = compute_shap(result, df, x_vars)
            shap_sum  = get_shap_summary(shap_vals, feat_names)

    except Exception as e:
        st.error(
            f"❌ Não foi possível executar a modelagem.\n\n"
            f"**Causa provável:** variáveis com valores insuficientes para OLS.\n\n"
            f"**Sugestão:** remova algumas variáveis ou amplie os filtros da base.\n\n"
            f"Detalhe técnico: `{type(e).__name__}: {e}`"
        )
        st.stop()

    st.session_state.update({
        "hypothesis":     hyp,
        "formula":        hyp.to_formula(),
        "y_label":        y_label,
        "model_result":   result,
        "model_metrics":  metrics,
        "coef_table":     coef_table,
        "vif_table":      vif_table,
        "residuals_diag": diag,
        "shap_values":    shap_vals,
        "shap_summary":   shap_sum,
        "r2_test":        round(r2_test, 4),
        "n_treino":       len(df_train),
        "n_teste":        len(df_test),
        "shap_visivel":   False,
        "config": {
            "exec_id": st.session_state.exec_id,
            "formula": hyp.to_formula(),
            "filtros": st.session_state.get("filtros", {}),
        },
    })
    st.success("✅ Modelagem concluída! Os resultados aparecem abaixo.")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ETAPAS 3–7  (aparecem após modelagem executada)
# ══════════════════════════════════════════════════════════════════════════════
if "model_result" not in st.session_state:
    st.stop()

import plotly.express as px

metrics    = st.session_state.model_metrics
coef_table = st.session_state.coef_table
y_label    = st.session_state.get("y_label", "Nota Geral")
vif        = st.session_state.vif_table
diag       = st.session_state.residuals_diag
shap_sum   = st.session_state.shap_summary
r2_test    = st.session_state.get("r2_test", None)
n_treino   = st.session_state.get("n_treino", metrics["n_obs"])
n_teste    = st.session_state.get("n_teste", 0)

# ── 3. Modelagem e Diagnóstico ────────────────────────────────────────────────
_section_divider()
_etapa_header(3, "Modelagem e Diagnóstico",
              "Coeficientes OLS, poder explicativo e interpretação automática")

r2_test_str = (f"{r2_test*100:.1f}%"
               if (r2_test is not None and r2_test == r2_test) else "N/A")
_metric_row([
    (f"{metrics['r2']*100:.1f}%",   f"Poder Explicativo (R²) — {y_label}", "dark"),
    (f"{metrics['r2_adj']:.4f}",    "R² Ajustado",                          "violet"),
    (r2_test_str,                   "R² no Conjunto de Teste (20%)",         "blue"),
    (f"{n_treino:,} / {n_teste:,}", "Treino / Teste (80/20)",                "slate"),
])

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**3.1 — Efeito de cada fator na nota**")
    st.caption(
        "🟢 Verde = efeito comprovado (p < 0,05)  ·  "
        "↑ Aumenta / ↓ Diminui = direção do efeito  ·  "
        "IC 95% = intervalo de confiança"
    )
    st.dataframe(
        _style_coef_display(_coef_table_display(coef_table)),
        use_container_width=True, hide_index=True,
    )
    with st.expander("ℹ️ Como ler esta tabela?"):
        st.markdown("""
- **Efeito na Nota**: quanto a nota muda, em média, a cada nível a mais nesta variável
- **Confiável?**: ✅ = efeito improvável de ser acaso (p < 0,05) | ❌ = sem evidência suficiente
- **P-valor**: abaixo de 0,05 é considerado estatisticamente significativo
- **IC 95%**: intervalo onde o efeito verdadeiro está em 95% das análises repetidas
        """)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**3.2 — Interpretação automática**")
    st.markdown(_interpretar_modelo(coef_table, metrics, y_label))


# ── 4. Multicolinearidade (VIF) ───────────────────────────────────────────────
_section_divider()
_etapa_header(4, "Multicolinearidade (VIF)",
              "Detecta correlação excessiva entre preditores — VIF > 10 indica problema")

with st.expander("ℹ️ O que é esta análise?"):
    st.markdown("""
Quando duas variáveis são muito parecidas entre si, os coeficientes ficam instáveis.

| Cor | VIF | Significado |
|---|---|---|
| 🟢 Verde | ≤ 5 | Sem problema — variáveis independentes |
| 🟡 Amarelo | 5–10 | Atenção — correlação moderada |
| 🔴 Vermelho | > 10 | Problema — considere remover a variável |
    """)

display_vif = vif.copy()
display_vif["variavel"] = display_vif["variavel"].apply(_label)
display_vif = display_vif.rename(columns={"variavel": "Fator", "vif": "VIF", "alerta": "Situação"})

with st.container(border=True):
    st.dataframe(_style_vif(display_vif), use_container_width=True, hide_index=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
vif_status, vif_msg = _interpretar_vif(vif)
if vif_status == "ok":
    st.success(vif_msg)
else:
    st.warning(vif_msg)
    remove_var = get_variable_to_remove(vif)
    if st.button(f"🗑️ Remover '{_label(remove_var)}' e Reexecutar", type="primary",
                 key="btn_remove_vif"):
        hyp   = st.session_state.hypothesis
        new_x = [v for v in hyp.x_vars if v != remove_var]
        if not new_x:
            st.error("Não é possível remover — nenhuma variável restaria no modelo.")
            st.stop()
        new_interactions = [(a, b) for a, b in hyp.interactions
                            if a != remove_var and b != remove_var]
        new_hyp = build_hypothesis(hyp.y, new_x, new_interactions, df)
        with st.spinner("Reexecutando modelo..."):
            result     = fit_ols(new_hyp, df)
            m          = get_model_metrics(result)
            ct         = get_coefficients_table(result)
            vt         = compute_vif(df, new_x)
            dg         = run_diagnostics(result)
            sv, fn     = compute_shap(result, df, new_x)
            ss         = get_shap_summary(sv, fn)
        st.session_state.update({
            "hypothesis": new_hyp, "formula": new_hyp.to_formula(),
            "model_result": result, "model_metrics": m, "coef_table": ct,
            "vif_table": vt, "residuals_diag": dg, "shap_values": sv,
            "shap_summary": ss, "shap_visivel": False,
        })
        st.success(f"✅ Variável '{_label(remove_var)}' removida. Modelo reexecutado.")
        st.rerun()


# ── 5. Diagnóstico de Resíduos ────────────────────────────────────────────────
_section_divider()
_etapa_header(5, "Diagnóstico de Resíduos",
              "Testes formais de normalidade e homocedasticidade dos resíduos OLS")

with st.expander("ℹ️ O que são estes testes e por que importam?"):
    st.markdown("""
Os pressupostos do modelo OLS exigem que os resíduos (erros do modelo) sejam:

1. **Normalmente distribuídos** — verificado pelo teste de Shapiro-Wilk e QQ-plot.
   - H₀: os resíduos seguem distribuição normal.
   - Rejeitar H₀ (p < 0,05) indica desvio da normalidade.

2. **Homocedásticos** (variância constante) — verificado pelo teste de Breusch-Pagan.
   - H₀: a variância dos erros é constante para todos os valores preditos.
   - Rejeitar H₀ (p < 0,05) indica **heterocedasticidade**.

> **Nota com n > 5.000:** o Teorema Central do Limite garante que os estimadores OLS
> continuam válidos assintoticamente mesmo com desvios leves da normalidade.
    """)

# ── Tabela formal de testes ───────────────────────────────────────────────────
sw_n    = diag.get("shapiro_n_usado", 5000)
sw_note = f"(amostra de {sw_n:,})" if not diag.get("shapiro_confiavel", True) else ""

testes_df = pd.DataFrame([
    {
        "Teste":         f"Shapiro-Wilk {sw_note}",
        "H₀":            "Resíduos normalmente distribuídos",
        "Estatística":   f"{diag['shapiro_stat']:.4f}",
        "p-valor":       _p_fmt(diag['shapiro_pvalue']),
        "Decisão":       "Não rejeita H₀ ✅" if diag["shapiro_normal"] else "Rejeita H₀ ⚠️",
    },
    {
        "Teste":         "Breusch-Pagan",
        "H₀":            "Variância dos erros é constante (homocedasticidade)",
        "Estatística":   f"{diag['bp_stat']:.4f}",
        "p-valor":       _p_fmt(diag['bp_pvalue']),
        "Decisão":       "Não rejeita H₀ ✅" if diag["bp_homocedastic"] else "Rejeita H₀ ⚠️",
    },
])

def _style_testes(df: pd.DataFrame):
    def color_row(row):
        ok = "Não rejeita" in row["Decisão"]
        bg  = "#f0fdf4" if ok else "#fff7ed"
        txt = "#14532d" if ok else "#92400e"
        return [f"background-color: {bg}; color: {txt}; font-weight: 600"
                if col == "Decisão" else f"color: #1e293b"
                for col in df.columns]
    return df.style.apply(color_row, axis=1)

with st.container(border=True):
    st.markdown("**Resultados dos Testes Formais**")
    st.dataframe(_style_testes(testes_df), use_container_width=True, hide_index=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Mensagens de diagnóstico específicas (formato TCC) ────────────────────────
n_obs = metrics["n_obs"]

if not diag["bp_homocedastic"]:
    # Com n > 5.000, o BP rejeita facilmente desvios mínimos — adicionar contexto
    contexto_n = (
        f" Com n = {n_obs:,}, o teste é altamente sensível e detecta "
        f"variações pequenas de variância que têm impacto prático mínimo."
        if n_obs > 5000 else ""
    )
    st.warning(
        f"⚠️ **Indícios de heterocedasticidade detectados** "
        f"(Breusch-Pagan p {_p_fmt(diag['bp_pvalue'])}).{contexto_n} "
        f"A variância dos erros não é constante entre os grupos — os coeficientes "
        f"continuam não-viesados, mas os erros-padrão e p-valores podem estar subestimados. "
        f"**Considere:** transformação da variável dependente (ex: escala logarítmica) "
        f"ou uso de erros-padrão robustos à heterocedasticidade (HC3)."
    )

if not diag["shapiro_normal"]:
    if not diag.get("shapiro_confiavel", True):
        st.info(
            f"ℹ️ **Leve desvio de normalidade nos resíduos** "
            f"(Shapiro-Wilk p {_p_fmt(diag['shapiro_pvalue'])}, "
            f"avaliado em amostra de {sw_n:,} observações). "
            f"Com n = {n_obs:,}, o **Teorema Central do Limite** garante a "
            f"validade assintótica dos estimadores OLS — este resultado tem impacto mínimo "
            f"nas inferências e não invalida o modelo."
        )
    else:
        st.warning(
            f"⚠️ **Indícios de não-normalidade dos resíduos** "
            f"(Shapiro-Wilk p {_p_fmt(diag['shapiro_pvalue'])}). "
            f"Os intervalos de confiança e p-valores podem estar afetados. "
            f"Verifique a presença de outliers ou considere transformação da variável dependente."
        )

if diag["shapiro_normal"] and diag["bp_homocedastic"]:
    st.success(
        f"✅ **Todos os pressupostos verificados.** "
        f"Resíduos normalmente distribuídos (Shapiro-Wilk p {_p_fmt(diag['shapiro_pvalue'])}) "
        f"e variância constante (Breusch-Pagan p {_p_fmt(diag['bp_pvalue'])}). "
        f"O modelo OLS é metodologicamente adequado para estes dados."
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Gráficos ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    col_g1, col_g2 = st.columns(2, gap="medium")
    plot_data = get_residuals_plot_data(st.session_state.model_result)

    fig_res = px.scatter(
        plot_data, x="fitted", y="residual",
        labels={"fitted": "Valores Preditos", "residual": "Resíduos"},
        title="Resíduos vs. Valores Preditos",
        opacity=0.3, color_discrete_sequence=["#7c3aed"],
    )
    fig_res.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=1.5)
    fig_res.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa",
                          margin=dict(t=44, b=20, l=20, r=20))
    col_g1.plotly_chart(fig_res, use_container_width=True)
    col_g1.caption("Pontos espalhados aleatoriamente em volta de y = 0 indicam homocedasticidade.")

    qq_data = get_qqplot_data(st.session_state.model_result)
    fig_qq = px.scatter(
        qq_data, x="theoretical", y="sample",
        labels={"theoretical": "Quantis Teóricos (Normal)", "sample": "Quantis Amostrais"},
        title="QQ-Plot dos Resíduos",
        opacity=0.3, color_discrete_sequence=["#7c3aed"],
    )
    fig_qq.add_scatter(
        x=qq_data["theoretical"], y=qq_data["line_y"],
        mode="lines", line=dict(color="#ef4444", dash="dash", width=1.5),
        name="Referência Normal", showlegend=False,
    )
    fig_qq.update_layout(paper_bgcolor="white", plot_bgcolor="#fafafa",
                         margin=dict(t=44, b=20, l=20, r=20))
    col_g2.plotly_chart(fig_qq, use_container_width=True)
    col_g2.caption("Pontos próximos à linha diagonal indicam distribuição normal dos resíduos.")


# ── 6. Explicabilidade (SHAP) ─────────────────────────────────────────────────
_section_divider()
_etapa_header(6, "Explicabilidade (SHAP)",
              "Importância de cada fator na nota — SHAP LinearExplainer")

st.markdown(
    "A análise SHAP revela **quais fatores mais influenciam** a nota e o **quanto** "
    "cada um contribui para as diferenças entre os estudantes."
)

if st.button("▶ Ver Explicabilidade SHAP", type="primary", key="btn_shap"):
    st.session_state["shap_visivel"] = True

if st.session_state.get("shap_visivel", False):
    display_shap = shap_sum.copy()
    display_shap["variavel"] = display_shap["variavel"].apply(
        lambda v: " × ".join(_label(p) for p in v.split(":")) if ":" in v else _label(v)
    )
    display_shap = display_shap.rename(columns={"shap_mean_abs": "Influência média (pontos)"})

    with st.container(border=True):
        fig = px.bar(
            display_shap,
            x="Influência média (pontos)", y="variavel",
            orientation="h",
            labels={"variavel": ""},
            title=f"Importância dos Fatores — {y_label}",
            color="Influência média (pontos)",
            color_continuous_scale=[[0, "#ede9fe"], [0.5, "#8b5cf6"], [1, "#4c1d95"]],
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=48, b=16, l=8, r=24),
            font=dict(size=13),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Como ler este gráfico?"):
            st.markdown("""
- **Barra maior** = esse fator faz mais diferença na nota entre os estudantes
- O número indica, em média, **quantos pontos** são atribuíveis a cada fator
- Não diz a direção (se aumenta ou diminui) — para isso, veja a Etapa 3
            """)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.success(_interpretar_shap(shap_sum, y_label))


# ── 7. Relatório PDF ──────────────────────────────────────────────────────────
_section_divider()
_etapa_header(7, "Relatório Técnico PDF",
              "Gera documento com todos os resultados, gráficos e interpretações")

formula_str = st.session_state.get("formula", "N/A")
_metric_row([
    (st.session_state.exec_id[:16] + "…", "Exec-ID da Sessão",  "dark"),
    (formula_str[:44] + ("…" if len(formula_str) > 44 else ""),
     "Fórmula do Modelo", "violet"),
])

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

if st.button("📄 Gerar Relatório PDF", type="primary", key="btn_pdf"):
    with st.spinner("Gerando PDF..."):
        path = generate_report(
            session={
                "exec_id":                st.session_state.exec_id,
                "config":                 st.session_state.get("config", {}),
                "formula":                st.session_state.get("formula", ""),
                "model_metrics":          metrics,
                "coef_table":             coef_table,
                "vif_table":              vif,
                "residuals_diag":         diag,
                "shap_summary":           shap_sum,
                "y_label":                y_label,
                "interpretacao_modelo":   _interpretar_modelo(coef_table, metrics, y_label),
                "interpretacao_vif":      _interpretar_vif(vif)[1],
                "interpretacao_residuos": _interpretar_residuos(diag),
                "interpretacao_shap":     _interpretar_shap(shap_sum, y_label),
                "plot_data":              get_residuals_plot_data(st.session_state.model_result),
            }
        )
    st.success(f"✅ Relatório gerado: `{path.name}`")
    with open(path, "rb") as f:
        st.download_button(
            label="⬇️ Baixar PDF",
            data=f,
            file_name=path.name,
            mime="application/pdf",
            type="primary",
            key="btn_download_pdf",
        )
