"""
Camada 8 — Gerador de relatório técnico em PDF.

Produz um relatório acessível para não-especialistas, com legendas,
textos explicativos e linguagem clara em cada seção.
"""
import io
from datetime import date
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

_ARIAL_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
_ARIAL_BOLD    = Path("C:/Windows/Fonts/arialbd.ttf")
_USE_UNICODE   = _ARIAL_REGULAR.exists()

# Paleta de cores
_COR_AZUL    = (41,  98, 168)
_COR_VERDE   = (40, 167,  69)
_COR_AMARELO = (255, 193,  7)
_COR_VERMELHO= (220,  53,  30)
_COR_CINZA   = (108, 117, 125)
_COR_FUNDO   = (245, 247, 250)
_COR_TITULO  = (33,  37,  41)


# ══════════════════════════════════════════════════════════════════════════════
# Classe base
# ══════════════════════════════════════════════════════════════════════════════

class _PDF(FPDF):
    def __init__(self):
        super().__init__()
        if _USE_UNICODE:
            self.add_font("Arial",  fname=str(_ARIAL_REGULAR))
            self.add_font("Arial", style="B", fname=str(_ARIAL_BOLD))
            self._f = "Arial"
        else:
            self._f = "Helvetica"
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(*_COR_AZUL)
        self.rect(0, 0, 210, 12, "F")
        self.set_font(self._f, "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(15, 2)
        self.cell(0, 8, "E-XplainENADE  -  Relatorio de Analise Estatistica", align="L")
        self.set_text_color(*_COR_TITULO)
        self.ln(14)

    def footer(self):
        self.set_y(-13)
        self.set_font(self._f, size=8)
        self.set_text_color(*_COR_CINZA)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")
        self.set_text_color(*_COR_TITULO)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def titulo_secao(self, numero: str, titulo: str):
        """Cabeçalho de seção com fundo colorido."""
        self.ln(4)
        self.set_fill_color(*_COR_AZUL)
        self.set_text_color(255, 255, 255)
        self.set_font(self._f, "B", 11)
        self.cell(0, 9, f"  {numero}. {titulo}", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_COR_TITULO)
        self.ln(2)

    def caixa_contexto(self, texto: str):
        """Caixa cinza claro com texto explicativo."""
        self.set_fill_color(*_COR_FUNDO)
        self.set_font(self._f, size=9)
        self.set_text_color(*_COR_CINZA)
        x = self.get_x()
        self.multi_cell(0, 5, texto, fill=True, border=0,
                        new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_COR_TITULO)
        self.ln(2)

    def caixa_destaque(self, texto: str, cor: tuple = None):
        """Caixa destacada (verde por padrão) para interpretações."""
        r, g, b = cor or (212, 237, 218)
        self.set_fill_color(r, g, b)
        self.set_font(self._f, size=10)
        self.multi_cell(0, 6, texto, fill=True, border=0,
                        new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def metrica(self, label: str, valor: str, largura: float = 55):
        """Bloco de métrica estilo cartão."""
        self.set_fill_color(*_COR_FUNDO)
        self.set_font(self._f, "B", 16)
        self.set_text_color(*_COR_AZUL)
        x, y = self.get_x(), self.get_y()
        self.cell(largura, 10, valor, align="C")
        self.set_xy(x, y + 10)
        self.set_font(self._f, size=8)
        self.set_text_color(*_COR_CINZA)
        self.cell(largura, 5, label, align="C")
        self.set_text_color(*_COR_TITULO)

    def th(self, texto: str, w: float, align: str = "C"):
        """Célula de cabeçalho de tabela."""
        self.set_fill_color(*_COR_AZUL)
        self.set_text_color(255, 255, 255)
        self.set_font(self._f, "B", 9)
        self.cell(w, 7, texto, border=1, fill=True, align=align)
        self.set_text_color(*_COR_TITULO)

    def td(self, texto: str, w: float, align: str = "L",
           fill_color: tuple = None, bold: bool = False):
        """Célula de dado de tabela."""
        if fill_color:
            self.set_fill_color(*fill_color)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font(self._f, "B" if bold else "", 9)
        self.cell(w, 6, str(texto)[:40], border=1,
                  fill=True, align=align)

    def img_buf(self, fig) -> io.BytesIO:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf


# ══════════════════════════════════════════════════════════════════════════════
# Gráficos
# ══════════════════════════════════════════════════════════════════════════════

def _grafico_shap(shap_summary, code_to_label: Dict) -> plt.Figure:
    labels = [
        " x ".join(code_to_label.get(p, p) for p in v.split(":"))
        if ":" in v else code_to_label.get(v, v)
        for v in shap_summary["variavel"]
    ]
    values = shap_summary["shap_mean_abs"].values

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55)))
    bars = ax.barh(labels[::-1], values[::-1],
                   color=plt.cm.Blues(np.linspace(0.45, 0.85, len(values))))
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Influencia media na nota (pontos)", fontsize=9)
    ax.set_title("Quais fatores mais influenciam a nota?", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _grafico_residuos(plot_data) -> plt.Figure:
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Resíduos vs fitted
    ax = axes[0]
    ax.scatter(plot_data["fitted"], plot_data["residual"],
               alpha=0.25, s=8, color="#4a90d9")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Nota prevista pelo modelo", fontsize=9)
    ax.set_ylabel("Erro do modelo (residuo)", fontsize=9)
    ax.set_title("Dispersao dos Erros\n(pontos devem ficar espalhados ao redor da linha vermelha)",
                 fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # QQ-plot
    ax2 = axes[1]
    residuals = plot_data["residual"].values
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
    ax2.scatter(osm, osr, alpha=0.25, s=8, color="#4a90d9")
    ax2.plot(osm, slope * osm + intercept,
             color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Padrao teorico esperado", fontsize=9)
    ax2.set_ylabel("Erros reais do modelo", fontsize=9)
    ax2.set_title("Grafico de Normalidade dos Erros\n(pontos devem seguir a linha vermelha)",
                  fontsize=9, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Função pública
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(session: Dict[str, Any], filename: str = "relatorio.pdf") -> Path:
    """
    Gera relatório PDF acessível a partir do dicionário de sessão.

    Chaves esperadas: exec_id, config, formula, y_label,
    model_metrics, coef_table, vif_table, residuals_diag,
    shap_summary, interpretacao_modelo, interpretacao_residuos,
    interpretacao_shap, plot_data
    """
    from config.variable_map import CODE_TO_LABEL
    from modules.multicollinearity import VIF_MODERADO, VIF_SEVERO

    pdf = _PDF()
    pdf.add_page()

    y_label  = session.get("y_label", "Nota Geral")
    filtros  = session.get("config", {}).get("filtros", {})
    hoje     = date.today().strftime("%d/%m/%Y")

    # ── Capa ──────────────────────────────────────────────────────────────────
    pdf.set_font(pdf._f, "B", 18)
    pdf.set_text_color(*_COR_AZUL)
    pdf.cell(0, 12, "Relatorio de Analise Estatistica", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(pdf._f, size=11)
    pdf.set_text_color(*_COR_CINZA)
    pdf.cell(0, 7, f"Microdados ENADE 2021  |  Gerado em {hoje}", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*_COR_TITULO)
    pdf.ln(6)

    # Bloco de contexto
    pdf.set_fill_color(*_COR_FUNDO)
    pdf.set_font(pdf._f, size=9)
    pdf.multi_cell(0, 6,
        f"Exec-ID: {session.get('exec_id','N/A')}\n"
        f"Variavel analisada: {y_label}\n"
        f"Curso: {filtros.get('curso','Todos')}  |  "
        f"Tipo de IES: {filtros.get('tipo_ies','Todas')}  |  "
        f"Ano: {filtros.get('ano','2021')}\n"
        f"Formula: {session.get('formula','N/A')}",
        fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── 1. Poder explicativo ──────────────────────────────────────────────────
    metrics = session.get("model_metrics", {})
    if metrics:
        pdf.titulo_secao("1", f"Quanto o modelo explica a {y_label}?")

        pdf.caixa_contexto(
            "O 'Poder Explicativo' (R2) mostra qual percentual da variacao nas notas "
            "e explicado pelos fatores selecionados. Quanto mais alto, melhor o modelo descreve "
            "a realidade dos dados."
        )

        # Métricas em linha
        r2_pct = f"{metrics['r2']*100:.1f}%"
        n      = f"{metrics['n_obs']:,}"
        r2adj  = f"{metrics['r2_adj']:.4f}"

        pdf.set_xy(15, pdf.get_y())
        pdf.metrica("Poder Explicativo (R2)", r2_pct, 60)
        pdf.set_xy(75, pdf.get_y() - 15)
        pdf.metrica("Estudantes Analisados", n, 60)
        pdf.set_xy(135, pdf.get_y() - 15)
        pdf.metrica("R2 Ajustado", r2adj, 60)
        pdf.ln(22)

        interp = session.get("interpretacao_modelo", "")
        if interp:
            pdf.caixa_destaque(interp)

    # ── 2. Efeito de cada fator ───────────────────────────────────────────────
    coef = session.get("coef_table")
    if coef is not None:
        pdf.titulo_secao("2", f"Efeito de cada fator na {y_label}")

        pdf.caixa_contexto(
            "Como ler esta tabela:\n"
            "  Efeito: quanto a nota muda, em media, a cada nivel a mais neste fator\n"
            "  Direcao: se o fator aumenta (↑) ou diminui (↓) a nota do estudante\n"
            "  Confiavel?: 'Sim' = efeito improvavel de ser coincidencia (p < 0,05)\n"
            "  IC 95%: intervalo onde o efeito verdadeiro esta em 95% das vezes"
        )

        # Cabeçalho
        widths = [62, 22, 22, 22, 18, 34]
        headers = ["Fator", "Efeito", "Direcao", "Confiavel?", "P-valor", "IC 95%"]
        for h, w in zip(headers, widths):
            pdf.th(h, w)
        pdf.ln()

        for _, row in coef.iterrows():
            var = str(row["variavel"])
            if var == "Intercept":
                continue
            label = " x ".join(CODE_TO_LABEL.get(p, p) for p in var.split(":")) \
                    if ":" in var else CODE_TO_LABEL.get(var, var)
            c, p  = row["coeficiente"], row["p_valor"]
            sig   = p < 0.05
            p_str = "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
            ic    = f"[{row['ic_inf_95']:.2f} ; {row['ic_sup_95']:.2f}]"
            fill  = (212, 237, 218) if sig else (255, 255, 255)

            vals = [
                label[:38],
                f"{c:+.2f}",
                "Aumenta" if c > 0 else "Diminui",
                "Sim" if sig else "Nao",
                p_str,
                ic,
            ]
            aligns = ["L", "C", "C", "C", "C", "C"]
            for v, w, a in zip(vals, widths, aligns):
                pdf.td(v, w, a, fill_color=fill)
            pdf.ln()

        pdf.ln(2)
        pdf.set_font(pdf._f, size=8)
        pdf.set_text_color(*_COR_CINZA)
        pdf.cell(0, 5, "Legenda: verde = efeito comprovado (p < 0,05)  |  "
                       "IC 95% = intervalo de confianca de 95%",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_COR_TITULO)
        pdf.ln(4)

    # ── 3. Correlação entre variáveis (VIF) ──────────────────────────────────
    vif = session.get("vif_table")
    if vif is not None:
        pdf.titulo_secao("3", "Correlacao entre as Variaveis (VIF)")

        pdf.caixa_contexto(
            "O que e esta analise: verifica se as variaveis escolhidas sao muito parecidas "
            "entre si. Quando duas variaveis andam sempre juntas (ex: renda e escolaridade da mae), "
            "os coeficientes ficam instáveis e menos confiáveis.\n\n"
            "Como interpretar:\n"
            "  Verde  (VIF <= 5): sem problema\n"
            "  Amarelo (VIF 5-10): atencao — correlacao moderada\n"
            "  Vermelho (VIF > 10): problema — considere remover a variavel"
        )

        widths_v = [90, 30, 60]
        for h, w in zip(["Fator", "VIF", "Situacao"], widths_v):
            pdf.th(h, w)
        pdf.ln()

        for _, row in vif.iterrows():
            label = CODE_TO_LABEL.get(row["variavel"], row["variavel"])
            v     = row["vif"]
            if v > VIF_SEVERO:
                fill = (255, 204, 204)
                sit  = "Problema (VIF > 10)"
            elif v > VIF_MODERADO:
                fill = (255, 243, 205)
                sit  = "Atencao (VIF > 5)"
            else:
                fill = (212, 237, 218)
                sit  = "OK"
            pdf.td(label[:42], 90, "L", fill)
            pdf.td(f"{v:.2f}", 30, "C", fill)
            pdf.td(sit, 60, "L", fill)
            pdf.ln()

        pdf.ln(4)
        interp_vif = session.get("interpretacao_vif", "")
        if interp_vif:
            fill_cor = (212, 237, 218) if "OK" in interp_vif or "ok" in interp_vif \
                       else (255, 243, 205)
            pdf.caixa_destaque(interp_vif, fill_cor)

    # ── 4. Qualidade do modelo (Resíduos) ────────────────────────────────────
    diag = session.get("residuals_diag", {})
    if diag:
        pdf.add_page()
        pdf.titulo_secao("4", "Qualidade do Modelo")

        pdf.caixa_contexto(
            "Esta secao verifica se o modelo se comporta de forma correta e honesta.\n\n"
            "Distribuicao dos erros: o modelo deveria errar de forma equilibrada — "
            "erros pequenos mais comuns, erros grandes raros.\n\n"
            "Consistencia dos erros: o modelo deveria errar de forma parecida para todos "
            "os perfis de estudante, independentemente de renda ou outros fatores."
        )

        # Tabela de testes
        widths_r = [70, 45, 65]
        for h, w in zip(["Verificacao", "Resultado", "O que significa"], widths_r):
            pdf.th(h, w)
        pdf.ln()

        sw_ok   = diag["shapiro_normal"]
        bp_ok   = diag["bp_homocedastic"]
        conf_sw = "" if diag.get("shapiro_confiavel", True) else "*"

        linhas_res = [
            (
                f"Distribuicao dos erros{conf_sw}",
                "Normal" if sw_ok else "Nao-normal",
                "Satisfatorio" if sw_ok else "Veja nota abaixo",
                (212, 237, 218) if sw_ok else (255, 243, 205),
            ),
            (
                "Consistencia dos erros",
                "Consistente" if bp_ok else "Inconsistente",
                "Satisfatorio" if bp_ok else "Erros variam entre grupos",
                (212, 237, 218) if bp_ok else (255, 204, 204),
            ),
        ]
        for nome, res, sig, fill in linhas_res:
            pdf.td(nome, 70, "L", fill)
            pdf.td(res,  45, "C", fill)
            pdf.td(sig,  65, "L", fill)
            pdf.ln()

        if not diag.get("shapiro_confiavel", True):
            pdf.ln(1)
            pdf.set_font(pdf._f, size=8)
            pdf.set_text_color(*_COR_CINZA)
            pdf.cell(0, 5,
                f"* Teste aplicado em amostra de {diag.get('shapiro_n_usado',5000):,} "
                f"estudantes. Com mais de 5.000 observacoes, pequenas irregularidades "
                f"sao detectadas mas tem impacto minimo na validade dos resultados.",
                new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_COR_TITULO)

        pdf.ln(4)
        interp_res = session.get("interpretacao_residuos", "")
        if interp_res:
            fill_cor = (212, 237, 218) if (sw_ok and bp_ok) else (255, 243, 205)
            # Remove markdown para PDF
            clean = interp_res.replace("✅", "OK:").replace("⚠️", "Atencao:") \
                              .replace("ℹ️", "Nota:").replace("**", "")
            pdf.caixa_destaque(clean, fill_cor)

        # Gráficos de resíduos
        plot_data = session.get("plot_data")
        if plot_data is not None:
            pdf.ln(3)
            fig = _grafico_residuos(plot_data)
            buf = pdf.img_buf(fig)
            pdf.image(buf, w=178)

            pdf.ln(2)
            pdf.set_font(pdf._f, size=8)
            pdf.set_text_color(*_COR_CINZA)
            pdf.multi_cell(0, 5,
                "Grafico da esquerda: cada ponto e um estudante. Os pontos devem ficar "
                "espalhados aleatoriamente em volta da linha vermelha (sem padroes visiveis).\n"
                "Grafico da direita (QQ-plot): os pontos devem seguir a linha diagonal. "
                "Desvios nas extremidades indicam que o modelo comete erros maiores "
                "para notas muito altas ou muito baixas.",
                new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*_COR_TITULO)

    # ── 5. Quais fatores mais influenciam (SHAP) ─────────────────────────────
    shap_summary = session.get("shap_summary")
    if shap_summary is not None:
        pdf.add_page()
        pdf.titulo_secao("5", f"Quais fatores mais influenciam a {y_label}?")

        pdf.caixa_contexto(
            "O grafico abaixo mostra a importancia relativa de cada fator. "
            "Quanto maior a barra, maior o impacto daquele fator para explicar "
            "as diferencas de notas entre os estudantes.\n\n"
            "Atencao: este grafico mostra O QUANTO cada fator importa (magnitude), "
            "nao a direcao do efeito. Para saber se o fator aumenta ou diminui a nota, "
            "consulte a Secao 2."
        )

        interp_shap = session.get("interpretacao_shap", "")
        if interp_shap:
            clean = interp_shap.replace("**", "")
            pdf.caixa_destaque(clean)

        fig = _grafico_shap(shap_summary, CODE_TO_LABEL)
        buf = pdf.img_buf(fig)
        pdf.image(buf, w=160)

        pdf.ln(2)
        pdf.set_font(pdf._f, size=8)
        pdf.set_text_color(*_COR_CINZA)
        pdf.multi_cell(0, 5,
            "Legenda: o valor ao lado de cada barra indica a influencia media em pontos "
            "— por exemplo, '7,95' significa que este fator esta associado, em media, "
            "a uma diferenca de 7,95 pontos na nota entre estudantes com valores "
            "altos e baixos nesta variavel.",
            new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_COR_TITULO)

    # ── Rodapé com nota metodológica ─────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font(pdf._f, size=8)
    pdf.set_text_color(*_COR_CINZA)
    pdf.multi_cell(0, 5,
        "Nota metodologica: os resultados foram obtidos por regressao linear multipla "
        "(OLS via statsmodels) com explicabilidade SHAP (LinearExplainer, Lundberg & Lee, 2017). "
        "Os dados sao os Microdados ENADE 2021 (INEP), versao LGPD, filtrados para as regioes "
        "Norte e Nordeste, cursos de Ciencia da Computacao e Sistemas de Informacao.",
        new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*_COR_TITULO)

    _OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = _OUTPUT_DIR / filename
    pdf.output(str(output_path))
    return output_path
