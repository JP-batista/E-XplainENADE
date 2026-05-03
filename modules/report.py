"""
Camada 8 — Gerador de relatório técnico em PDF.

Consolida todos os resultados da sessão analítica em um relatório PDF
exportável, usando fpdf2. Registra o Exec-ID e a configuração JSON da
sessão para garantir rastreabilidade e reprodutibilidade.
"""
from pathlib import Path
from typing import Any, Dict

from fpdf import FPDF

_OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Fonte com suporte a Unicode/UTF-8 para cobrir português e símbolos
# Usa Arial do Windows; fallback para Helvetica (só ASCII/Latin-1)
_ARIAL_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
_ARIAL_BOLD    = Path("C:/Windows/Fonts/arialbd.ttf")
_USE_UNICODE   = _ARIAL_REGULAR.exists()


class _RelatorioPDF(FPDF):
    def __init__(self):
        super().__init__()
        if _USE_UNICODE:
            self.add_font("Arial", fname=str(_ARIAL_REGULAR))
            self.add_font("Arial", style="B", fname=str(_ARIAL_BOLD))
            self._f = "Arial"
        else:
            self._f = "Helvetica"

    def header(self):
        self.set_font(self._f, "B", 12)
        self.cell(0, 10, "E-XplainENADE - Relatorio Tecnico", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._f, size=8)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")

    def section(self, title: str):
        self.set_font(self._f, "B", 11)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font(self._f, size=10)

    def row(self, text: str, size: int = 10):
        self.set_font(self._f, size=size)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")


def generate_report(session: Dict[str, Any], filename: str = "relatorio.pdf") -> Path:
    """
    Gera o relatório PDF a partir do dicionário de sessão.

    Parameters
    ----------
    session : dict
        Chaves esperadas: exec_id, formula, model_metrics,
        vif_table, residuals_diag, shap_summary
    filename : str
        Nome do arquivo de saída (salvo em outputs/).

    Returns
    -------
    Path
        Caminho absoluto do PDF gerado.
    """
    pdf = _RelatorioPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── 1. Identificação ───────────────────────────────────────────────────
    pdf.section("1. Identificacao")
    pdf.row(f"Exec-ID : {session.get('exec_id', 'N/A')}")
    pdf.row(f"Formula : {session.get('formula', 'N/A')}")
    pdf.ln(4)

    # ── 2. Métricas do modelo ──────────────────────────────────────────────
    metrics = session.get("model_metrics", {})
    if metrics:
        pdf.section("2. Metricas do Modelo")
        pdf.row(
            f"R2={metrics.get('r2')}  |  R2adj={metrics.get('r2_adj')}"
            f"  |  n={metrics.get('n_obs')}  |  F p-valor={metrics.get('f_pvalue')}"
        )
        pdf.ln(4)

    # ── 3. VIF ────────────────────────────────────────────────────────────
    vif = session.get("vif_table")
    if vif is not None:
        pdf.section("3. Diagnostico de Multicolinearidade (VIF)")
        pdf.set_font(pdf._f, size=9)
        for _, row in vif.iterrows():
            pdf.cell(0, 5,
                f"  {str(row['variavel']):30s}  VIF={row['vif']:.3f}  [{row['alerta']}]",
                new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── 4. Diagnóstico de resíduos ─────────────────────────────────────────
    diag = session.get("residuals_diag", {})
    if diag:
        pdf.section("4. Diagnostico de Residuos")
        sw_ok  = "Normal"         if diag.get("shapiro_normal")    else "Nao-normal"
        bp_ok  = "Homocedastico"  if diag.get("bp_homocedastic")   else "Heterocedastico"
        conf   = "" if diag.get("shapiro_confiavel", True) else f"  [amostra={diag.get('shapiro_n_usado')}]"
        pdf.row(f"Shapiro-Wilk : p={diag.get('shapiro_pvalue')}  ->  {sw_ok}{conf}")
        pdf.row(f"Breusch-Pagan: p={diag.get('bp_pvalue')}  ->  {bp_ok}")
        pdf.ln(4)

    # ── 5. SHAP ───────────────────────────────────────────────────────────
    shap_summary = session.get("shap_summary")
    if shap_summary is not None:
        pdf.section("5. Impacto das Variaveis (SHAP - media |valor|)")
        pdf.set_font(pdf._f, size=9)
        for _, row in shap_summary.head(10).iterrows():
            pdf.cell(0, 5,
                f"  {str(row['variavel']):35s}  {row['shap_mean_abs']:.4f}",
                new_x="LMARGIN", new_y="NEXT")

    _OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = _OUTPUT_DIR / filename
    pdf.output(str(output_path))
    return output_path
