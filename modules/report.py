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


class _RelatorioPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "E-XplainENADE — Relatório Técnico", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")


def generate_report(session: Dict[str, Any], filename: str = "relatorio.pdf") -> Path:
    """
    Gera o relatório PDF a partir do dicionário de sessão.

    Parameters
    ----------
    session : dict
        Dicionário com as chaves:
          exec_id, config, hypothesis, formula,
          coef_table, model_metrics, vif_table,
          residuals_diag, shap_summary
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
    pdf.set_font("Helvetica", size=10)

    # ── Identificação da execução ──────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "1. Identificação", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Exec-ID: {session.get('exec_id', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Fórmula: {session.get('formula', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Métricas do modelo ─────────────────────────────────────────────────
    metrics = session.get("model_metrics", {})
    if metrics:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "2. Métricas do Modelo", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, f"R² = {metrics.get('r2')} | R² ajustado = {metrics.get('r2_adj')} | n = {metrics.get('n_obs')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── Diagnóstico VIF ────────────────────────────────────────────────────
    vif = session.get("vif_table")
    if vif is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "3. Diagnóstico de Multicolinearidade (VIF)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        for _, row in vif.iterrows():
            pdf.cell(0, 5, f"  {row['variavel']:30s}  VIF={row['vif']:.3f}  [{row['alerta']}]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── Diagnóstico de resíduos ────────────────────────────────────────────
    diag = session.get("residuals_diag", {})
    if diag:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "4. Diagnóstico de Resíduos", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        sw_result = "Normal" if diag.get("shapiro_normal") else "Não-normal"
        bp_result = "Homocedástico" if diag.get("bp_homocedastic") else "Heterocedástico"
        pdf.cell(0, 6, f"Shapiro-Wilk: p={diag.get('shapiro_pvalue')} → {sw_result}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"Breusch-Pagan: p={diag.get('bp_pvalue')} → {bp_result}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── SHAP ───────────────────────────────────────────────────────────────
    shap_summary = session.get("shap_summary")
    if shap_summary is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "5. Impacto das Variáveis (SHAP)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        for _, row in shap_summary.head(10).iterrows():
            pdf.cell(0, 5, f"  {row['variavel']:30s}  impacto médio = {row['shap_mean_abs']:.4f}", new_x="LMARGIN", new_y="NEXT")

    output_path = _OUTPUT_DIR / filename
    pdf.output(str(output_path))
    return output_path
