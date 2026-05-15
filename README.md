# E-XplainENADE

Arquitetura computacional reprodutível e explicável para modelagem estatística multivariada dos microdados ENADE.

Desenvolvido como Trabalho de Conclusão de Curso (TCC) — parte do ecossistema **EnadeX** (4 TCCs interligados).

---

## Sobre o projeto

O E-XplainENADE permite que pesquisadores definam hipóteses sobre fatores que influenciam o desempenho de estudantes no ENADE e obtenham:

- Modelo de regressão linear múltipla (OLS) com coeficientes, p-valores e IC 95%
- Diagnóstico automático de multicolinearidade (VIF)
- Testes formais de normalidade (Shapiro-Wilk) e homocedasticidade (Breusch-Pagan) dos resíduos
- Explicabilidade via SHAP LinearExplainer (importância de cada fator)
- Relatório técnico em PDF com interpretação automática em linguagem acadêmica

A abordagem é **confirmatória**, não preditiva — o objetivo é testar hipóteses sobre relações entre variáveis socioeconômicas e desempenho acadêmico, o que exige p-valores, intervalos de confiança e testes de hipótese (statsmodels, não scikit-learn).

---

## Requisitos

- Python 3.9+
- Os microdados brutos do ENADE 2021 (INEP/LGPD) para gerar o arquivo de dados — **não incluídos no repositório**

---

## Instalação

```bash
git clone https://github.com/JP-batista/E-XplainENADE.git
cd E-XplainENADE
pip install -r requirements.txt
```

---

## Como rodar localmente

```bash
streamlit run app.py
```

A aplicação abre em `http://localhost:8501`. Na Etapa 1, faça upload do arquivo `.csv.gz` gerado.

---

## Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub (sem os arquivos de dados — `.gitignore` já os exclui)
2. Acesse [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Selecione o repositório e `app.py` como entry point
4. O Streamlit Cloud instala automaticamente as dependências de `requirements.txt` e `packages.txt`
5. Acesse a URL gerada e faça upload do `.csv.gz` na Etapa 1

O `packages.txt` instala `fonts-liberation` (Liberation Sans, compatível com Arial) para geração correta do PDF no Linux.

---

## Fluxo da aplicação

A interface é uma **página única vertical** com 7 etapas que aparecem progressivamente:

| Etapa | Descrição | Pré-requisito |
|---|---|---|
| 1 | Upload do CSV + filtros de curso e tipo de IES | — |
| 2 | Definição da hipótese: Y (nota), X (preditores), interações | Base carregada |
| 3 | Coeficientes OLS, R², interpretação automática | Modelagem executada |
| 4 | Diagnóstico de multicolinearidade (VIF) com remoção interativa | Modelagem executada |
| 5 | Testes formais de resíduos + gráficos (resíduos vs preditos, QQ-plot) | Modelagem executada |
| 6 | Importância dos fatores via SHAP LinearExplainer | Modelagem executada |
| 7 | Geração e download do relatório técnico em PDF | Modelagem executada |

---

## Arquitetura

```
app.py                      — Camada 9: interface Streamlit (entry point)
modules/
  etl.py                    — Camada 1: ingestão dos 43 arquivos (SWAP POINT)
  loader.py                 — Camada 2: filtros, recodificação, renomeação, get_dataset_from_csv()
  hypothesis.py             — Camada 3: HypothesisConfig + to_formula()
  modeling.py               — Camada 4: statsmodels OLS
  multicollinearity.py      — Camada 5: VIF com retroalimentação interativa
  residuals.py              — Camada 6: Shapiro-Wilk + Breusch-Pagan
  explainability.py         — Camada 7: shap.LinearExplainer
  report.py                 — Camada 8: PDF via fpdf2 (fonte cross-platform)
config/
  variable_map.py           — mapeamento QE_I08 → QE_RENDA, labels PT-BR, tipos
.streamlit/
  config.toml               — tema claro, cor primária violeta, limite de upload 500 MB
packages.txt                — dependência de sistema para Streamlit Cloud (fonts-liberation)
```

---

## Dados

| Dado | Total |
|---|---|
| Inscritos no ENADE 2021 | 489.866 |
| Participantes efetivos (TP_PRES = 555) | 355.095 |
| Variáveis analíticas disponíveis | ~124 |

Apenas participantes efetivos são carregados (`apenas_presentes=True`), pois estudantes ausentes não possuem notas e não podem integrar o modelo de regressão.

**Variável `TP_CATEGAD_BIN`:** `CO_CATEGAD` (5 categorias nominais) é binarizada em `TP_CATEGAD_BIN` durante o pré-processamento: `{1,2,3} → 0` (Pública), `{4,5} → 1` (Privada). Usar CO_CATEGAD como preditora contínua em OLS seria metodologicamente incorreto.

---

## Dependências Python

| Biblioteca | Uso |
|---|---|
| `statsmodels` | Regressão OLS confirmatória (p-valores, IC, testes F) |
| `shap` | LinearExplainer — explicabilidade dos coeficientes |
| `patsy` | Reconstrói design matrix para SHAP com termos de interação |
| `scipy` | Shapiro-Wilk, probplot |
| `fpdf2` | Geração do relatório PDF |
| `plotly` | Gráficos interativos (resíduos, QQ-plot, SHAP) |
| `streamlit` | Interface web |

---

## Reprodutibilidade

Cada sessão gera um **Exec-ID** único (UUID). A divisão treino/teste usa `random_state=42`. O Shapiro-Wilk com n > 5.000 amostra exatamente 5.000 observações com `seed=42`. Esses parâmetros garantem que os resultados sejam reprodutíveis dado o mesmo conjunto de dados e hipótese.

---

## Fonte dos dados

[INEP — Microdados ENADE 2021](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enade)

Os dados não são distribuídos neste repositório.
