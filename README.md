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

A aplicação abre em `http://localhost:8501`. Os dados do recorte do TCC já fazem
parte do sistema (`dados/enade_2021_nne_ccsi.csv.gz` — versionado no repositório);
na Etapa 1 basta escolher os filtros e clicar em **Carregar Base**.

### Regenerar a base do recorte (opcional)

O arquivo embutido é gerado de forma reprodutível a partir dos microdados brutos
do INEP (43 arquivos `.txt`, não distribuídos no repositório):

```bash
python gerar_dados.py
```

Recorte aplicado: ENADE **2021** · regiões **Norte + Nordeste** (CO_REGIAO_CURSO 1, 2) ·
cursos **CC + SI** (CO_GRUPO 4004, 4006) · apenas **presentes** (TP_PRES = 555) →
**5.114 estudantes**, 157 colunas brutas (~0,4 MB comprimido).

---

## Deploy no Streamlit Cloud

1. Faça push do repositório para o GitHub (a base do recorte em `dados/` é versionada; os microdados brutos não)
2. Acesse [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Selecione o repositório e `app.py` como entry point
4. O Streamlit Cloud instala automaticamente as dependências de `requirements.txt` e `packages.txt`

O `packages.txt` instala `fonts-liberation` (Liberation Sans, compatível com Arial) para geração correta do PDF no Linux.

---

## Fluxo da aplicação

A interface é uma **página única vertical** com 7 etapas que aparecem progressivamente:

| Etapa | Descrição | Pré-requisito |
|---|---|---|
| 1 | Seleção de base: filtros de curso e tipo de IES sobre a base embutida | — |
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
| Inscritos no ENADE 2021 (nacional) | 489.866 |
| Recorte do TCC: N+NE · CC+SI (antes do filtro de presença) | 6.303 |
| **Base embutida** (presentes, TP_PRES = 555) | **5.114** |
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

- A base embutida é gerada de forma determinística por `gerar_dados.py` e versionada no git — todos analisam exatamente os mesmos dados.
- O modelo OLS é ajustado na **base completa do recorte** (abordagem confirmatória — inferência via p-valores e IC 95%) e é determinístico: mesma hipótese → mesmos coeficientes.
- O Shapiro-Wilk com n > 5.000 amostra exatamente 5.000 observações com `seed=42`.
- Cada sessão gera um **Exec-ID** único (UUID), gravado no relatório PDF (`relatorio_<exec-id>.pdf`) e no registro de configuração exportável (`config_<exec-id>.json`) — que contém fórmula, variáveis, interações, filtros e seed, suficiente para replicar a análise.

---

## Fonte dos dados

[INEP — Microdados ENADE 2021](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enade)

Os microdados brutos completos não são distribuídos neste repositório — apenas o
subconjunto do recorte do TCC (dados públicos e anonimizados pelo INEP, versão LGPD).
