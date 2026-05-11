# E-XplainENADE

Arquitetura computacional reprodutível e explicável para modelagem estatística multivariada dos microdados ENADE.

Desenvolvido como Trabalho de Conclusão de Curso (TCC) — parte do ecossistema **EnadeX** (4 TCCs interligados).

---

## Sobre o projeto

O E-XplainENADE permite que pesquisadores definam hipóteses sobre fatores que influenciam o desempenho de estudantes no ENADE e obtenham:

- Modelo de regressão linear múltipla (OLS) com coeficientes, p-valores e IC 95%
- Diagnóstico automático de multicolinearidade (VIF)
- Testes formais de normalidade e homocedasticidade dos resíduos
- Explicabilidade via SHAP (importância de cada fator)
- Relatório técnico em PDF com interpretação automática em linguagem acessível

A abordagem é **confirmatória**, não preditiva — o objetivo é testar hipóteses sobre relações entre variáveis socioeconômicas e desempenho acadêmico, o que exige p-valores, intervalos de confiança e testes de hipótese (statsmodels, não scikit-learn).

---

## Requisitos

- Python 3.9+
- Os microdados brutos do ENADE 2021 (INEP/LGPD) — **não incluídos no repositório**

```
microdados_Enade_2021_LGPD/
  1.LEIA-ME/   ← dicionário de variáveis, questionários, manual
  2.DADOS/     ← 43 arquivos .txt separados por ";"
```

---

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/JP-batista/E-XplainENADE.git

# Instalar dependências
pip install -r requirements.txt
```

---

## Como rodar

```bash
# Verificar carregamento dos dados (primeira execução leva ~30s)
python -c "from modules.loader import get_dataset; df = get_dataset(); print(df.shape)"

# Iniciar a aplicação
streamlit run app.py
```

A aplicação abre no navegador em `http://localhost:8501`.

---

## Fluxo da aplicação

A interface é uma **página única vertical** com 7 etapas que aparecem progressivamente:

| Etapa | Descrição | Pré-requisito |
|---|---|---|
| 1 | Seleção de Base | — |
| 2 | Definição da Hipótese (Y, X, interações) | Base carregada |
| 3 | Modelagem OLS + Interpretação automática | Modelagem executada |
| 4 | Multicolinearidade (VIF) | Modelagem executada |
| 5 | Diagnóstico de Resíduos (Shapiro-Wilk + Breusch-Pagan) | Modelagem executada |
| 6 | Explicabilidade SHAP | Modelagem executada |
| 7 | Geração do Relatório PDF | Modelagem executada |

---

## Arquitetura

```
app.py                      ← Camada 9: interface Streamlit (entry point)
modules/
  etl.py                    ← Camada 1: ingestão dos 43 arquivos  ← SWAP POINT
  loader.py                 ← Camada 2: filtros, recodificação, renomeação
  hypothesis.py             ← Camada 3: HypothesisConfig + to_formula()
  modeling.py               ← Camada 4: statsmodels OLS
  multicollinearity.py      ← Camada 5: VIF com retroalimentação interativa
  residuals.py              ← Camada 6: Shapiro-Wilk + Breusch-Pagan
  explainability.py         ← Camada 7: shap.LinearExplainer
  report.py                 ← Camada 8: PDF via fpdf2
config/
  variable_map.py           ← mapeamento QE_I08 → QE_RENDA, labels PT-BR, tipos
outputs/                    ← PDFs gerados pela aplicação
.streamlit/
  config.toml               ← tema claro, cor primária violeta
```

### Dados

Os 43 arquivos são **row-aligned**: a linha N de `arq1` corresponde à linha N de `arq2`, etc. O join é posicional — sem chave de estudante (removida por LGPD). O filtro por curso/região é aplicado durante o carregamento para evitar ler os ~500 MB completos.

| Dado | Total |
|---|---|
| Inscritos no ENADE 2021 | 489.866 |
| Participantes efetivos (TP_PRES = 555) | 355.095 |
| Variáveis analíticas disponíveis | ~124 |

Apenas participantes efetivos são carregados por padrão (`apenas_presentes=True`), pois estudantes ausentes não possuem notas e não podem integrar o modelo.

### Variável `TP_CATEGAD_BIN`

`CO_CATEGAD` (5 categorias nominais) é binarizada em `TP_CATEGAD_BIN` durante o pré-processamento: `{1,2,3} → 0` (Pública), `{4,5} → 1` (Privada). Usar CO_CATEGAD como preditora contínua seria metodologicamente incorreto.

---

## Dependências principais

| Biblioteca | Uso |
|---|---|
| `statsmodels` | Regressão OLS confirmatória (p-valores, IC, testes F) |
| `shap` | LinearExplainer — explicabilidade dos coeficientes |
| `scipy` | Shapiro-Wilk, probplot |
| `statsmodels.stats` | Breusch-Pagan, VIF |
| `patsy` | Reconstrói design matrix para SHAP com interações |
| `streamlit` | Interface web |
| `fpdf2` | Geração do relatório PDF com fonte Arial (Unicode) |
| `plotly` | Gráficos interativos (resíduos, QQ-plot, SHAP) |

---

## Reprodutibilidade

Cada sessão gera um **Exec-ID** único (UUID). A divisão treino/teste usa `random_state=42`. O Shapiro-Wilk com n > 5.000 amostra exatamente 5.000 observações com `seed=42`. Esses parâmetros garantem que os resultados sejam reprodutíveis dado o mesmo conjunto de dados e hipótese.

---

## Dados brutos

Fonte: [INEP — Microdados ENADE 2021](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enade)

Os dados não são distribuídos neste repositório por restrições de tamanho e LGPD. Após download, extraia na raiz do projeto mantendo a estrutura de pastas original do INEP.
