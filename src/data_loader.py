"""
Data ingestion layer for E-XplainENADE.

SWAP POINT: quando Lucas entregar os dados normalizados, substituir
_build_dataset() por uma função que leia o formato dele.
A assinatura pública load_dataset() e o schema do DataFrame retornado
devem permanecer os mesmos para não quebrar o restante do sistema.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── Configuração ───────────────────────────────────────────────────────────────

_RAW_DIR = (
    Path(__file__).parent.parent
    / "microdados_Enade_2021_LGPD"
    / "2.DADOS"
)

# Códigos CO_GRUPO — confirmar no arquivo
# Dicionário_arquivos_variáveis_microdados_Enade_2021.xlsx (pasta 1.LEIA-ME)
CO_GRUPO_CC = 4004  # Ciência da Computação
CO_GRUPO_SI = 4006  # Sistemas de Informação

DEFAULT_GRUPOS = [CO_GRUPO_CC, CO_GRUPO_SI]
DEFAULT_REGIOES = [1, 2]  # 1 = Norte, 2 = Nordeste

# Ordem de carregamento:
#   arq1  → dados institucionais + filtros (CO_GRUPO, CO_REGIAO_CURSO)
#   arq2  → dados de matrícula (ANO_IN_GRAD, CO_TURNO_GRADUACAO, ANO_FIM_EM)
#   arq3  → desempenho (NT_GER, NT_FG, NT_CE) + presença (TP_PRES)
#   arq5  → TP_SEXO
#   arq6  → NU_IDADE
#   arq7–32  → QE_I01–QE_I26 (questionário socioeconômico, parte 1)
#   arq4  → QE_I27–QE_I68 (questionário socioeconômico, parte 2)
#   arq33–43 → QE_I69–QE_I92 (questionário socioeconômico, parte 3)
_LOAD_ORDER = (
    [1, 2, 3, 5, 6]
    + list(range(7, 33))
    + [4]
    + list(range(33, 44))
)


# ── Funções internas ───────────────────────────────────────────────────────────

def _read_arq(n: int) -> pd.DataFrame:
    path = _RAW_DIR / f"microdados2021_arq{n}.txt"
    return pd.read_csv(path, sep=";", dtype=str, encoding="latin-1")


def _build_dataset(grupos: List[int], regioes: List[int]) -> pd.DataFrame:
    """Carrega e une os arq files por posição (row-aligned, sem chave de estudante — LGPD)."""
    base = _read_arq(1)

    mask = (
        base["CO_GRUPO"].astype(int).isin(grupos)
        & base["CO_REGIAO_CURSO"].astype(int).isin(regioes)
    )

    parts = [base[mask].reset_index(drop=True)]

    for n in _LOAD_ORDER[1:]:
        df = _read_arq(n)
        parts.append(df[mask].reset_index(drop=True).drop(columns=["NU_ANO", "CO_CURSO"]))

    return pd.concat(parts, axis=1)


# ── API pública ────────────────────────────────────────────────────────────────

def load_dataset(
    regioes: Optional[List[int]] = None,
    grupos: Optional[List[int]] = None,
    apenas_presentes: bool = True,
) -> pd.DataFrame:
    """
    Retorna DataFrame filtrado e pronto para modelagem.

    Parameters
    ----------
    regioes : list[int], optional
        Valores de CO_REGIAO_CURSO a incluir.
        Padrão: [1, 2] (Norte + Nordeste).
    grupos : list[int], optional
        Valores de CO_GRUPO a incluir.
        Padrão: [4004, 4006] (Ciência da Computação + Sistemas de Informação).
    apenas_presentes : bool
        Se True, mantém apenas estudantes presentes (TP_PRES == 555).
        Padrão: True.

    Returns
    -------
    pd.DataFrame
        Uma linha por estudante com todas as variáveis dos arq files unidos.
        Colunas principais:
          - NT_GER, NT_FG, NT_CE  → variáveis-alvo (notas)
          - CO_GRUPO, CO_REGIAO_CURSO, CO_CATEGAD, CO_ORGACAD → contexto institucional
          - TP_SEXO, NU_IDADE     → perfil demográfico
          - QE_I01–QE_I92        → questionário socioeconômico
    """
    _grupos = grupos if grupos is not None else DEFAULT_GRUPOS
    _regioes = regioes if regioes is not None else DEFAULT_REGIOES

    df = _build_dataset(_grupos, _regioes)

    if apenas_presentes:
        df = df[df["TP_PRES"] == "555"].reset_index(drop=True)

    return df
