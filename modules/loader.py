"""
Camada 2 — Filtros, recodificação e pré-processamento.

Recebe o DataFrame bruto de etl.load_raw() e entrega um DataFrame
padronizado, com tipos corretos e colunas renomeadas, pronto para modelagem.
Esta camada permanece estável mesmo quando a fonte de dados mudar
(o SWAP POINT está em etl.py).
"""
from typing import List, Optional

import pandas as pd

from modules import etl
from config.variable_map import VARIABLE_MAP

_DEFAULT_GRUPOS = [4004, 4006]  # Ciência da Computação + Sistemas de Informação
_DEFAULT_REGIOES = [1, 2]       # Norte + Nordeste

# Mapeamento letra → inteiro para QE_I01–QE_I26 (A-E/F/G/H conforme a questão)
_LETTER_TO_INT = {chr(65 + i): i + 1 for i in range(26)}  # A=1 … Z=26


def _recode_questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recodifica variáveis do questionário:
    - QE_I01–I26 (letras A-H): A→1, B→2, ... (mapeamento ordinal alfabético)
    - QE_I27–I68 (Likert '1'–'6'): converte para int; 7/8 → NaN (não sei/não se aplica)
    - QE_I16 (código IBGE da UF): converte para int, não reordena
    - QE_I69+ (pandemia): converte para int se possível
    """
    for col in df.columns:
        if not col.startswith("QE_I"):
            continue

        try:
            idx = int(col.replace("QE_I", ""))
        except ValueError:
            continue

        if 1 <= idx <= 26:
            if idx == 16:
                # UF do EM — código numérico IBGE, não ordinal
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].map(_LETTER_TO_INT)
        elif 27 <= idx <= 68:
            # Likert 1–6; 7 = "Não sei responder", 8 = "Não se aplica" → NaN
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.where(numeric <= 6, other=pd.NA)
        else:
            # QE_I69+ (questões de pandemia): recodifica letras se necessário
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if first_valid and str(first_valid).isalpha():
                df[col] = df[col].map(_LETTER_TO_INT)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas para os tipos numéricos corretos."""
    qe_cols = {c for c in df.columns if c.startswith("QE_I")}

    for col in df.columns:
        if col in qe_cols:
            continue
        if col.startswith("NT_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col == "TP_SEXO":
            df[col] = df[col].map({"F": 0, "M": 1})
        elif not col.startswith("DS_"):
            # DS_* são strings de gabarito/resposta — não usar em modelagem
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas brutas para os códigos amigáveis definidos no variable_map."""
    rename = {
        raw: info["code"]
        for raw, info in VARIABLE_MAP.items()
        if raw in df.columns and info["code"] != raw
    }
    return df.rename(columns=rename)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica recodificação, conversão de tipos e renomeação de colunas.
    Entrada: DataFrame bruto de etl.load_raw().
    Saída: DataFrame tipado com nomes amigáveis, pronto para modelagem.
    """
    df = df.copy()
    df = _recode_questionnaire(df)
    df = _convert_types(df)
    df = _rename_columns(df)
    return df


def get_dataset(
    regioes: Optional[List[int]] = None,
    grupos: Optional[List[int]] = None,
    apenas_presentes: bool = True,
) -> pd.DataFrame:
    """
    Ponto de entrada principal do pipeline de dados.
    Retorna DataFrame filtrado, tipado e renomeado, pronto para modelagem.

    Parameters
    ----------
    regioes : list[int], optional
        CO_REGIAO_CURSO a incluir. Padrão: [1, 2] (Norte + Nordeste).
    grupos : list[int], optional
        CO_GRUPO a incluir. Padrão: [4004, 4006] (CC + SI).
    apenas_presentes : bool
        Se True, mantém apenas estudantes com TP_PRES == 555. Padrão: True.

    Returns
    -------
    pd.DataFrame
        Uma linha por estudante com colunas tipadas e nomes amigáveis.
        Variáveis-alvo: NT_GER, NT_FG, NT_CE.
        Preditores socioeconômicos: QE_RENDA, QE_ESC_MAE, QE_HORAS_ESTUDO, etc.
    """
    _grupos = grupos if grupos is not None else _DEFAULT_GRUPOS
    _regioes = regioes if regioes is not None else _DEFAULT_REGIOES

    df = etl.load_raw(grupos=_grupos, regioes=_regioes)
    df = preprocess(df)

    if apenas_presentes:
        df = df[df["TP_PRES"] == 555].reset_index(drop=True)

    return df
