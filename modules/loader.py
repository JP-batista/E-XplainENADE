"""
Camada 2 — Filtros, recodificação e pré-processamento.

Recebe o DataFrame bruto de etl.load_raw() (ou de um CSV via upload) e entrega
um DataFrame padronizado, com tipos corretos e colunas renomeadas, pronto para modelagem.
Esta camada permanece estável mesmo quando a fonte de dados mudar
(o SWAP POINT está em etl.py, ou via get_dataset_from_csv() para upload).
"""
from typing import IO, List, Optional, Union

import pandas as pd

from modules import etl
from config.variable_map import VARIABLE_MAP

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
    # CO_CATEGAD (nominal 1-5) binarizado: 1/2/3=Pública→0, 4/5=Privada→1
    # Uso como preditora contínua seria metodologicamente incorreto (não há ordenamento real)
    if "CO_CATEGAD" in df.columns:
        df["TP_CATEGAD_BIN"] = df["CO_CATEGAD"].map(
            {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        )
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
        CO_REGIAO_CURSO a incluir. None = todas as regiões.
    grupos : list[int], optional
        CO_GRUPO a incluir. None = todos os cursos.
    apenas_presentes : bool
        Se True, mantém apenas estudantes com TP_PRES == 555. Padrão: True.

    Returns
    -------
    pd.DataFrame
        Uma linha por estudante com colunas tipadas e nomes amigáveis.
        Variáveis-alvo: NT_GER, NT_FG, NT_CE.
        Preditores socioeconômicos: QE_RENDA, QE_ESC_MAE, QE_HORAS_ESTUDO, etc.
    """
    df = etl.load_raw(grupos=grupos, regioes=regioes)
    df = preprocess(df)

    if apenas_presentes:
        df = df[df["TP_PRES"] == 555].reset_index(drop=True)

    return df


def get_dataset_from_csv(
    csv_file: Union[str, IO],
    grupos: Optional[List[int]] = None,
    ies_filter: Optional[List[int]] = None,
    apenas_presentes: bool = True,
) -> pd.DataFrame:
    """
    Alternativa a get_dataset() para quando os dados vêm de um CSV gerado por gerar_csv.py.

    O CSV deve ter sido gerado com etl.load_raw() (colunas brutas: QE_I08, CO_GRUPO, etc.).
    O pré-processamento (recodificação, tipagem, renomeação) é aplicado aqui da mesma forma.

    Parameters
    ----------
    csv_file : str | file-like
        Caminho do CSV ou objeto de arquivo (ex: UploadedFile do Streamlit).
    grupos : list[int], optional
        CO_GRUPO a incluir. None = todos os cursos.
    ies_filter : list[int], optional
        Valores de CO_CATEGAD a incluir. None = todos os tipos de IES.
    apenas_presentes : bool
        Se True, mantém apenas TP_PRES == 555. Padrão: True.

    Returns
    -------
    pd.DataFrame
        Mesmo formato de get_dataset() — pronto para modelagem.
    """
    # Detecta compressão pelo nome do arquivo (suporta .csv e .csv.gz)
    nome = getattr(csv_file, "name", str(csv_file))
    compression = "gzip" if str(nome).endswith(".gz") else None
    df = pd.read_csv(csv_file, dtype=str, encoding="utf-8",
                     low_memory=False, compression=compression)
    df = preprocess(df)

    if grupos:
        df = df[df["CO_GRUPO"].isin(grupos)].reset_index(drop=True)
    if ies_filter:
        df = df[df["CO_CATEGAD"].isin(ies_filter)].reset_index(drop=True)
    if apenas_presentes:
        df = df[df["TP_PRES"] == 555].reset_index(drop=True)

    return df
