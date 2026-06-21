"""
Camada 1 — Ingestão e ETL dos microdados brutos.

SWAP POINT: quando Lucas (ENADE-Time) entregar os dados normalizados,
substituir _load_raw_files() por uma função que leia o novo formato.
A função pública load_raw() mantém a mesma assinatura e schema de saída.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

_RAW_DIR = (
    Path(__file__).parent.parent
    / "microdados_Enade_2021_LGPD"
    / "2.DADOS"
)

# Ordem garante que QE_I01–QE_I92 fiquem em sequência no DataFrame final.
# arq7–32: QE_I01–I26 | arq4: QE_I27–I68 | arq33–43: QE_I69–I92
_LOAD_ORDER = (
    [1, 2, 3, 5, 6]
    + list(range(7, 33))
    + [4]
    + list(range(33, 44))
)


def _read_arq(n: int, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"microdados2021_arq{n}.txt"
    return pd.read_csv(path, sep=";", dtype=str, encoding="latin-1")


def load_raw(
    grupos: Optional[List[int]] = None,
    regioes: Optional[List[int]] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Lê os 43 arquivos row-aligned do ENADE 2021, une por posição e retorna
    o DataFrame bruto com todas as colunas.

    Join posicional: linha N de arq1 = linha N de arq2 = ... (sem ID de
    estudante — LGPD). O filtro de grupos/regioes é aplicado durante o
    carregamento para eficiência de memória (evita carregar ~500 MB completos).

    Parameters
    ----------
    grupos : list[int], optional
        Valores de CO_GRUPO a incluir (ex: [4004, 4006] para CC + SI).
    regioes : list[int], optional
        Valores de CO_REGIAO_CURSO a incluir (ex: [1, 2] para Norte + Nordeste).
    raw_dir : Path, optional
        Pasta com os 43 arquivos microdados2021_arq*.txt. Padrão: _RAW_DIR.

    Returns
    -------
    pd.DataFrame
        DataFrame bruto com todas as colunas, sem recodificação de tipos.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else _RAW_DIR
    base = _read_arq(1, raw_dir)

    mask = pd.Series([True] * len(base), index=base.index)
    if grupos:
        mask &= base["CO_GRUPO"].astype(int).isin(grupos)
    if regioes:
        mask &= base["CO_REGIAO_CURSO"].astype(int).isin(regioes)

    parts = [base[mask].reset_index(drop=True)]

    for n in _LOAD_ORDER[1:]:
        df = _read_arq(n, raw_dir)
        parts.append(
            df[mask].reset_index(drop=True).drop(columns=["NU_ANO", "CO_CURSO"])
        )

    return pd.concat(parts, axis=1)
