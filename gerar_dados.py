"""
Gera a base de dados embutida do recorte do TCC.

Recorte fixo (E-XplainENADE):
  - ENADE 2021
  - Regiões: Norte (1) + Nordeste (2)        [CO_REGIAO_CURSO]
  - Cursos: Ciência da Computação (4004) + Sistemas de Informação (4006)  [CO_GRUPO]
  - Apenas presentes (TP_PRES == 555)

Saída: dados/enade_2021_nne_ccsi.csv.gz  (~1–2 MB — versionado no git)

O arquivo mantém as COLUNAS BRUTAS dos microdados. Toda recodificação,
tipagem e renomeação continua centralizada em modules/loader.py (preprocess),
de modo que este script é apenas o ETL provisório do recorte — quando o
ENADE-Time (Lucas) entregar a base harmonizada, o SWAP POINT em
modules/etl.py substitui esta etapa sem alterar o restante do pipeline.

Uso:
    python gerar_dados.py
    python gerar_dados.py --fonte "caminho/para/2.DADOS"
    python gerar_dados.py --saida dados/meu_arquivo.csv.gz
"""
import argparse
import sys
import time
from pathlib import Path

from modules import etl

_ROOT = Path(__file__).parent

REGIOES = [1, 2]        # Norte + Nordeste
GRUPOS  = [4004, 4006]  # CC + SI
SAIDA_PADRAO = _ROOT / "dados" / "enade_2021_nne_ccsi.csv.gz"

# Locais onde os microdados brutos costumam estar neste projeto
_FONTES_PADRAO = [
    _ROOT / "microdados_Enade_2021_LGPD" / "2.DADOS",
    _ROOT / "gerar_csv" / "microdados_Enade_2021_LGPD" / "2.DADOS",
]


def _localizar_fonte(arg: str) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        sys.exit(f"Erro: pasta de origem não encontrada: {p}")
    for p in _FONTES_PADRAO:
        if p.exists():
            return p
    sys.exit(
        "Erro: microdados brutos não encontrados.\n"
        "Esperado em uma destas pastas:\n"
        + "\n".join(f"  - {p}" for p in _FONTES_PADRAO)
        + "\nOu informe o caminho com: python gerar_dados.py --fonte <pasta 2.DADOS>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gera a base embutida do recorte do TCC (2021 · N+NE · CC+SI · presentes)"
    )
    parser.add_argument("--fonte", "-f", default=None,
                        help="Pasta com os 43 arquivos microdados2021_arq*.txt")
    parser.add_argument("--saida", "-o", default=None,
                        help=f"Arquivo de saída (padrão: {SAIDA_PADRAO.relative_to(_ROOT)})")
    args = parser.parse_args()

    fonte = _localizar_fonte(args.fonte)
    saida = Path(args.saida) if args.saida else SAIDA_PADRAO

    print("=" * 64)
    print("  E-XplainENADE — Geração da base do recorte (ENADE 2021)")
    print("=" * 64)
    print(f"  Fonte   : {fonte}")
    print(f"  Saída   : {saida}")
    print(f"  Recorte : regiões {REGIOES} (N+NE) · grupos {GRUPOS} (CC+SI) · TP_PRES=555")
    print()

    t0 = time.time()
    print("Lendo os 43 arquivos brutos (filtro aplicado durante a carga)...")
    df = etl.load_raw(grupos=GRUPOS, regioes=REGIOES, raw_dir=fonte)
    print(f"  Após filtro de região/curso: {len(df):,} registros · "
          f"{len(df.columns)} colunas · {time.time() - t0:.1f}s")

    antes = len(df)
    df = df[df["TP_PRES"].astype(str).str.strip() == "555"].reset_index(drop=True)
    print(f"  Filtro TP_PRES==555: {antes - len(df):,} removidos -> {len(df):,} presentes")

    saida.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSalvando {len(df):,} linhas × {len(df.columns)} colunas (gzip)...")
    df.to_csv(saida, index=False, encoding="utf-8", compression="gzip")

    tamanho_mb = saida.stat().st_size / 1024 ** 2
    print(f"\n  Concluído em {time.time() - t0:.1f}s")
    print(f"  Arquivo : {saida.resolve()}")
    print(f"  Tamanho : {tamanho_mb:.2f} MB")
    print(f"  Linhas  : {len(df):,}")
    print()
    print("Pronto. O E-XplainENADE carrega este arquivo automaticamente (streamlit run app.py).")


if __name__ == "__main__":
    main()
