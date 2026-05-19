"""
Gera uma amostra estratificada válida dos microdados ENADE para demonstração.

A amostra é estratificada por CO_GRUPO (tipo de curso) para garantir que todos
os cursos fiquem representados proporcionalmente — ao contrário de um simples
head(N) que capturaria apenas os primeiros cursos da ordenação.

Uso:
    python gerar_amostra.py enade_2021_completo.csv.gz
    python gerar_amostra.py enade_2021_completo.csv.gz --fracao 0.15
    python gerar_amostra.py enade_2021_completo.csv.gz --output demo_enade.csv
    python gerar_amostra.py enade_2021_completo.csv.gz --comprimido   # gera .csv.gz

Saída padrão: enade_2021_amostra_10pct.csv  (nesta mesma pasta)
Tamanho esperado: ~50–80 MB  |  ~35.000 registros com fracao=0.10
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Gera amostra estratificada dos microdados ENADE"
    )
    parser.add_argument(
        "input",
        help="Arquivo CSV ou CSV.GZ de entrada (gerado por gerar_csv_gz.py)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Arquivo de saída (padrão: enade_{ANO}_amostra_{N}pct.csv)",
    )
    parser.add_argument(
        "--comprimido", action="store_true",
        help="Gera .csv.gz em vez de .csv (menor, mas requer descompressão).",
    )
    parser.add_argument(
        "--fracao", "-f",
        type=float,
        default=0.10,
        help="Fração da amostra por curso (padrão: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (padrão: 42)",
    )
    parser.add_argument(
        "--minimo-por-curso",
        type=int,
        default=100,
        dest="minimo",
        help="Mínimo de registros por curso (padrão: 100 — evita cursos com amostra zero)",
    )
    args = parser.parse_args()

    if not (0 < args.fracao < 1):
        print("Erro: --fracao deve ser entre 0 e 1 (ex: 0.10 para 10%%)")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        # Tenta relativo à pasta deste script
        alt = Path(__file__).parent / args.input
        if alt.exists():
            input_path = alt
        else:
            print(f"Arquivo não encontrado: {args.input}")
            sys.exit(1)

    pct_label = int(args.fracao * 100)
    ext = ".csv.gz" if args.comprimido else ".csv"
    if args.output:
        output = Path(args.output)
    else:
        stem = input_path.stem.replace(".csv", "")
        output = input_path.parent / f"{stem}_amostra_{pct_label}pct{ext}"

    compression_in = "gzip" if str(input_path).endswith(".gz") else None
    compression_out = "gzip" if args.comprimido else None

    print("=" * 60)
    print(f"  E-XplainENADE — Gerador de Amostra Estratificada")
    print("=" * 60)
    print(f"\n  Entrada : {input_path.name}")
    print(f"  Saída   : {output.name}")
    print(f"  Fração  : {args.fracao:.0%} por curso (mín. {args.minimo} por curso)")
    print(f"  Seed    : {args.seed}")
    print()

    t0 = time.time()
    print("Lendo arquivo...")
    df = pd.read_csv(
        input_path, dtype=str, encoding="utf-8",
        compression=compression_in, low_memory=False,
    )
    print(f"  {len(df):,} registros · {len(df.columns)} colunas · {time.time()-t0:.1f}s")

    # Amostragem estratificada por CO_GRUPO
    if "CO_GRUPO" in df.columns:
        print(f"\nAmostrando {args.fracao:.0%} de cada curso (CO_GRUPO)...")
        grupos = df["CO_GRUPO"].value_counts()
        print(f"  {len(grupos)} cursos encontrados")

        partes = []
        for grupo, total in grupos.items():
            subset = df[df["CO_GRUPO"] == grupo]
            n_amostrar = max(args.minimo, int(total * args.fracao))
            n_amostrar = min(n_amostrar, total)  # não amostrar mais do que existe
            partes.append(subset.sample(n=n_amostrar, random_state=args.seed))

        amostra = pd.concat(partes).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"  Amostra final: {len(amostra):,} registros")
    else:
        # Fallback: amostra aleatória simples
        print("Aviso: CO_GRUPO não encontrado — usando amostra aleatória simples.")
        amostra = df.sample(frac=args.fracao, random_state=args.seed).reset_index(drop=True)
        print(f"  Amostra: {len(amostra):,} registros")

    # Verifica distribuição dos principais campos
    if "CO_GRUPO" in amostra.columns:
        print("\n  Distribuição por curso na amostra:")
        dist = amostra["CO_GRUPO"].value_counts().head(10)
        for grupo, n in dist.items():
            print(f"    CO_GRUPO {grupo}: {n:,} registros")
        if len(dist) < amostra["CO_GRUPO"].nunique():
            print(f"    ... e mais {amostra['CO_GRUPO'].nunique() - len(dist)} cursos")

    modo = "comprimindo com gzip" if compression_out else "sem compressão"
    print(f"\nSalvando em {output.name} ({modo})...")
    amostra.to_csv(output, index=False, encoding="utf-8", compression=compression_out)

    tamanho_mb = output.stat().st_size / 1024 ** 2
    print(f"\n  Concluído em {time.time()-t0:.1f}s")
    print(f"  Arquivo : {output.resolve()}")
    print(f"  Tamanho : {tamanho_mb:.1f} MB")
    print(f"  Linhas  : {len(amostra):,}")
    print()
    print("Próximo passo: faça upload deste arquivo na Etapa 1 do E-XplainENADE.")


if __name__ == "__main__":
    main()
