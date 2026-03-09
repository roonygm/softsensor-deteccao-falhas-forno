import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PASTA_SAIDA = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\18"      # onde estão os CSVs gerados
PASTA_PLOTS = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\18" # onde salvar os plots

SEP = ";"
DECIMAL = ","
ENCODING = "utf-8-sig"

VARIAVEIS = [
    "Intouch.U7_TT2421_UE",
    "Intouch.U7_TT2423_UE",
    "Intouch.U7_TT2425_UE",
    "Intouch.FZIT_G55_2301_UE"
]

# ============================================================
# PREPARO
# ============================================================
os.makedirs(PASTA_PLOTS, exist_ok=True)

arquivos = sorted(glob.glob(os.path.join(PASTA_SAIDA, "bloco_1Hz_*.csv")))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo 'bloco_1Hz_*.csv' encontrado em: {PASTA_SAIDA}")

print(f"Encontrados {len(arquivos)} arquivos para plotar.")

# ============================================================
# PLOTA 1 FIGURA POR ARQUIVO
# ============================================================
for arq in arquivos:
    dfb = pd.read_csv(arq, sep=SEP, decimal=DECIMAL, encoding=ENCODING, low_memory=False)
    dfb.columns = dfb.columns.str.strip()

    if "Timestamp" not in dfb.columns:
        print(f"[PULANDO] Sem coluna Timestamp: {os.path.basename(arq)}")
        continue

    dfb["Timestamp"] = pd.to_datetime(dfb["Timestamp"], dayfirst=True, errors="coerce")
    dfb = dfb.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Checa variáveis existentes
    vars_ok = [v for v in VARIAVEIS if v in dfb.columns]
    vars_faltando = [v for v in VARIAVEIS if v not in dfb.columns]

    if len(vars_ok) == 0:
        print(f"[PULANDO] Nenhuma das 4 variáveis encontrada em: {os.path.basename(arq)}")
        continue

    # Plot
    plt.figure(figsize=(14, 6))
    for v in vars_ok:
        plt.plot(dfb["Timestamp"], dfb[v], label=v)

    titulo = os.path.basename(arq).replace(".csv", "")
    if vars_faltando:
        titulo += f" (faltando: {len(vars_faltando)})"

    plt.title(titulo)
    plt.xlabel("Timestamp")
    plt.ylabel("Valor")
    plt.legend(loc="best")
    plt.tight_layout()

    # Salva PNG
    nome_png = os.path.basename(arq).replace(".csv", ".png")
    caminho_png = os.path.join(PASTA_PLOTS, nome_png)
    plt.savefig(caminho_png, dpi=200, bbox_inches="tight")
    plt.close()

print(f"Plots salvos em: {PASTA_PLOTS}")
