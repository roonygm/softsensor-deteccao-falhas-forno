import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# CONFIG
# ============================================================
PASTA_SAIDA = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h"
PASTA_PLOTS = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\Plots"

SEP = ";"
DECIMAL = ","
ENCODING = "utf-8-sig"

VARIAVEIS_TEMPERATURA = [
    "Intouch.U7_TT2421_UE",
    "Intouch.U7_TT2423_UE",
    "Intouch.U7_TT2425_UE"
]

VARIAVEL_VAZAO = "Intouch.FZIT_G55_2301_UE"

# ============================================================
# PREPARO
# ============================================================
os.makedirs(PASTA_PLOTS, exist_ok=True)

arquivos = sorted(glob.glob(os.path.join(PASTA_SAIDA, "*.csv")))

if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo encontrado em: {PASTA_SAIDA}")

print(f"Encontrados {len(arquivos)} arquivos para plotar.")


# ============================================================
# FUÇÃO FAIXA DINAMICA EIXO Y
# ============================================================

def limites_dinamicos(serie, margem=0.05):
    s = pd.to_numeric(serie, errors="coerce").dropna()

    if len(s) == 0:
        return None, None

    vmin = s.min()
    vmax = s.max()

    faixa = vmax - vmin

    if faixa == 0:
        faixa = abs(vmax) * 0.1

    return vmin - faixa * margem, vmax + faixa * margem

# ============================================================
# PLOTA 1 FIGURA POR ARQUIVO
# ============================================================
for arq in arquivos:
    dfb = pd.read_csv(
        arq,
        sep=SEP,
        decimal=DECIMAL,
        encoding=ENCODING,
        low_memory=False
    )
    dfb.columns = dfb.columns.str.strip()

    if "Timestamp" not in dfb.columns:
        print(f"[PULANDO] Sem coluna Timestamp: {os.path.basename(arq)}")
        continue

    dfb["Timestamp"] = pd.to_datetime(dfb["Timestamp"], dayfirst=True, errors="coerce")
    dfb = dfb.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Variáveis disponíveis
    temps_ok = [v for v in VARIAVEIS_TEMPERATURA if v in dfb.columns]
    vazao_ok = VARIAVEL_VAZAO in dfb.columns

    if not temps_ok and not vazao_ok:
        print(f"[PULANDO] Nenhuma variável encontrada em: {os.path.basename(arq)}")
        continue

    # ========================================================
    # FIGURA COM 2 SUBPLOTS
    # ========================================================
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # --------------------------------------------------------
    # GRÁFICO SUPERIOR - TEMPERATURAS
    # --------------------------------------------------------
    for v in temps_ok:
        ax1.plot(dfb["Timestamp"], pd.to_numeric(dfb[v], errors="coerce"), label=v)

    ax1.set_title(os.path.basename(arq).replace(".csv", ""))
    ax1.set_ylabel("Temperatura (°C)")
    temp_series = pd.concat([pd.to_numeric(dfb[v], errors="coerce") for v in temps_ok])

    ymin, ymax = limites_dinamicos(temp_series, margem=0.08)

    if ymin is not None:
        ax1.set_ylim(ymin, ymax)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # --------------------------------------------------------
    # GRÁFICO INFERIOR - VAZÃO
    # --------------------------------------------------------
    if vazao_ok:
        ax2.plot(
            dfb["Timestamp"],
            pd.to_numeric(dfb[VARIAVEL_VAZAO], errors="coerce"),
            color="red",
            label=VARIAVEL_VAZAO
        )

    ax2.set_ylabel("Vazão (m³/h)")
    ax2.set_xlabel("Hora")
    if vazao_ok:
        ymin, ymax = limites_dinamicos(dfb[VARIAVEL_VAZAO], margem=0.10)

        if ymin is not None:
            ax2.set_ylim(ymin, ymax)
    ax2.grid(True, alpha=0.3)

    if vazao_ok:
        ax2.legend(loc="best")

    # --------------------------------------------------------
    # FORMATAÇÃO DO EIXO X: SOMENTE HORA
    # --------------------------------------------------------
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # opcional: controla espaçamento dos ticks
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.setp(ax2.get_xticklabels(), rotation=0)

    plt.tight_layout()

    # ========================================================
    # SALVA PNG
    # ========================================================
    nome_png = os.path.basename(arq).replace(".csv", ".png")
    caminho_png = os.path.join(PASTA_PLOTS, nome_png)
    plt.savefig(caminho_png, dpi=200, bbox_inches="tight")
    plt.close()

print(f"Plots salvos em: {PASTA_PLOTS}")
