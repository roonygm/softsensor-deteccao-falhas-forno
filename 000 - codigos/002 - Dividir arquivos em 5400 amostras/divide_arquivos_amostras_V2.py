import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CAMINHO_CSV = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\coleta_18.csv"
PASTA_SAIDA = r"C:\Dados\Usina_7_Forno\003 - Dados CSV\18"
TAMANHO_BLOCO = 7300

DESCARTA_INICIO = 50
DESCARTA_FIM = 50

SEP = ";"
DECIMAL = ","
ENCODING = "utf-8-sig"

MAX_FALTAS = 600   # até 180 amostras faltando (gaps até 181s entre pontos)
RUIDO_REL = 0.02
SEED = 42

# =========================
# NOVO: limpeza pós-gap
# =========================
GAP_STALE_MIN = 3        # se ficar >3s sem amostra, considera retorno suspeito
DESCARTA_POS_GAP = 1     # remove 1ª amostra após o gap (pode colocar 2, 3...)
# =========================

# ============================================================
# 1) LEITURA E PREPARO
# ============================================================
df = pd.read_csv(
    CAMINHO_CSV,
    sep=SEP,
    decimal=DECIMAL,
    encoding=ENCODING,
    low_memory=False
)

df.columns = df.columns.str.strip()

if "Timestamp" not in df.columns:
    raise ValueError("Coluna 'Timestamp' não encontrada!")

df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
df = df.drop_duplicates(subset=["Timestamp"], keep="first").reset_index(drop=True)

# Define colunas numéricas
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_nao_numericas = [c for c in df.columns if c not in (["Timestamp"] + colunas_numericas)]

# ============================================================
# 1.5) (NOVO) REMOVE AMOSTRAS "VELHAS" NO RETORNO APÓS GAP
# ============================================================
def remover_amostras_pos_gap(df_in: pd.DataFrame, gap_min_s: int, n_descartar: int) -> pd.DataFrame:
    if n_descartar <= 0:
        return df_in

    df2 = df_in.copy()
    dt = df2["Timestamp"].diff().dt.total_seconds()

    # onde começa um gap (a linha atual é a primeira após o gap)
    idx_retorno = df2.index[(dt > gap_min_s)].to_list()

    # remove N linhas a partir de cada retorno
    idx_remover = set()
    for idx in idx_retorno:
        for k in range(n_descartar):
            j = idx + k
            if j in df2.index:
                idx_remover.add(j)

    if idx_remover:
        df2 = df2.drop(index=sorted(idx_remover)).reset_index(drop=True)

    return df2

df = remover_amostras_pos_gap(df, GAP_STALE_MIN, DESCARTA_POS_GAP)

# ============================================================
# 2) IDENTIFICA TRECHOS (quebra quando gap > MAX_FALTAS+1)
# ============================================================
dt = df["Timestamp"].diff().dt.total_seconds()
novo_trecho = dt.isna() | (dt > (MAX_FALTAS + 1))
df["_trecho_id"] = novo_trecho.cumsum()

rng = np.random.default_rng(SEED)

# ============================================================
# 3) RECONSTRÓI 1Hz NO TRECHO + INTERPOLA + RUIDO NAS LINHAS INSERIDAS
# ============================================================
def reconstruir_1hz_com_ruido(g: pd.DataFrame) -> pd.DataFrame:
    g = g.drop(columns=["_trecho_id"]).copy()
    g = g.set_index("Timestamp")

    idx_1hz = pd.date_range(g.index.min(), g.index.max(), freq="1s")

    # Índices originais
    idx_original = g.index

    # Reindexa para inserir timestamps faltantes
    g2 = g.reindex(idx_1hz)

    # Marca linhas inseridas
    inseridas = ~g2.index.isin(idx_original)

    # Interpola colunas numéricas
    if colunas_numericas:
        g2[colunas_numericas] = g2[colunas_numericas].interpolate(
            method="time",
            limit_direction="both"
        )

        # Ruído proporcional à variabilidade local
        win = 300  # ~5 minutos
        std_local = g2[colunas_numericas].rolling(
            win, min_periods=30, center=True
        ).std()

        std_global = g2[colunas_numericas].std().replace(0, np.nan)
        std_use = std_local.fillna(std_global)

        ruido = rng.normal(
            loc=0.0,
            scale=(RUIDO_REL * std_use),
            size=g2[colunas_numericas].shape
        )

        # Aplica ruído SOMENTE nas linhas inseridas
        g2.loc[inseridas, colunas_numericas] += ruido[inseridas, :]

    # Colunas não numéricas
    if colunas_nao_numericas:
        g2[colunas_nao_numericas] = g2[colunas_nao_numericas].ffill().bfill()

    g2 = g2.reset_index().rename(columns={"index": "Timestamp"})

    # Auditoria
    g2["_inserida"] = inseridas

    return g2

# ============================================================
# 4) SALVA BLOCOS POR TRECHO (SEM CRUZAR LACUNAS)
# ============================================================
os.makedirs(PASTA_SAIDA, exist_ok=True)

total_blocos = 0
linhas_salvas = 0
linhas_lidas = len(df)
linhas_inseridas_total = 0

for trecho_id, g in df.groupby("_trecho_id", sort=True):
    g_1hz = reconstruir_1hz_com_ruido(g)

    # segurança: checa 1Hz
    diffs = g_1hz["Timestamp"].diff().dropna()
    if not (diffs == pd.Timedelta(seconds=1)).all():
        print(f"[ALERTA] Trecho {trecho_id} não ficou 100% 1Hz. Ignorando trecho.")
        continue

    linhas_inseridas_total += int(g_1hz["_inserida"].sum())

    n_blocos = len(g_1hz) // TAMANHO_BLOCO
    if n_blocos == 0:
        continue

    for b in range(n_blocos):
        ini = b * TAMANHO_BLOCO
        fim = (b + 1) * TAMANHO_BLOCO
        bloco = g_1hz.iloc[ini:fim].copy()

        if len(bloco) <= (DESCARTA_INICIO + DESCARTA_FIM):
            continue

        bloco = bloco.iloc[DESCARTA_INICIO:-DESCARTA_FIM].copy()

        diffs_bloco = bloco["Timestamp"].diff().dropna()
        if not (diffs_bloco == pd.Timedelta(seconds=1)).all():
            print(f"[ALERTA] Bloco inválido no trecho {trecho_id}. Pulando.")
            continue

        total_blocos += 1
        linhas_salvas += len(bloco)

        nome = f"bloco_1Hz_{total_blocos:03d}_trecho{trecho_id:03d}.csv"
        caminho = os.path.join(PASTA_SAIDA, nome)

        bloco.to_csv(
            caminho,
            sep=SEP,
            decimal=DECIMAL,
            encoding=ENCODING,
            index=False
        )

# ============================================================
# 5) RESUMO
# ============================================================
print(f"Linhas lidas (Timestamp válido, pós-limpeza): {linhas_lidas}")
print(f"Linhas inseridas por interpolação: {linhas_inseridas_total}")
print(f"Blocos salvos ({TAMANHO_BLOCO} linhas antes do descarte de bordas): {total_blocos}")
print(f"Linhas salvas: {linhas_salvas}")
print(f"Saída em: {PASTA_SAIDA}")
