
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DATA_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo'
OUT_DIR = os.path.join(DATA_DIR, 'correlacao_cruzada_atrasos_resumo')
os.makedirs(OUT_DIR, exist_ok=True)

MAX_FILES = 10
TEMP_MAX_LAG = 200
GAS_MIN_LAG = 0
GAS_MAX_LAG = 210
NORMALIZE_SERIES = True

# =========================================================
# ARQUIVOS
# =========================================================
FILES = sorted(
    f for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)[:MAX_FILES]

if len(FILES) == 0:
    raise RuntimeError('Nenhum CSV encontrado. Verifique DATA_DIR.')

# =========================================================
# LEITURA
# =========================================================
def read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8', engine='python')
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]

    rename_map = {
        'Intouch.U7_TT2421_UE': 'TL',
        'Intouch.U7_TT2425_UE': 'TR',
        'Intouch.U7_TT2423_UE': 'TC',
        'Intouch.FZIT_G55_2301_UE': 'GAS'
    }
    df = df.rename(columns=rename_map)

    required = ['TL', 'TR', 'TC', 'GAS']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Colunas faltando em {path}: {missing}')

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.dropna().reset_index(drop=True)

# =========================================================
# FUNÇÕES
# =========================================================
def standardize(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s

# correlação entre x(k-lag) e y(k)
def lagged_corr(x, y, lag):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) <= lag + 5 or len(y) <= lag + 5:
        return np.nan

    x_lag = x[:-lag] if lag > 0 else x.copy()
    y_cut = y[lag:] if lag > 0 else y.copy()

    n = min(len(x_lag), len(y_cut))
    x_lag = x_lag[:n]
    y_cut = y_cut[:n]

    if NORMALIZE_SERIES:
        x_lag = standardize(x_lag)
        y_cut = standardize(y_cut)

    if np.std(x_lag) < 1e-12 or np.std(y_cut) < 1e-12:
        return np.nan

    return np.corrcoef(x_lag, y_cut)[0, 1]

def sweep_corr(x, y, lags):
    vals = []
    for lag in lags:
        vals.append(lagged_corr(x, y, lag))
    return np.array(vals, dtype=float)

# =========================================================
# PROCESSAMENTO
# =========================================================
tl_lags = np.arange(0, TEMP_MAX_LAG + 1)
tr_lags = np.arange(0, TEMP_MAX_LAG + 1)
gas_lags = np.arange(GAS_MIN_LAG, GAS_MAX_LAG + 1)

all_tl = []
all_tr = []
all_gas = []
summary_rows = []

for f in FILES:
    df = read_one_csv(f)
    tl = df['TL'].values
    tr = df['TR'].values
    tc = df['TC'].values
    gas = df['GAS'].values

    corr_tl = sweep_corr(tl, tc, tl_lags)
    corr_tr = sweep_corr(tr, tc, tr_lags)
    corr_gas = sweep_corr(gas, tc, gas_lags)

    all_tl.append(corr_tl)
    all_tr.append(corr_tr)
    all_gas.append(corr_gas)

    best_tl_lag = int(tl_lags[np.nanargmax(np.abs(corr_tl))])
    best_tr_lag = int(tr_lags[np.nanargmax(np.abs(corr_tr))])
    best_gas_lag = int(gas_lags[np.nanargmax(np.abs(corr_gas))])

    summary_rows.append({
        'arquivo': os.path.basename(f),
        'best_tl_lag': best_tl_lag,
        'best_tr_lag': best_tr_lag,
        'best_gas_lag': best_gas_lag,
        'best_tl_corr': float(corr_tl[np.nanargmax(np.abs(corr_tl))]),
        'best_tr_corr': float(corr_tr[np.nanargmax(np.abs(corr_tr))]),
        'best_gas_corr': float(corr_gas[np.nanargmax(np.abs(corr_gas))]),
    })

all_tl = np.array(all_tl, dtype=float)
all_tr = np.array(all_tr, dtype=float)
all_gas = np.array(all_gas, dtype=float)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(OUT_DIR, '00_resumo_por_arquivo.csv'),
    sep=';', decimal=',', index=False, encoding='utf-8-sig'
)

# =========================================================
# AGREGAÇÃO
# =========================================================
mean_tl = np.nanmean(all_tl, axis=0)
std_tl = np.nanstd(all_tl, axis=0)

mean_tr = np.nanmean(all_tr, axis=0)
std_tr = np.nanstd(all_tr, axis=0)

mean_gas = np.nanmean(all_gas, axis=0)
std_gas = np.nanstd(all_gas, axis=0)

best_tl_lag_global = int(tl_lags[np.nanargmax(np.abs(mean_tl))])
best_tr_lag_global = int(tr_lags[np.nanargmax(np.abs(mean_tr))])
best_gas_lag_global = int(gas_lags[np.nanargmax(np.abs(mean_gas))])

resumo_global = pd.DataFrame([{
    'n_arquivos': len(FILES),
    'best_tl_lag_global': best_tl_lag_global,
    'best_tr_lag_global': best_tr_lag_global,
    'best_gas_lag_global': best_gas_lag_global,
    'media_best_tl_lag_por_arquivo': summary_df['best_tl_lag'].mean(),
    'mediana_best_tl_lag_por_arquivo': summary_df['best_tl_lag'].median(),
    'media_best_tr_lag_por_arquivo': summary_df['best_tr_lag'].mean(),
    'mediana_best_tr_lag_por_arquivo': summary_df['best_tr_lag'].median(),
    'media_best_gas_lag_por_arquivo': summary_df['best_gas_lag'].mean(),
    'mediana_best_gas_lag_por_arquivo': summary_df['best_gas_lag'].median(),
}])

resumo_global.to_csv(
    os.path.join(OUT_DIR, '01_resumo_global.csv'),
    sep=';', decimal=',', index=False, encoding='utf-8-sig'
)

# =========================================================
# GRÁFICO ÚNICO
# =========================================================
fig = plt.figure(figsize=(12, 10))

# TL
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(tl_lags, mean_tl, marker='o', label='Correlação média')
ax1.axvline(best_tl_lag_global, linestyle='--', label=f"melhor atraso global = {best_tl_lag_global}")
ax1.set_title('Correlação cruzada média: TL → TC')
ax1.set_xlabel('Atraso de TL (amostras)')
ax1.set_ylabel('Correlação')
ax1.grid(True, alpha=0.3)
ax1.legend()

# TR
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(tr_lags, mean_tr, marker='o', label='Correlação média')
ax2.axvline(best_tr_lag_global, linestyle='--', label=f"melhor atraso global = {best_tr_lag_global}")
ax2.set_title('Correlação cruzada média: TR → TC')
ax2.set_xlabel('Atraso de TR (amostras)')
ax2.set_ylabel('Correlação')
ax2.grid(True, alpha=0.3)
ax2.legend()

# GAS
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(gas_lags, mean_gas, marker='o', label='Correlação média')
ax3.axvline(best_gas_lag_global, linestyle='--', label=f"melhor atraso global = {best_gas_lag_global}")
ax3.set_title('Correlação cruzada média: GAS → TC')
ax3.set_xlabel('Atraso de GAS (amostras)')
ax3.set_ylabel('Correlação')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_correlacao_cruzada_resumo_unico.png'), dpi=150)
plt.close()