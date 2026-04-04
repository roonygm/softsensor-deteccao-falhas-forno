import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DATA_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo'
OUT_DIR = os.path.join(DATA_DIR, 'plots_modelo_3_real_vs_estimado')
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_MAX_POINTS = None   # use None para plotar tudo
SAVE_DPI = 150

# =========================================================
# PARÂMETROS DO MODELO 3
# =========================================================
ALPHA = 0.9945
BETA_GF = 0.0055   # 1 - alpha

C = 27.49
A_TL = 0.402
B_TR = 0.551
D_GF = 0.094

TL_DELAY = 1
TR_DELAY = 1

# =========================================================
# LEITURA DOS ARQUIVOS
# =========================================================
FILES = sorted(
    f for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

if len(FILES) == 0:
    raise RuntimeError('Nenhum arquivo CSV numérico foi encontrado na pasta.')

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

    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# FILTRO DO GÁS
# =========================================================
def first_order_filter_gas(gas, alpha=0.9945):
    gas = np.asarray(gas, dtype=float)
    gf = np.zeros_like(gas)
    gf[0] = gas[0]

    for k in range(1, len(gas)):
        gf[k] = alpha * gf[k - 1] + (1 - alpha) * gas[k]

    return gf

# =========================================================
# PREDIÇÃO DO MODELO 3
# =========================================================
def predict_model_3(df):
    """
    Modelo 3:
        Gf(k) = 0.9945*Gf(k-1) + 0.0055*GAS(k)

        TChat(k) = 27.49
                   + 0.402*TL(k-1)
                   + 0.551*TR(k-1)
                   + 0.094*Gf(k)
    """
    if len(df) <= 1:
        raise ValueError('Arquivo muito curto para aplicar atraso de 1 amostra.')

    tl = df['TL'].values
    tr = df['TR'].values
    tc = df['TC'].values
    gas = df['GAS'].values

    gf = first_order_filter_gas(gas, alpha=ALPHA)

    # alinhamento para k >= 1
    y_real = tc[1:]
    y_est = (
        C
        + A_TL * tl[:-1]
        + B_TR * tr[:-1]
        + D_GF * gf[1:]
    )

    return y_real, y_est, gf

# =========================================================
# MÉTRICAS OPCIONAIS
# =========================================================
def calc_metrics(y, yhat):
    err = y - yhat
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))

    ss_res = np.sum(err**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return rmse, mae, r2

# =========================================================
# PLOT
# =========================================================
def plot_real_vs_est(y_real, y_est, file_name, save_path):
    if PLOT_MAX_POINTS is not None:
        n = min(len(y_real), PLOT_MAX_POINTS)
        y_real = y_real[:n]
        y_est = y_est[:n]

    x = np.arange(len(y_real))

    rmse, mae, r2 = calc_metrics(y_real, y_est)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_real, label='TC real')
    plt.plot(x, y_est, label='TC estimado (Modelo 3)')

    plt.xlabel('Amostra')
    plt.ylabel('Temperatura')
    plt.title(
        f'{file_name}\n'
        f'RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}'
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()

# =========================================================
# EXECUÇÃO
# =========================================================
resumo = []

for path in FILES:
    nome = os.path.basename(path)

    try:
        df = read_one_csv(path)
        y_real, y_est, gf = predict_model_3(df)

        rmse, mae, r2 = calc_metrics(y_real, y_est)

        save_name = os.path.splitext(nome)[0] + '_modelo3_real_vs_estimado.png'
        save_path = os.path.join(OUT_DIR, save_name)

        plot_real_vs_est(y_real, y_est, nome, save_path)

        resumo.append({
            'arquivo': nome,
            'n_amostras_plotadas': len(y_real),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'imagem': save_name
        })

        print(f'OK: {nome}')

    except Exception as e:
        print(f'ERRO em {nome}: {e}')

# salva resumo em csv
if len(resumo) > 0:
    resumo_df = pd.DataFrame(resumo)
    resumo_df.to_csv(
        os.path.join(OUT_DIR, 'resumo_metricas_modelo_3.csv'),
        sep=';',
        decimal=',',
        index=False,
        encoding='utf-8-sig'
    )

print('\nConcluído.')
print(f'Imagens salvas em: {OUT_DIR}')