import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\03 - Falha tipo congelamento'
OUT_DIR = os.path.join(TEST_DIR, 'graficos_kalman_inovacao')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    f for f in glob.glob(os.path.join(TEST_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

FILES = [
    f for f in FILES
    if 1 <= int(os.path.basename(f).replace('.csv', '')) <= 50
]

# =========================================================
# PARÂMETROS DO MODELO
# =========================================================
alpha = 0.995
c0 = -5.0506262271
a_TL = 0.4455542167
b_TR = 0.5379910660
d_Gf = 0.0671649306

# espaço de estados
A = alpha
B = 1.0 - alpha
C = d_Gf

# parâmetros do Kalman já definidos
#Q = 5.027024170347726
Q = 0.10
#R = 0.5669387250352926
R = 10
# =========================================================
# LEITURA DOS DADOS
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
        print('\nArquivo com problema:', path)
        print('Colunas encontradas:', df.columns.tolist())
        raise ValueError(f'Colunas faltando após rename: {missing}')

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# FILTRO DE KALMAN
# =========================================================
def kalman_filter_scalar(df, Q, R, x0=None, P0=10.0):
    """
    Modelo:
        x(k) = A*x(k-1) + B*GAS(k) + w(k)
        y(k) = C*x(k) + a*TL(k) + b*TR(k) + c0 + v(k)
    """
    TL = df['TL'].values
    TR = df['TR'].values
    GAS = df['GAS'].values
    y_meas = df['TC'].values

    n = len(df)

    x_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_pred = np.zeros(n)
    P_filt = np.zeros(n)

    y_pred = np.zeros(n)
    y_filt = np.zeros(n)

    innovation = np.zeros(n)
    innovation_var = np.zeros(n)
    kalman_gain = np.zeros(n)

    if x0 is None:
        x0 = GAS[0]

    x_prev = x0
    P_prev = P0

    for k in range(n):
        # predição
        x_k_pred = A * x_prev + B * GAS[k]
        P_k_pred = A * P_prev * A + Q

        y_k_pred = C * x_k_pred + a_TL * TL[k] + b_TR * TR[k] + c0

        # atualização
        innov_k = y_meas[k] - y_k_pred
        S_k = C * P_k_pred * C + R
        K_k = (P_k_pred * C) / S_k

        x_k_filt = x_k_pred + K_k * innov_k
        P_k_filt = (1.0 - K_k * C) * P_k_pred

        y_k_filt = C * x_k_filt + a_TL * TL[k] + b_TR * TR[k] + c0

        # armazenar
        x_pred[k] = x_k_pred
        x_filt[k] = x_k_filt
        P_pred[k] = P_k_pred
        P_filt[k] = P_k_filt

        y_pred[k] = y_k_pred
        y_filt[k] = y_k_filt

        innovation[k] = innov_k
        innovation_var[k] = S_k
        kalman_gain[k] = K_k

        x_prev = x_k_filt
        P_prev = P_k_filt

    return {
        'x_pred': x_pred,
        'x_filt': x_filt,
        'P_pred': P_pred,
        'P_filt': P_filt,
        'y_pred': y_pred,
        'y_filt': y_filt,
        'innovation': innovation,
        'innovation_var': innovation_var,
        'kalman_gain': kalman_gain
    }

# =========================================================
# MÉTRICAS
# =========================================================
def calc_metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2

# =========================================================
# PLOT POR ARQUIVO
# =========================================================
def plot_tc_and_innovation(df, y_est, innovation, file_name, save_path):
    t = np.arange(len(df))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5]}
    )

    # gráfico superior
    axes[0].plot(t, df['TC'].values, label='TC real')
    axes[0].plot(t, y_est, label='TC estimado (Kalman)')
    axes[0].set_ylabel('Temperatura')
    axes[0].set_title(f'Arquivo {file_name} - TC real vs estimado')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # gráfico inferior
    axes[1].plot(t, innovation, label='Inovação')
    axes[1].axhline(0.0, linestyle='--')
    axes[1].set_xlabel('Amostra (s)')
    axes[1].set_ylabel('Inovação')
    axes[1].set_title('Inovação do filtro de Kalman')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
def main():
    print('=' * 80)
    print('Processando arquivos...')
    print('=' * 80)

    results_rows = []

    for i, f in enumerate(FILES, start=1):
        file_name = os.path.basename(f)
        print(f'[{i:02d}/{len(FILES):02d}] {file_name}')

        df = read_one_csv(f)

        out = kalman_filter_scalar(df, Q=Q, R=R, x0=None, P0=10.0)

        y_true = df['TC'].values
        y_est = out['y_filt']
        innovation = out['innovation']

        rmse, mae, r2 = calc_metrics(y_true, y_est)

        results_rows.append({
            'Arquivo': file_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Media_Inovacao': float(np.mean(innovation)),
            'Std_Inovacao': float(np.std(innovation))
        })

        save_path = os.path.join(
            OUT_DIR,
            f'kalman_tc_inovacao_{file_name.replace(".csv", ".png")}'
        )

        plot_tc_and_innovation(
            df=df,
            y_est=y_est,
            innovation=innovation,
            file_name=file_name,
            save_path=save_path
        )

    # salvar métricas
    results_df = pd.DataFrame(results_rows)
    results_csv = os.path.join(OUT_DIR, 'metricas_kalman_50_arquivos.csv')
    results_df.to_csv(results_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 80)
    print('Concluído.')
    print(f'Gráficos salvos em: {OUT_DIR}')
    print(f'Métricas salvas em: {results_csv}')
    print('=' * 80)

if __name__ == '__main__':
    main()
