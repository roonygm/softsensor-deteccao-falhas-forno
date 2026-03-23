import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo'
OUT_DIR = os.path.join(TEST_DIR, 'graficos_modelo_kalman_inovacao')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    f for f in glob.glob(os.path.join(TEST_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

# Se quiser limitar os arquivos, descomente:
# FILES = [
#     f for f in FILES
#     if 1 <= int(os.path.basename(f).replace('.csv', '')) <= 50
# ]

# =========================================================
# PARÂMETROS DO MODELO
# =========================================================
alpha = 0.995
c0 = -5.0506262271
a_TL = 0.4455542167
b_TR = 0.5379910660
d_Gf = 0.0671649306

# =========================================================
# MODELO EM ESPAÇO DE ESTADOS AUMENTADO
# =========================================================
A = np.array([
    [alpha, 0.0,   0.0],
    [0.0,   0.0,   0.0],
    [0.0,   0.0,   0.0]
], dtype=float)

B = np.array([
    [1.0 - alpha, 0.0, 0.0],
    [0.0,         1.0, 0.0],
    [0.0,         0.0, 1.0]
], dtype=float)

C = np.array([[d_Gf, a_TL, b_TR]], dtype=float)

# =========================================================
# PARÂMETROS DO KALMAN
# =========================================================
Q = np.diag([0.07, 0.07, 0.07])
R = 9.77

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
# PREDIÇÃO PURA DO MODELO (SEM KALMAN)
# =========================================================
def model_prediction_open_loop(df):
    """
    Gera a predição do modelo usando apenas as entradas medidas,
    sem correção pela medição de TC.
    """
    TL = df['TL'].values
    TR = df['TR'].values
    GAS = df['GAS'].values

    n = len(df)

    Gf_model = np.zeros(n)
    y_model = np.zeros(n)

    # condição inicial
    Gf_model[0] = GAS[0]

    # saída inicial
    y_model[0] = c0 + a_TL * TL[0] + b_TR * TR[0] + d_Gf * Gf_model[0]

    for k in range(1, n):
        Gf_model[k] = alpha * Gf_model[k - 1] + (1.0 - alpha) * GAS[k - 1]
        y_model[k] = c0 + a_TL * TL[k] + b_TR * TR[k] + d_Gf * Gf_model[k]

    return y_model, Gf_model

# =========================================================
# FILTRO DE KALMAN VETORIAL - ESTADO AUMENTADO
# =========================================================
def kalman_filter_augmented(df, Q, R, x0=None, P0=None):
    """
    Modelo:
        x(k) = A*x(k-1) + B*u(k-1) + w(k)
        y(k) = C*x(k) + c0 + v(k)

    onde:
        x = [Gf, TL_est, TR_est]^T
        u = [GAS, TL, TR]^T
    """

    TL = df['TL'].values
    TR = df['TR'].values
    GAS = df['GAS'].values
    y_meas = df['TC'].values

    n = len(df)
    nx = 3

    x_pred = np.zeros((n, nx))
    x_filt = np.zeros((n, nx))

    P_pred = np.zeros((n, nx, nx))
    P_filt = np.zeros((n, nx, nx))

    y_pred = np.zeros(n)
    y_filt = np.zeros(n)

    innovation = np.zeros(n)
    innovation_var = np.zeros(n)
    kalman_gain = np.zeros((n, nx))

    if x0 is None:
        x0 = np.array([GAS[0], TL[0], TR[0]], dtype=float)

    if P0 is None:
        P0 = np.diag([10.0, 1.0, 1.0])

    x_prev = x0.reshape(nx, 1)
    P_prev = P0.copy()

    for k in range(n):
        # -------------------------------------------------
        # PREDIÇÃO
        # -------------------------------------------------
        if k == 0:
            x_k_pred = x_prev
            P_k_pred = P_prev
        else:
            u_prev = np.array([[GAS[k - 1]],
                               [TL[k - 1]],
                               [TR[k - 1]]], dtype=float)

            x_k_pred = A @ x_prev + B @ u_prev
            P_k_pred = A @ P_prev @ A.T + Q

        y_k_pred = (C @ x_k_pred).item() + c0

        # -------------------------------------------------
        # ATUALIZAÇÃO
        # -------------------------------------------------
        innov_k = y_meas[k] - y_k_pred
        S_k = (C @ P_k_pred @ C.T).item() + R
        K_k = (P_k_pred @ C.T) / S_k

        x_k_filt = x_k_pred + K_k * innov_k
        P_k_filt = (np.eye(nx) - K_k @ C) @ P_k_pred

        y_k_filt = (C @ x_k_filt).item() + c0

        # -------------------------------------------------
        # ARMAZENAR
        # -------------------------------------------------
        x_pred[k, :] = x_k_pred.flatten()
        x_filt[k, :] = x_k_filt.flatten()

        P_pred[k, :, :] = P_k_pred
        P_filt[k, :, :] = P_k_filt

        y_pred[k] = y_k_pred
        y_filt[k] = y_k_filt

        innovation[k] = innov_k
        innovation_var[k] = S_k
        kalman_gain[k, :] = K_k.flatten()

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
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2

# =========================================================
# PLOT POR ARQUIVO
# =========================================================
def plot_tc_model_kalman_innovation(df, y_model, y_kalman, innovation, file_name, save_path):
    t = np.arange(len(df))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.3]}
    )

    # -------------------------------------------------
    # subplot 1: TC real vs modelo vs Kalman
    # -------------------------------------------------
    axes[0].plot(t, df['TC'].values, label='TC real', linewidth=1.5)
    axes[0].plot(t, y_model, label='Predição do modelo', linewidth=1.2)
    axes[0].plot(t, y_kalman, label='Predição do filtro de Kalman', linewidth=1.2)

    axes[0].set_ylabel('Temperatura')
    axes[0].set_title(f'Arquivo {file_name} - TC real, modelo e Kalman')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -------------------------------------------------
    # subplot 2: inovação
    # -------------------------------------------------
    axes[1].plot(t, innovation, label='Inovação do filtro de Kalman', linewidth=1.0)
    axes[1].axhline(0.0, linestyle='--', color='black', linewidth=1.0)
    axes[1].set_xlabel('Amostra')
    axes[1].set_ylabel('Inovação')
    axes[1].set_title('Inovação')
    axes[1].legend()
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

    print('\nMatrizes do modelo aumentado:')
    print('A =\n', A)
    print('B =\n', B)
    print('C =\n', C)
    print('c0 =', c0)
    print('Q =\n', Q)
    print('R =', R)

    results_rows = []

    for i, f in enumerate(FILES, start=1):
        file_name = os.path.basename(f)
        print(f'[{i:02d}/{len(FILES):02d}] {file_name}')

        df = read_one_csv(f)

        # Predição do modelo sem Kalman
        y_model, _ = model_prediction_open_loop(df)

        # Kalman
        out = kalman_filter_augmented(df, Q=Q, R=R, x0=None, P0=None)

        y_true = df['TC'].values
        y_kalman = out['y_filt']
        innovation = out['innovation']

        rmse_model, mae_model, r2_model = calc_metrics(y_true, y_model)
        rmse_kf, mae_kf, r2_kf = calc_metrics(y_true, y_kalman)

        results_rows.append({
            'Arquivo': file_name,
            'RMSE_Modelo': rmse_model,
            'MAE_Modelo': mae_model,
            'R2_Modelo': r2_model,
            'RMSE_Kalman': rmse_kf,
            'MAE_Kalman': mae_kf,
            'R2_Kalman': r2_kf,
            'Media_Inovacao': float(np.mean(innovation)),
            'Std_Inovacao': float(np.std(innovation))
        })

        save_path = os.path.join(
            OUT_DIR,
            f'modelo_kalman_inovacao_{file_name.replace(".csv", ".png")}'
        )

        plot_tc_model_kalman_innovation(
            df=df,
            y_model=y_model,
            y_kalman=y_kalman,
            innovation=innovation,
            file_name=file_name,
            save_path=save_path
        )

    results_df = pd.DataFrame(results_rows)
    results_csv = os.path.join(OUT_DIR, 'metricas_modelo_kalman.csv')
    results_df.to_csv(results_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 80)
    print('Concluído.')
    print(f'Gráficos salvos em: {OUT_DIR}')
    print(f'Métricas salvas em: {results_csv}')
    print('=' * 80)

if __name__ == '__main__':
    main()
