import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CONFIGURAÇÕES
# =========================================================
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\00 - Dados originais'
OUT_DIR = os.path.join(TEST_DIR, 'resultados_kalman')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    f for f in glob.glob(os.path.join(TEST_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

# Se quiser limitar de 01 a 50 explicitamente
FILES = [
    f for f in FILES
    if 1 <= int(os.path.basename(f).replace('.csv', '')) <= 50
]

# =========================================================
# PARÂMETROS DO MODELO IDENTIFICADO
# =========================================================
alpha = 0.995
c0 = -5.0506262271
a_TL = 0.4455542167
b_TR = 0.5379910660
d_Gf = 0.0671649306

A = alpha
B = 1.0 - alpha
C = d_Gf

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
# MÉTRICAS
# =========================================================
def calc_metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# =========================================================
# SOFTSENSOR EM MALHA ABERTA
# =========================================================
def first_order_filter(u, alpha):
    u = np.asarray(u, dtype=float)
    gf = np.zeros_like(u)
    gf[0] = u[0]
    for k in range(1, len(u)):
        gf[k] = alpha * gf[k - 1] + (1 - alpha) * u[k]
    return gf

def softsensor_open_loop(df):
    TL = df['TL'].values
    TR = df['TR'].values
    GAS = df['GAS'].values

    Gf = first_order_filter(GAS, alpha)
    yhat = c0 + a_TL * TL + b_TR * TR + d_Gf * Gf

    return yhat, Gf

# =========================================================
# FILTRO DE KALMAN ESCALAR
# =========================================================
def kalman_filter_scalar(df, Q, R, x0=None, P0=1.0):
    """
    Modelo:
        x(k) = A*x(k-1) + B*GAS(k) + w(k)
        y(k) = C*x(k) + a*TL(k) + b*TR(k) + c0 + v(k)

    onde:
        w ~ N(0,Q)
        v ~ N(0,R)
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

    y_pred = np.zeros(n)     # saída prevista antes da atualização
    y_filt = np.zeros(n)     # saída filtrada após atualização
    innov = np.zeros(n)      # inovação
    S_arr = np.zeros(n)      # variância da inovação
    K_arr = np.zeros(n)      # ganho de Kalman

    if x0 is None:
        x0 = GAS[0]

    x_prev = x0
    P_prev = P0

    for k in range(n):
        # -------------------------
        # PREDIÇÃO
        # -------------------------
        x_k_pred = A * x_prev + B * GAS[k]
        P_k_pred = A * P_prev * A + Q

        y_k_pred = C * x_k_pred + a_TL * TL[k] + b_TR * TR[k] + c0

        # -------------------------
        # ATUALIZAÇÃO
        # -------------------------
        innov_k = y_meas[k] - y_k_pred
        S_k = C * P_k_pred * C + R
        K_k = (P_k_pred * C) / S_k

        x_k_filt = x_k_pred + K_k * innov_k
        P_k_filt = (1 - K_k * C) * P_k_pred

        y_k_filt = C * x_k_filt + a_TL * TL[k] + b_TR * TR[k] + c0

        # armazenar
        x_pred[k] = x_k_pred
        x_filt[k] = x_k_filt
        P_pred[k] = P_k_pred
        P_filt[k] = P_k_filt

        y_pred[k] = y_k_pred
        y_filt[k] = y_k_filt
        innov[k] = innov_k
        S_arr[k] = S_k
        K_arr[k] = K_k

        x_prev = x_k_filt
        P_prev = P_k_filt

    return {
        'x_pred': x_pred,
        'x_filt': x_filt,
        'P_pred': P_pred,
        'P_filt': P_filt,
        'y_pred': y_pred,
        'y_filt': y_filt,
        'innovation': innov,
        'innovation_var': S_arr,
        'kalman_gain': K_arr
    }

# =========================================================
# AJUSTE INICIAL DE Q E R
# =========================================================
def estimate_base_R(files):
    """
    Usa o erro do softsensor em malha aberta para estimar uma escala inicial de R.
    """
    vars_res = []

    for f in files:
        df = read_one_csv(f)
        y_true = df['TC'].values
        y_open, _ = softsensor_open_loop(df)
        res = y_true - y_open
        vars_res.append(np.var(res))

    return float(np.median(vars_res))

def tune_QR(files, base_R):
    """
    Busca simples para Q e R.
    Observação:
    - Para avaliação inicial, usa os próprios arquivos de teste.
    - Na dissertação, o ideal é separar calibração de Q/R de avaliação final.
    """
    q_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    r_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]

    best = None
    best_rmse = np.inf

    # usar um subconjunto para acelerar
    tune_files = files[:min(10, len(files))]

    print('\nBuscando Q e R para o filtro de Kalman...')

    for q_scale in q_candidates:
        for r_scale in r_candidates:
            R = base_R * r_scale
            Q = (base_R / max(C**2, 1e-9)) * q_scale

            rmses = []

            for f in tune_files:
                df = read_one_csv(f)
                y_true = df['TC'].values
                out_kf = kalman_filter_scalar(df, Q=Q, R=R, x0=None, P0=10.0)
                y_kf = out_kf['y_filt']
                met = calc_metrics(y_true, y_kf)
                rmses.append(met['RMSE'])

            avg_rmse = float(np.mean(rmses))
            print(f'Q={Q:.6f} | R={R:.6f} | RMSE médio={avg_rmse:.4f}')

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best = {'Q': Q, 'R': R, 'avg_rmse': avg_rmse}

    return best

# =========================================================
# PLOTS
# =========================================================
def plot_one_file(df, y_open, y_kf, innovation, file_name, save_dir):
    t = np.arange(len(df))

    plt.figure(figsize=(14, 6))
    plt.plot(t, df['TC'].values, label='TC real')
    plt.plot(t, y_open, label='Softsensor malha aberta')
    plt.plot(t, y_kf, label='Kalman')
    plt.xlabel('Amostra (s)')
    plt.ylabel('Temperatura')
    plt.title(f'Comparação das estimativas - {file_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'comparacao_{file_name}.png'), dpi=150)
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(t, innovation)
    plt.xlabel('Amostra (s)')
    plt.ylabel('Inovação')
    plt.title(f'Inovação do filtro de Kalman - {file_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'inovacao_{file_name}.png'), dpi=150)
    plt.show()

def plot_summary_bar(df_metrics, save_path):
    labels = df_metrics['Arquivo'].tolist()
    rmse_open = df_metrics['RMSE_OpenLoop'].tolist()
    rmse_kf = df_metrics['RMSE_Kalman'].tolist()

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(16, 6))
    plt.bar(x - width/2, rmse_open, width, label='Softsensor')
    plt.bar(x + width/2, rmse_kf, width, label='Kalman')
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('RMSE')
    plt.title('Comparação de RMSE por arquivo')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# =========================================================
# MAIN
# =========================================================
def main():
    print('=' * 80)
    print('ARQUIVOS DE TESTE')
    print('=' * 80)
    for f in FILES:
        print(os.path.basename(f))

    # -------------------------
    # estima escala de R
    # -------------------------
    base_R = estimate_base_R(FILES)
    print('\nVariância base estimada para o erro de medição/modelo:')
    print(f'base_R = {base_R:.6f}')

    # -------------------------
    # ajuste de Q e R
    # -------------------------
    best_qr = tune_QR(FILES, base_R)
    Q = best_qr['Q']
    R = best_qr['R']

    print('\nMelhores parâmetros encontrados:')
    print(best_qr)

    # -------------------------
    # avaliação final em todos os arquivos
    # -------------------------
    rows = []
    all_y_true = []
    all_y_open = []
    all_y_kf = []
    all_innov = []

    plot_dir = os.path.join(OUT_DIR, 'graficos')
    os.makedirs(plot_dir, exist_ok=True)

    for idx, f in enumerate(FILES, start=1):
        df = read_one_csv(f)
        y_true = df['TC'].values

        y_open, gf = softsensor_open_loop(df)

        out_kf = kalman_filter_scalar(df, Q=Q, R=R, x0=None, P0=10.0)
        y_kf = out_kf['y_filt']
        innovation = out_kf['innovation']

        met_open = calc_metrics(y_true, y_open)
        met_kf = calc_metrics(y_true, y_kf)

        rows.append({
            'Arquivo': os.path.basename(f),
            'RMSE_OpenLoop': met_open['RMSE'],
            'MAE_OpenLoop': met_open['MAE'],
            'R2_OpenLoop': met_open['R2'],
            'RMSE_Kalman': met_kf['RMSE'],
            'MAE_Kalman': met_kf['MAE'],
            'R2_Kalman': met_kf['R2'],
            'Melhora_RMSE_%': 100.0 * (met_open['RMSE'] - met_kf['RMSE']) / met_open['RMSE']
        })

        all_y_true.append(y_true)
        all_y_open.append(y_open)
        all_y_kf.append(y_kf)
        all_innov.append(innovation)

        # plota alguns arquivos
        if idx <= 5:
            file_name = os.path.basename(f).replace('.csv', '')
            plot_one_file(df, y_open, y_kf, innovation, file_name, plot_dir)

    df_metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, 'metricas_kalman_por_arquivo.csv')
    df_metrics.to_csv(metrics_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    # métricas globais
    y_true_all = np.concatenate(all_y_true)
    y_open_all = np.concatenate(all_y_open)
    y_kf_all = np.concatenate(all_y_kf)
    innov_all = np.concatenate(all_innov)

    met_open_all = calc_metrics(y_true_all, y_open_all)
    met_kf_all = calc_metrics(y_true_all, y_kf_all)

    print('\n' + '=' * 80)
    print('RESULTADOS GLOBAIS')
    print('=' * 80)
    print('Softsensor em malha aberta:')
    print(met_open_all)
    print('\nFiltro de Kalman:')
    print(met_kf_all)

    print('\nMédia da inovação:', float(np.mean(innov_all)))
    print('Desvio padrão da inovação:', float(np.std(innov_all)))

    summary_path = os.path.join(OUT_DIR, 'comparacao_rmse_por_arquivo.png')
    plot_summary_bar(df_metrics, summary_path)

    # salvar resumo global
    resumo = pd.DataFrame({
        'Modelo': ['Softsensor_OpenLoop', 'Kalman'],
        'RMSE': [met_open_all['RMSE'], met_kf_all['RMSE']],
        'MAE': [met_open_all['MAE'], met_kf_all['MAE']],
        'R2': [met_open_all['R2'], met_kf_all['R2']]
    })
    resumo_path = os.path.join(OUT_DIR, 'resumo_global_kalman.csv')
    resumo.to_csv(resumo_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    print('\nArquivos salvos em:')
    print(OUT_DIR)
    print(metrics_path)
    print(summary_path)
    print(resumo_path)

if __name__ == '__main__':
    main()
