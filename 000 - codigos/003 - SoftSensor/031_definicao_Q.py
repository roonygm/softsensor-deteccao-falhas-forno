import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\02 - Falha deriva'
OUT_DIR = os.path.join(TEST_DIR, 'otimizacao_Q_kalman_q_unico')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    f for f in glob.glob(os.path.join(TEST_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

FILES = [
    f for f in FILES
    if 1 <= int(os.path.basename(f).replace('.csv', '')) <= 3
]

# =========================================================
# PARÂMETROS DO MODELO
# =========================================================
alpha = 0.995
c0 = -5.0506262271
a_TL = 0.4455542167
b_TR = 0.5379910660
d_Gf = 0.0671649306

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
# PARÂMETROS DO FILTRO
# =========================================================
R = 9.77

# =========================================================
# PARÂMETROS DE OTIMIZAÇÃO
# =========================================================
FAULT_SAMPLES = 2000
CHI2_LIMIT_95 = 3.841
NIS_TARGET = 4.0

# busca de q único
Q_CANDIDATES = np.logspace(np.log10(0.0001), np.log10(0.5), 40)

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
        raise ValueError(f'Colunas faltando após rename: {missing}')

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# FILTRO DE KALMAN - NOVA ARQUITETURA
# =========================================================
def kalman_filter_augmented(df, Q, R, x0=None, P0=None):
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
        P0 = np.diag([10.0, 10.0, 10.0])

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
            u_prev = np.array([
                [GAS[k - 1]],
                [TL[k - 1]],
                [TR[k - 1]]
            ], dtype=float)

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
def rmse(y_true, y_hat):
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))

def evaluate_q_for_file(df, q, R, fault_samples=2000):
    Q = np.diag([q, q, q])

    out = kalman_filter_augmented(df, Q=Q, R=R, x0=None, P0=None)

    y_true = df['TC'].values
    y_pred = out['y_pred']
    y_filt = out['y_filt']
    innovation = out['innovation']
    innovation_var = out['innovation_var']

    nis = (innovation ** 2) / innovation_var
    fault_start = max(0, len(df) - fault_samples)

    metrics = {
        'rmse_pred': rmse(y_true, y_pred),
        'rmse_filt': rmse(y_true, y_filt),
        'mean_nis_total': float(np.mean(nis)),
        'mean_nis_fault': float(np.mean(nis[fault_start:])),
        'max_nis_fault': float(np.max(nis[fault_start:])),
        'pct_above_95_total': float(np.mean(nis > CHI2_LIMIT_95) * 100.0),
        'pct_above_95_fault': float(np.mean(nis[fault_start:] > CHI2_LIMIT_95) * 100.0)
    }

    return metrics, out

def evaluate_q_global(files, q_candidates, R, fault_samples=2000, nis_target=4.0):
    rows = []

    for q in q_candidates:
        rmse_pred_list = []
        rmse_filt_list = []
        nis_total_list = []
        nis_fault_list = []
        max_nis_fault_list = []
        pct95_total_list = []
        pct95_fault_list = []

        for f in files:
            df = read_one_csv(f)
            metrics, _ = evaluate_q_for_file(df, q, R, fault_samples)

            rmse_pred_list.append(metrics['rmse_pred'])
            rmse_filt_list.append(metrics['rmse_filt'])
            nis_total_list.append(metrics['mean_nis_total'])
            nis_fault_list.append(metrics['mean_nis_fault'])
            max_nis_fault_list.append(metrics['max_nis_fault'])
            pct95_total_list.append(metrics['pct_above_95_total'])
            pct95_fault_list.append(metrics['pct_above_95_fault'])

        mean_rmse_pred = float(np.mean(rmse_pred_list))
        mean_rmse_filt = float(np.mean(rmse_filt_list))
        mean_nis_total = float(np.mean(nis_total_list))
        mean_nis_fault = float(np.mean(nis_fault_list))

        rows.append({
            'q': q,
            'Q11': q,
            'Q22': q,
            'Q33': q,
            'RMSE_pred_medio': mean_rmse_pred,
            'RMSE_pred_std': float(np.std(rmse_pred_list)),
            'RMSE_filt_medio': mean_rmse_filt,
            'RMSE_filt_std': float(np.std(rmse_filt_list)),
            'NIS_total_medio': mean_nis_total,
            'NIS_fault_medio': mean_nis_fault,
            'NIS_fault_std': float(np.std(nis_fault_list)),
            'NIS_fault_max_medio': float(np.mean(max_nis_fault_list)),
            'Pct_NIS_acima_95_total_medio': float(np.mean(pct95_total_list)),
            'Pct_NIS_acima_95_fault_medio': float(np.mean(pct95_fault_list)),
            'Atende_NIS_minimo': mean_nis_fault >= nis_target,
            'Desvio_NIS_alvo': abs(mean_nis_fault - nis_target)
        })

    results = pd.DataFrame(rows)

    valid_results = results[results['Atende_NIS_minimo']].copy()

    if not valid_results.empty:
        valid_results = valid_results.sort_values(
            ['RMSE_pred_medio', 'Desvio_NIS_alvo'],
            ascending=[True, True]
        ).reset_index(drop=True)
        return results, valid_results, valid_results.iloc[0]

    fallback_results = results.sort_values(
        ['Desvio_NIS_alvo', 'RMSE_pred_medio'],
        ascending=[True, True]
    ).reset_index(drop=True)

    return results, pd.DataFrame(), fallback_results.iloc[0]

# =========================================================
# PLOTS DA OTIMIZAÇÃO
# =========================================================
def plot_optimization_results(results_df, out_dir, best_row):
    # RMSE predito
    plt.figure(figsize=(12, 5))
    plt.semilogx(results_df['q'], results_df['RMSE_pred_medio'], marker='o', label='RMSE predito médio')
    plt.axvline(best_row['q'], linestyle='--', color='red', label=f"Melhor q = {best_row['q']:.6f}")
    plt.xlabel('q')
    plt.ylabel('RMSE predito médio')
    plt.title('RMSE predito médio em função de q')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'otimizacao_q_rmse_pred.png'), dpi=150)
    plt.close()

    # RMSE filtrado
    plt.figure(figsize=(12, 5))
    plt.semilogx(results_df['q'], results_df['RMSE_filt_medio'], marker='s', label='RMSE filtrado médio')
    plt.axvline(best_row['q'], linestyle='--', color='red', label=f"Melhor q = {best_row['q']:.6f}")
    plt.xlabel('q')
    plt.ylabel('RMSE filtrado médio')
    plt.title('RMSE filtrado médio em função de q')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'otimizacao_q_rmse_filt.png'), dpi=150)
    plt.close()

    # NIS falha
    plt.figure(figsize=(12, 5))
    plt.semilogx(results_df['q'], results_df['NIS_fault_medio'], marker='^', label='NIS médio na falha')
    plt.axhline(NIS_TARGET, linestyle='--', color='black', label=f'Meta NIS = {NIS_TARGET}')
    plt.axvline(best_row['q'], linestyle='--', color='red', label=f"Melhor q = {best_row['q']:.6f}")
    plt.xlabel('q')
    plt.ylabel('NIS médio na falha')
    plt.title('NIS médio na região de falha em função de q')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'otimizacao_q_nis_fault.png'), dpi=150)
    plt.close()

    # porcentagem acima do limite
    plt.figure(figsize=(12, 5))
    plt.semilogx(results_df['q'], results_df['Pct_NIS_acima_95_fault_medio'], marker='d',
                 label='% NIS > limite 95% na falha')
    plt.axvline(best_row['q'], linestyle='--', color='red', label=f"Melhor q = {best_row['q']:.6f}")
    plt.xlabel('q')
    plt.ylabel('% acima do limite')
    plt.title('Percentual de NIS acima do limite de 95% na falha')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'otimizacao_q_pct95_fault.png'), dpi=150)
    plt.close()

# =========================================================
# PLOT DETALHADO DO MELHOR q
# =========================================================
def plot_best_case(df, out, file_name, save_path):
    t = np.arange(len(df))
    innovation = out['innovation']
    innovation_var = out['innovation_var']
    nis = (innovation ** 2) / innovation_var

    falha_inicio = max(0, len(df) - FAULT_SAMPLES)
    falha_fim = len(df) - 1

    fig, axes = plt.subplots(
        3, 1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1.5, 1.5]}
    )

    axes[0].axvspan(falha_inicio, falha_fim, color='red', alpha=0.10, label='Região de falha')
    axes[0].plot(t, df['TC'].values, label='TC real')
    axes[0].plot(t, out['y_filt'], label='TC estimado (Kalman)')
    axes[0].set_ylabel('Temperatura')
    axes[0].set_title(f'Arquivo {file_name} - melhor q')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvspan(falha_inicio, falha_fim, color='red', alpha=0.10)
    axes[1].plot(t, innovation, label='Inovação')
    axes[1].axhline(0.0, linestyle='--', color='black')
    axes[1].set_ylabel('Inovação')
    axes[1].set_title('Inovação')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].axvspan(falha_inicio, falha_fim, color='red', alpha=0.10)
    axes[2].plot(t, nis, label='NIS')
    axes[2].axhline(CHI2_LIMIT_95, linestyle='--', color='red',
                    label=fr'Limite $\chi^2$ 95% = {CHI2_LIMIT_95:.3f}')
    axes[2].set_xlabel('Amostra')
    axes[2].set_ylabel('NIS')
    axes[2].set_title('Estatística NIS')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
def main():
    print('=' * 80)
    print('Otimizando q único para Q = diag(q, q, q)...')
    print('=' * 80)

    if not FILES:
        print('Nenhum arquivo CSV encontrado.')
        return

    all_results_df, valid_results_df, best_row = evaluate_q_global(
        files=FILES,
        q_candidates=Q_CANDIDATES,
        R=R,
        fault_samples=FAULT_SAMPLES,
        nis_target=NIS_TARGET
    )

    best_q = float(best_row['q'])
    best_Q = np.diag([best_q, best_q, best_q])

    all_results_csv = os.path.join(OUT_DIR, 'ranking_otimizacao_q.csv')
    all_results_df.to_csv(all_results_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    if not valid_results_df.empty:
        valid_results_csv = os.path.join(OUT_DIR, 'ranking_otimizacao_q_validos.csv')
        valid_results_df.to_csv(valid_results_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')
        print('\nMelhor q encontrado entre os que atendem NIS >= alvo:')
    else:
        print('\nNenhum q atingiu o NIS alvo.')
        print('Foi selecionado o q mais próximo da meta com menor RMSE associado.')

    print(f'\nMelhor q = {best_q:.10f}')
    print('Melhor Q =')
    print(best_Q)
    print(f'RMSE_pred_medio = {best_row["RMSE_pred_medio"]:.6f}')
    print(f'RMSE_filt_medio = {best_row["RMSE_filt_medio"]:.6f}')
    print(f'NIS_fault_medio = {best_row["NIS_fault_medio"]:.6f}')
    print(f'Pct_NIS_acima_95_fault_medio = {best_row["Pct_NIS_acima_95_fault_medio"]:.2f}%')

    plot_optimization_results(all_results_df, OUT_DIR, best_row)

    # gera gráficos detalhados com o melhor q
    for i, f in enumerate(FILES, start=1):
        file_name = os.path.basename(f)
        print(f'Gerando gráfico detalhado [{i:02d}/{len(FILES):02d}] {file_name}')

        df = read_one_csv(f)
        out = kalman_filter_augmented(df, Q=best_Q, R=R, x0=None, P0=None)

        save_path = os.path.join(
            OUT_DIR,
            f'melhor_q_kalman_tc_inovacao_nis_{file_name.replace(".csv", ".png")}'
        )

        plot_best_case(df, out, file_name, save_path)

    print('\nArquivos salvos em:')
    print(OUT_DIR)
    print(all_results_csv)

    print('\nResumo do melhor resultado:')
    print(best_row.to_string())

if __name__ == '__main__':
    main()
