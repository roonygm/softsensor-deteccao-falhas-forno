import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\03 - Falha tipo congelamento'

OUT_DIR = os.path.join(TEST_DIR, 'graficos_std_exp_tc_duplo_limiar')
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
# PARÂMETROS DE DETECÇÃO
# =========================================================
FAULT_SAMPLES = 3600
IGNORE_FIRST_SAMPLES = 300
PERSISTENCE_SAMPLES = 100

# =========================================================
# PARÂMETROS DO DESVIO PADRÃO MÓVEL EXPONENCIAL
# =========================================================
EWMA_ALPHA = 0.01

# Novos limiares
STD_THRESHOLD_LOW = 0.1139
STD_THRESHOLD_HIGH = 3.5703

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
# MÉTRICAS AUXILIARES
# =========================================================
def calc_regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2


# =========================================================
# MÉDIA E DESVIO PADRÃO MÓVEL EXPONENCIAL
# =========================================================
def compute_ewma_std(signal, alpha=EWMA_ALPHA):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    ewma_mean = np.zeros(n, dtype=float)
    residual = np.zeros(n, dtype=float)
    ewma_var = np.zeros(n, dtype=float)
    ewma_std = np.zeros(n, dtype=float)

    ewma_mean[0] = signal[0]
    residual[0] = 0.0
    ewma_var[0] = 0.0
    ewma_std[0] = 0.0

    for k in range(1, n):
        ewma_mean[k] = alpha * signal[k] + (1 - alpha) * ewma_mean[k - 1]
        residual[k] = signal[k] - ewma_mean[k]
        ewma_var[k] = alpha * (residual[k] ** 2) + (1 - alpha) * ewma_var[k - 1]
        ewma_std[k] = np.sqrt(max(ewma_var[k], 0.0))

    return {
        'ewma_mean': ewma_mean,
        'residual': residual,
        'ewma_var': ewma_var,
        'ewma_std': ewma_std
    }


# =========================================================
# DETECÇÃO SEM PERSISTÊNCIA
# =========================================================
def detect_fault_points(
    statistic,
    threshold_high=STD_THRESHOLD_HIGH,
    threshold_low=STD_THRESHOLD_LOW,
    ignore_first=IGNORE_FIRST_SAMPLES
):
    """
    Marca falha ponto a ponto:
    - nas primeiras `ignore_first` amostras não detecta
    - depois disso, marca falha se estiver fora da faixa
    """
    n = len(statistic)
    fault_mask = np.zeros(n, dtype=bool)

    for k in range(ignore_first, n):
        out_of_band = (statistic[k] > threshold_high) or (statistic[k] < threshold_low)
        if out_of_band:
            fault_mask[k] = True

    return fault_mask


# =========================================================
# DETECÇÃO COM PERSISTÊNCIA
# =========================================================
def detect_fault_points_with_persistence(
    statistic,
    threshold_high=STD_THRESHOLD_HIGH,
    threshold_low=STD_THRESHOLD_LOW,
    ignore_first=IGNORE_FIRST_SAMPLES,
    persistence=PERSISTENCE_SAMPLES
):
    """
    Marca falha apenas em trechos com pelo menos `persistence`
    amostras consecutivas fora da faixa normal.
    """
    n = len(statistic)
    fault_mask = np.zeros(n, dtype=bool)

    count = 0
    start_idx = None

    for k in range(n):
        if k < ignore_first:
            count = 0
            start_idx = None
            continue

        out_of_band = (statistic[k] > threshold_high) or (statistic[k] < threshold_low)

        if out_of_band:
            if count == 0:
                start_idx = k
            count += 1
        else:
            if count >= persistence and start_idx is not None:
                fault_mask[start_idx:k] = True
            count = 0
            start_idx = None

    if count >= persistence and start_idx is not None:
        fault_mask[start_idx:n] = True

    return fault_mask


# =========================================================
# TEMPO DE DETECÇÃO
# =========================================================
def compute_detection_delay_samples(fault_mask_detected, fault_start_idx):
    idx_detected = np.where(fault_mask_detected[fault_start_idx:])[0]

    if len(idx_detected) == 0:
        return np.nan

    return int(idx_detected[0])


# =========================================================
# AVALIAÇÃO AMOSTRA POR AMOSTRA
# =========================================================
def evaluate_sample_by_sample(fault_mask_detected, n_samples, fault_samples=FAULT_SAMPLES):
    fault_start_idx = max(0, n_samples - fault_samples)

    y_true_fault = np.zeros(n_samples, dtype=bool)
    y_true_fault[fault_start_idx:] = True

    y_pred_fault = fault_mask_detected.astype(bool)

    tp = int(np.sum((y_true_fault == True)  & (y_pred_fault == True)))
    tn = int(np.sum((y_true_fault == False) & (y_pred_fault == False)))
    fp = int(np.sum((y_true_fault == False) & (y_pred_fault == True)))
    fn = int(np.sum((y_true_fault == True)  & (y_pred_fault == False)))

    return {
        'fault_start_idx': fault_start_idx,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


# =========================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# =========================================================
def compute_classification_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    acuracia = (tp + tn) / total if total > 0 else np.nan
    sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precisao = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    f1_score = (
        2 * precisao * sensibilidade / (precisao + sensibilidade)
        if (precisao + sensibilidade) > 0 else np.nan
    )

    return {
        'Acuracia': acuracia,
        'Sensibilidade': sensibilidade,
        'Especificidade': especificidade,
        'Precisao': precisao,
        'F1_score': f1_score
    }


# =========================================================
# FIGURA DA MATRIZ DE CONFUSÃO
# =========================================================
def save_confusion_matrix_figure(tp, tn, fp, fn, save_path, title='Matriz de Confusão'):
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_title(title)
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Falha'])
    ax.set_yticklabels(['Normal', 'Falha'])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f'{cm[i, j]}',
                ha='center', va='center',
                color='black', fontsize=14, fontweight='bold'
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================
# FIGURA COM MATRIZ + MÉTRICAS
# =========================================================
def save_confusion_matrix_with_metrics_figure(
    tp, tn, fp, fn, metrics, mean_delay, std_delay, save_path, title
):
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_title(title)
    ax1.set_xlabel('Predito')
    ax1.set_ylabel('Real')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Falha'])
    ax1.set_yticklabels(['Normal', 'Falha'])

    for i in range(2):
        for j in range(2):
            ax1.text(
                j, i, f'{cm[i, j]}',
                ha='center', va='center',
                color='black', fontsize=14, fontweight='bold'
            )

    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    delay_text = (
        f'Tempo médio de detecção: {mean_delay:.2f} amostras\n'
        f'Desvio padrão: {std_delay:.2f} amostras'
        if not np.isnan(mean_delay)
        else 'Tempo médio de detecção: não foi possível calcular'
    )

    text = (
        f'TP = {tp}\n'
        f'TN = {tn}\n'
        f'FP = {fp}\n'
        f'FN = {fn}\n\n'
        f"Acurácia: {metrics['Acuracia']:.6f}\n"
        f"Sensibilidade: {metrics['Sensibilidade']:.6f}\n"
        f"Especificidade: {metrics['Especificidade']:.6f}\n"
        f"Precisão: {metrics['Precisao']:.6f}\n"
        f"F1-score: {metrics['F1_score']:.6f}\n\n"
        f'{delay_text}'
    )

    ax2.text(
        0.0, 1.0, text,
        va='top', ha='left',
        fontsize=11,
        family='monospace'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================
# PLOT POR ARQUIVO
# =========================================================
def plot_tc_and_std(df, ewma_mean, ewma_std, fault_mask_detected, file_name, save_path):
    """
    Gráfico mantido no mesmo estilo:
    - sem linha vertical das 300 amostras
    - pontos vermelhos usando detecção sem persistência
    """
    t = np.arange(len(df))
    falha_inicio = max(0, len(df) - FAULT_SAMPLES)
    falha_fim = len(df) - 1

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2]}
    )

    axes[0].axvspan(
        falha_inicio, falha_fim,
        color='red', alpha=0.10,
        label='Região real de falha'
    )

    axes[0].plot(t, df['TC'].values, label='TC real')
    axes[0].plot(t, ewma_mean, label='Média móvel exponencial do TC')

    axes[0].set_ylabel('Temperatura')
    axes[0].set_title(f'Arquivo {file_name} - TC e média móvel exponencial')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvspan(
        falha_inicio, falha_fim,
        color='red', alpha=0.10
    )

    axes[1].plot(t, ewma_std, label='Desvio padrão móvel exponencial')

    axes[1].axhline(
        STD_THRESHOLD_HIGH,
        linestyle='--',
        color='red',
        label=f'Limiar alto = {STD_THRESHOLD_HIGH:.4f}'
    )

    axes[1].axhline(
        STD_THRESHOLD_LOW,
        linestyle='--',
        color='orange',
        label=f'Limiar baixo = {STD_THRESHOLD_LOW:.4f}'
    )

    axes[1].plot(
        t[fault_mask_detected],
        ewma_std[fault_mask_detected],
        'ro',
        markersize=3,
        label='Falha detectada'
    )

    axes[1].set_xlabel('Amostra')
    axes[1].set_ylabel('Desvio padrão')
    axes[1].set_title('Desvio padrão móvel exponencial com detecção de falhas')
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

    print('\nParâmetros do detector:')
    print(f'FAULT_SAMPLES = {FAULT_SAMPLES}')
    print(f'IGNORE_FIRST_SAMPLES = {IGNORE_FIRST_SAMPLES}')
    print(f'PERSISTENCE_SAMPLES = {PERSISTENCE_SAMPLES}')
    print(f'EWMA_ALPHA = {EWMA_ALPHA}')
    print(f'STD_THRESHOLD_HIGH = {STD_THRESHOLD_HIGH}')
    print(f'STD_THRESHOLD_LOW = {STD_THRESHOLD_LOW}')

    # ---------------------------------------------
    # Acumuladores - sem persistência
    # ---------------------------------------------
    global_tp = 0
    global_tn = 0
    global_fp = 0
    global_fn = 0
    detection_delays = []

    # ---------------------------------------------
    # Acumuladores - com persistência
    # ---------------------------------------------
    global_tp_persist = 0
    global_tn_persist = 0
    global_fp_persist = 0
    global_fn_persist = 0
    detection_delays_persist = []

    results_rows = []

    for i, f in enumerate(FILES, start=1):
        file_name = os.path.basename(f)
        print(f'[{i:02d}/{len(FILES):02d}] {file_name}')

        df = read_one_csv(f)
        tc = df['TC'].values

        out = compute_ewma_std(tc, alpha=EWMA_ALPHA)

        ewma_mean = out['ewma_mean']
        residual = out['residual']
        ewma_std = out['ewma_std']

        # -------------------------------------------------
        # DETECÇÃO SEM PERSISTÊNCIA
        # -------------------------------------------------
        fault_mask_detected = detect_fault_points(
            statistic=ewma_std,
            threshold_high=STD_THRESHOLD_HIGH,
            threshold_low=STD_THRESHOLD_LOW,
            ignore_first=IGNORE_FIRST_SAMPLES
        )

        eval_result = evaluate_sample_by_sample(
            fault_mask_detected=fault_mask_detected,
            n_samples=len(df),
            fault_samples=FAULT_SAMPLES
        )

        tp = eval_result['TP']
        tn = eval_result['TN']
        fp = eval_result['FP']
        fn = eval_result['FN']
        fault_start_idx = eval_result['fault_start_idx']

        global_tp += tp
        global_tn += tn
        global_fp += fp
        global_fn += fn

        delay_samples = compute_detection_delay_samples(
            fault_mask_detected=fault_mask_detected,
            fault_start_idx=fault_start_idx
        )

        if not np.isnan(delay_samples):
            detection_delays.append(delay_samples)

        # -------------------------------------------------
        # DETECÇÃO COM PERSISTÊNCIA
        # -------------------------------------------------
        fault_mask_detected_persist = detect_fault_points_with_persistence(
            statistic=ewma_std,
            threshold_high=STD_THRESHOLD_HIGH,
            threshold_low=STD_THRESHOLD_LOW,
            ignore_first=IGNORE_FIRST_SAMPLES,
            persistence=PERSISTENCE_SAMPLES
        )

        eval_result_persist = evaluate_sample_by_sample(
            fault_mask_detected=fault_mask_detected_persist,
            n_samples=len(df),
            fault_samples=FAULT_SAMPLES
        )

        tp_persist = eval_result_persist['TP']
        tn_persist = eval_result_persist['TN']
        fp_persist = eval_result_persist['FP']
        fn_persist = eval_result_persist['FN']

        global_tp_persist += tp_persist
        global_tn_persist += tn_persist
        global_fp_persist += fp_persist
        global_fn_persist += fn_persist

        delay_samples_persist = compute_detection_delay_samples(
            fault_mask_detected=fault_mask_detected_persist,
            fault_start_idx=fault_start_idx
        )

        if not np.isnan(delay_samples_persist):
            detection_delays_persist.append(delay_samples_persist)

        # -------------------------------------------------
        # MÉTRICAS AUXILIARES
        # -------------------------------------------------
        rmse_mean, mae_mean, r2_mean = calc_regression_metrics(tc, ewma_mean)

        # -------------------------------------------------
        # SALVA RESULTADOS POR ARQUIVO
        # -------------------------------------------------
        results_rows.append({
            'Arquivo': file_name,

            'RMSE_TC_vs_EWMA': rmse_mean,
            'MAE_TC_vs_EWMA': mae_mean,
            'R2_TC_vs_EWMA': r2_mean,

            'Media_TC': float(np.mean(tc)),
            'Std_TC': float(np.std(tc)),
            'Media_Residual': float(np.mean(residual)),
            'Std_Residual': float(np.std(residual)),
            'Media_EWMA_STD': float(np.mean(ewma_std)),
            'Std_EWMA_STD': float(np.std(ewma_std)),
            'Max_EWMA_STD': float(np.max(ewma_std)),
            'Min_EWMA_STD': float(np.min(ewma_std)),

            'Pct_EWMA_STD_Acima_Limiar_Alto': float(np.mean(ewma_std > STD_THRESHOLD_HIGH) * 100.0),
            'Pct_EWMA_STD_Abaixo_Limiar_Baixo': float(np.mean(ewma_std < STD_THRESHOLD_LOW) * 100.0),
            'Pct_EWMA_STD_Fora_Da_Faixa': float(
                np.mean((ewma_std > STD_THRESHOLD_HIGH) | (ewma_std < STD_THRESHOLD_LOW)) * 100.0
            ),

            'TP_sem_persistencia': tp,
            'TN_sem_persistencia': tn,
            'FP_sem_persistencia': fp,
            'FN_sem_persistencia': fn,
            'Delay_sem_persistencia': float(delay_samples) if not np.isnan(delay_samples) else np.nan,

            'TP_com_persistencia_100': tp_persist,
            'TN_com_persistencia_100': tn_persist,
            'FP_com_persistencia_100': fp_persist,
            'FN_com_persistencia_100': fn_persist,
            'Delay_com_persistencia_100': float(delay_samples_persist) if not np.isnan(delay_samples_persist) else np.nan,

            'Indice_Inicio_Falha_Real': int(fault_start_idx)
        })

        # -------------------------------------------------
        # GERA GRÁFICO
        # -------------------------------------------------
        save_path_plot = os.path.join(
            OUT_DIR,
            f'stdexp_tc_{file_name.replace(".csv", ".png")}'
        )

        plot_tc_and_std(
            df=df,
            ewma_mean=ewma_mean,
            ewma_std=ewma_std,
            fault_mask_detected=fault_mask_detected,
            file_name=file_name,
            save_path=save_path_plot
        )

    # =====================================================
    # MÉTRICAS GLOBAIS - SEM PERSISTÊNCIA
    # =====================================================
    global_metrics = compute_classification_metrics(
        tp=global_tp,
        tn=global_tn,
        fp=global_fp,
        fn=global_fn
    )

    mean_detection_delay = (
        float(np.mean(detection_delays))
        if len(detection_delays) > 0 else np.nan
    )

    std_detection_delay = (
        float(np.std(detection_delays))
        if len(detection_delays) > 0 else np.nan
    )

    # =====================================================
    # MÉTRICAS GLOBAIS - COM PERSISTÊNCIA
    # =====================================================
    global_metrics_persist = compute_classification_metrics(
        tp=global_tp_persist,
        tn=global_tn_persist,
        fp=global_fp_persist,
        fn=global_fn_persist
    )

    mean_detection_delay_persist = (
        float(np.mean(detection_delays_persist))
        if len(detection_delays_persist) > 0 else np.nan
    )

    std_detection_delay_persist = (
        float(np.std(detection_delays_persist))
        if len(detection_delays_persist) > 0 else np.nan
    )

    # =====================================================
    # DATAFRAMES
    # =====================================================
    results_df = pd.DataFrame(results_rows)

    summary_df = pd.DataFrame([
        {
            'Cenario': 'Sem persistencia',
            'TP_global': global_tp,
            'TN_global': global_tn,
            'FP_global': global_fp,
            'FN_global': global_fn,
            'Acuracia': global_metrics['Acuracia'],
            'Sensibilidade': global_metrics['Sensibilidade'],
            'Especificidade': global_metrics['Especificidade'],
            'Precisao': global_metrics['Precisao'],
            'F1_score': global_metrics['F1_score'],
            'Tempo_Medio_Deteccao_Amostras': mean_detection_delay,
            'Tempo_STD_Deteccao_Amostras': std_detection_delay
        },
        {
            'Cenario': 'Com persistencia de 100 amostras',
            'TP_global': global_tp_persist,
            'TN_global': global_tn_persist,
            'FP_global': global_fp_persist,
            'FN_global': global_fn_persist,
            'Acuracia': global_metrics_persist['Acuracia'],
            'Sensibilidade': global_metrics_persist['Sensibilidade'],
            'Especificidade': global_metrics_persist['Especificidade'],
            'Precisao': global_metrics_persist['Precisao'],
            'F1_score': global_metrics_persist['F1_score'],
            'Tempo_Medio_Deteccao_Amostras': mean_detection_delay_persist,
            'Tempo_STD_Deteccao_Amostras': std_detection_delay_persist
        }
    ])

    # =====================================================
    # SALVA CSVs
    # =====================================================
    results_csv = os.path.join(OUT_DIR, 'metricas_por_arquivo_stdexp_duplo_limiar.csv')
    summary_csv = os.path.join(OUT_DIR, 'resumo_classificacao_global_stdexp_duplo_limiar.csv')

    results_df.to_csv(
        results_csv,
        sep=';',
        decimal=',',
        index=False,
        encoding='utf-8-sig'
    )

    summary_df.to_csv(
        summary_csv,
        sep=';',
        decimal=',',
        index=False,
        encoding='utf-8-sig'
    )

    # =====================================================
    # MATRIZES DE CONFUSÃO
    # =====================================================
    confusion_matrix_png = os.path.join(OUT_DIR, 'matriz_confusao_stdexp_sem_persistencia.png')
    confusion_matrix_persist_png = os.path.join(OUT_DIR, 'matriz_confusao_stdexp_com_persistencia_100.png')

    save_confusion_matrix_figure(
        tp=global_tp,
        tn=global_tn,
        fp=global_fp,
        fn=global_fn,
        save_path=confusion_matrix_png,
        title='Matriz de Confusão - Sem persistência'
    )

    save_confusion_matrix_figure(
        tp=global_tp_persist,
        tn=global_tn_persist,
        fp=global_fp_persist,
        fn=global_fn_persist,
        save_path=confusion_matrix_persist_png,
        title='Matriz de Confusão - Persistência 100 amostras'
    )

    # =====================================================
    # MATRIZES + MÉTRICAS + TEMPO MÉDIO
    # =====================================================
    confusion_matrix_metrics_png = os.path.join(
        OUT_DIR,
        'matriz_confusao_metricas_stdexp_sem_persistencia.png'
    )

    confusion_matrix_metrics_persist_png = os.path.join(
        OUT_DIR,
        'matriz_confusao_metricas_stdexp_com_persistencia_100.png'
    )

    save_confusion_matrix_with_metrics_figure(
        tp=global_tp,
        tn=global_tn,
        fp=global_fp,
        fn=global_fn,
        metrics=global_metrics,
        mean_delay=mean_detection_delay,
        std_delay=std_detection_delay,
        save_path=confusion_matrix_metrics_png,
        title='Sem persistência'
    )

    save_confusion_matrix_with_metrics_figure(
        tp=global_tp_persist,
        tn=global_tn_persist,
        fp=global_fp_persist,
        fn=global_fn_persist,
        metrics=global_metrics_persist,
        mean_delay=mean_detection_delay_persist,
        std_delay=std_detection_delay_persist,
        save_path=confusion_matrix_metrics_persist_png,
        title='Persistência de 100 amostras'
    )

    # =====================================================
    # IMPRESSÃO NO TERMINAL
    # =====================================================
    print('\n' + '=' * 80)
    print('RESULTADOS GLOBAIS - SEM PERSISTÊNCIA')
    print('=' * 80)
    print(f'TP = {global_tp}')
    print(f'TN = {global_tn}')
    print(f'FP = {global_fp}')
    print(f'FN = {global_fn}')
    print('-' * 80)
    print(f"Acurácia       = {global_metrics['Acuracia']:.6f}")
    print(f"Sensibilidade  = {global_metrics['Sensibilidade']:.6f}")
    print(f"Especificidade = {global_metrics['Especificidade']:.6f}")
    print(f"Precisão       = {global_metrics['Precisao']:.6f}")
    print(f"F1-score       = {global_metrics['F1_score']:.6f}")
    if not np.isnan(mean_detection_delay):
        print(f'Tempo médio de detecção = {mean_detection_delay:.2f} amostras')
        print(f'Desvio padrão do tempo de detecção = {std_detection_delay:.2f} amostras')
    else:
        print('Tempo médio de detecção = não foi possível calcular')

    print('\n' + '=' * 80)
    print('RESULTADOS GLOBAIS - COM PERSISTÊNCIA DE 100 AMOSTRAS')
    print('=' * 80)
    print(f'TP = {global_tp_persist}')
    print(f'TN = {global_tn_persist}')
    print(f'FP = {global_fp_persist}')
    print(f'FN = {global_fn_persist}')
    print('-' * 80)
    print(f"Acurácia       = {global_metrics_persist['Acuracia']:.6f}")
    print(f"Sensibilidade  = {global_metrics_persist['Sensibilidade']:.6f}")
    print(f"Especificidade = {global_metrics_persist['Especificidade']:.6f}")
    print(f"Precisão       = {global_metrics_persist['Precisao']:.6f}")
    print(f"F1-score       = {global_metrics_persist['F1_score']:.6f}")
    if not np.isnan(mean_detection_delay_persist):
        print(f'Tempo médio de detecção = {mean_detection_delay_persist:.2f} amostras')
        print(f'Desvio padrão do tempo de detecção = {std_detection_delay_persist:.2f} amostras')
    else:
        print('Tempo médio de detecção = não foi possível calcular')

    print('\n' + '-' * 80)
    print(f'Gráficos salvos em: {OUT_DIR}')
    print(f'Resultados por arquivo: {results_csv}')
    print(f'Resumo global: {summary_csv}')
    print(f'Matriz sem persistência: {confusion_matrix_png}')
    print(f'Matriz com persistência: {confusion_matrix_persist_png}')
    print(f'Matriz + métricas sem persistência: {confusion_matrix_metrics_png}')
    print(f'Matriz + métricas com persistência: {confusion_matrix_metrics_persist_png}')
    print('=' * 80)


if __name__ == '__main__':
    main()
