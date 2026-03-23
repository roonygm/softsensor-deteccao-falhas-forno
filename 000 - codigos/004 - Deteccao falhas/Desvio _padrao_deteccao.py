import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
# Diretório onde estão os arquivos CSV de teste
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\03 - Falha tipo congelamento'

# Diretório de saída para gráficos e métricas
OUT_DIR = os.path.join(TEST_DIR, 'graficos_std_exp_tc_duplo_limiar')
os.makedirs(OUT_DIR, exist_ok=True)

# Lista todos os arquivos .csv cujo nome seja numérico
FILES = sorted(
    f for f in glob.glob(os.path.join(TEST_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

# Filtra apenas arquivos de 1 até 50
FILES = [
    f for f in FILES
    if 1 <= int(os.path.basename(f).replace('.csv', '')) <= 50
]

# =========================================================
# PARÂMETROS DE DETECÇÃO
# =========================================================
# Últimas 3600 amostras são consideradas a região real de falha
FAULT_SAMPLES = 3600

# Nas primeiras 300 amostras não faz detecção
IGNORE_FIRST_SAMPLES = 300

# Critério: mais de 100 amostras consecutivas fora da faixa
# 100 exatas -> não detecta
# 101 ou mais -> detecta
MIN_CONSECUTIVE_FAULT_SAMPLES = 100

# =========================================================
# PARÂMETROS DO DESVIO PADRÃO MÓVEL EXPONENCIAL
# =========================================================
# Fator de suavização exponencial
EWMA_ALPHA = 0.01

# Faixa normal do desvio padrão exponencial
# Detecta falha se:
#   ewma_std > STD_THRESHOLD_HIGH
#   ou
#   ewma_std < STD_THRESHOLD_LOW
STD_THRESHOLD_HIGH = 4.0
STD_THRESHOLD_LOW = 0.2

# =========================================================
# LEITURA DOS DADOS
# =========================================================
def read_one_csv(path: str) -> pd.DataFrame:
    """
    Lê um arquivo CSV, renomeia colunas de interesse e retorna
    apenas as variáveis necessárias.
    """
    df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8', engine='python')

    # Remove possíveis caracteres estranhos no cabeçalho
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]

    # Renomeia as colunas para nomes simplificados
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

    # Mantém apenas as colunas necessárias
    df = df[required].copy()

    # Converte para numérico
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove linhas inválidas
    df = df.dropna().reset_index(drop=True)

    return df

# =========================================================
# MÉTRICAS AUXILIARES
# =========================================================
def calc_regression_metrics(y_true, y_pred):
    """
    Calcula RMSE, MAE e R².
    """
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
    """
    Calcula:
    - média móvel exponencial do sinal
    - resíduo = sinal - média
    - variância móvel exponencial do resíduo
    - desvio padrão móvel exponencial
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    ewma_mean = np.zeros(n, dtype=float)
    residual = np.zeros(n, dtype=float)
    ewma_var = np.zeros(n, dtype=float)
    ewma_std = np.zeros(n, dtype=float)

    # Inicialização
    ewma_mean[0] = signal[0]
    residual[0] = 0.0
    ewma_var[0] = 0.0
    ewma_std[0] = 0.0

    for k in range(1, n):
        # Média móvel exponencial
        ewma_mean[k] = alpha * signal[k] + (1 - alpha) * ewma_mean[k - 1]

        # Resíduo
        residual[k] = signal[k] - ewma_mean[k]

        # Variância móvel exponencial do resíduo
        ewma_var[k] = alpha * (residual[k] ** 2) + (1 - alpha) * ewma_var[k - 1]

        # Desvio padrão
        ewma_std[k] = np.sqrt(max(ewma_var[k], 0.0))

    return {
        'ewma_mean': ewma_mean,
        'residual': residual,
        'ewma_var': ewma_var,
        'ewma_std': ewma_std
    }

# =========================================================
# DETECÇÃO COM DOIS LIMIARES E PERSISTÊNCIA
# =========================================================
def detect_fault_points_with_persistence(
    statistic,
    threshold_high=STD_THRESHOLD_HIGH,
    threshold_low=STD_THRESHOLD_LOW,
    ignore_first=IGNORE_FIRST_SAMPLES,
    min_consecutive=MIN_CONSECUTIVE_FAULT_SAMPLES
):
    """
    Marca como falha os pontos pertencentes a trechos com mais de
    `min_consecutive` amostras consecutivas fora da faixa normal:

        statistic > threshold_high
        ou
        statistic < threshold_low

    Nas primeiras `ignore_first` amostras não faz detecção.
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
            if count > min_consecutive:
                fault_mask[start_idx:k] = True
            count = 0
            start_idx = None

    # Caso termine ainda dentro de um trecho fora da faixa
    if count > min_consecutive and start_idx is not None:
        fault_mask[start_idx:n] = True

    return fault_mask

# =========================================================
# TEMPO DE DETECÇÃO
# =========================================================
def compute_detection_delay_samples(fault_mask_detected, fault_start_idx):
    """
    Calcula o atraso de detecção em número de amostras,
    usando a primeira detecção dentro da região real de falha.
    """
    idx_detected = np.where(fault_mask_detected[fault_start_idx:])[0]

    if len(idx_detected) == 0:
        return np.nan

    return int(idx_detected[0])

# =========================================================
# AVALIAÇÃO POR REGIÃO DO ARQUIVO
# =========================================================
def evaluate_file_by_regions(fault_mask_detected, n_samples, fault_samples=FAULT_SAMPLES):
    """
    Avalia cada arquivo em duas regiões:
    - região normal: antes do início da falha
    - região de falha: últimas `fault_samples` amostras

    Regras:
    Região normal:
    - detectou alguma falha -> 1 FP
    - não detectou falha    -> 1 TN

    Região de falha:
    - detectou alguma falha -> 1 TP
    - não detectou falha    -> 1 FN
    """
    fault_start_idx = max(0, n_samples - fault_samples)

    normal_region = fault_mask_detected[:fault_start_idx]
    fault_region = fault_mask_detected[fault_start_idx:]

    detected_in_normal = bool(np.any(normal_region))
    detected_in_fault = bool(np.any(fault_region))

    fp = int(detected_in_normal)
    tn = int(not detected_in_normal)

    tp = int(detected_in_fault)
    fn = int(not detected_in_fault)

    return {
        'fault_start_idx': fault_start_idx,
        'detected_in_normal': detected_in_normal,
        'detected_in_fault': detected_in_fault,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

# =========================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# =========================================================
def compute_classification_metrics(tp, tn, fp, fn):
    """
    Calcula métricas a partir da matriz de confusão.
    """
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
def save_confusion_matrix_figure(tp, tn, fp, fn, save_path):
    """
    Salva uma imagem da matriz de confusão.

    Organização:
                 Predito
               Normal  Falha
    Real Normal   TN      FP
    Real Falha    FN      TP
    """
    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_title('Matriz de Confusão')
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
# PLOT POR ARQUIVO
# =========================================================
def plot_tc_and_std(df, ewma_mean, ewma_std, fault_mask_detected, file_name, save_path):
    """
    Gera gráfico com:
    1) TC real e média móvel exponencial
    2) desvio padrão móvel exponencial + limiares + pontos detectados
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

    # -------------------------------------------------
    # subplot 1: TC real
    # -------------------------------------------------
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

    # -------------------------------------------------
    # subplot 2: desvio padrão móvel exponencial
    # -------------------------------------------------
    axes[1].axvspan(
        falha_inicio, falha_fim,
        color='red', alpha=0.10
    )

    axes[1].plot(t, ewma_std, label='Desvio padrão móvel exponencial')

    axes[1].axhline(
        STD_THRESHOLD_HIGH,
        linestyle='--',
        color='red',
        label=f'Limiar alto = {STD_THRESHOLD_HIGH:.3f}'
    )

    axes[1].axhline(
        STD_THRESHOLD_LOW,
        linestyle='--',
        color='orange',
        label=f'Limiar baixo = {STD_THRESHOLD_LOW:.3f}'
    )

    axes[1].axvline(
        IGNORE_FIRST_SAMPLES,
        linestyle='--',
        color='gray',
        alpha=0.8,
        label=f'Início da detecção (amostra {IGNORE_FIRST_SAMPLES})'
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
    print(f'MIN_CONSECUTIVE_FAULT_SAMPLES = {MIN_CONSECUTIVE_FAULT_SAMPLES}')
    print(f'EWMA_ALPHA = {EWMA_ALPHA}')
    print(f'STD_THRESHOLD_HIGH = {STD_THRESHOLD_HIGH}')
    print(f'STD_THRESHOLD_LOW = {STD_THRESHOLD_LOW}')

    # Acumuladores globais da matriz de confusão
    global_tp = 0
    global_tn = 0
    global_fp = 0
    global_fn = 0

    # Lista para tempos de detecção
    detection_delays = []

    # Lista para salvar resultados por arquivo
    results_rows = []

    for i, f in enumerate(FILES, start=1):
        file_name = os.path.basename(f)
        print(f'[{i:02d}/{len(FILES):02d}] {file_name}')

        # -------------------------------------------------
        # LEITURA
        # -------------------------------------------------
        df = read_one_csv(f)
        tc = df['TC'].values

        # -------------------------------------------------
        # CÁLCULO DO DESVIO PADRÃO MÓVEL EXPONENCIAL
        # -------------------------------------------------
        out = compute_ewma_std(tc, alpha=EWMA_ALPHA)

        ewma_mean = out['ewma_mean']
        residual = out['residual']
        ewma_std = out['ewma_std']

        # -------------------------------------------------
        # DETECÇÃO DE FALHAS
        # -------------------------------------------------
        fault_mask_detected = detect_fault_points_with_persistence(
            statistic=ewma_std,
            threshold_high=STD_THRESHOLD_HIGH,
            threshold_low=STD_THRESHOLD_LOW,
            ignore_first=IGNORE_FIRST_SAMPLES,
            min_consecutive=MIN_CONSECUTIVE_FAULT_SAMPLES
        )

        # -------------------------------------------------
        # AVALIAÇÃO POR REGIÃO DO ARQUIVO
        # -------------------------------------------------
        eval_result = evaluate_file_by_regions(
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

        # -------------------------------------------------
        # TEMPO DE DETECÇÃO
        # -------------------------------------------------
        delay_samples = compute_detection_delay_samples(
            fault_mask_detected=fault_mask_detected,
            fault_start_idx=fault_start_idx
        )

        if not np.isnan(delay_samples):
            detection_delays.append(delay_samples)

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

            'Detectou_Regiao_Normal': int(eval_result['detected_in_normal']),
            'Detectou_Regiao_Falha': int(eval_result['detected_in_fault']),

            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,

            'Indice_Inicio_Falha_Real': int(fault_start_idx),
            'Delay_Deteccao_Amostras': float(delay_samples) if not np.isnan(delay_samples) else np.nan
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
    # MÉTRICAS GLOBAIS
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
    # DATAFRAMES DE SAÍDA
    # =====================================================
    results_df = pd.DataFrame(results_rows)

    summary_df = pd.DataFrame([{
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
    }])

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
    # SALVA FIGURA DA MATRIZ DE CONFUSÃO
    # =====================================================
    confusion_matrix_png = os.path.join(OUT_DIR, 'matriz_confusao_stdexp_duplo_limiar.png')

    save_confusion_matrix_figure(
        tp=global_tp,
        tn=global_tn,
        fp=global_fp,
        fn=global_fn,
        save_path=confusion_matrix_png
    )

    # =====================================================
    # IMPRESSÃO NO TERMINAL
    # =====================================================
    print('\n' + '=' * 80)
    print('RESULTADOS GLOBAIS POR REGIÃO - DESVIO PADRÃO EXPONENCIAL')
    print('=' * 80)
    print(f'TP = {global_tp}')
    print(f'TN = {global_tn}')
    print(f'FP = {global_fp}')
    print(f'FN = {global_fn}')
    print('-' * 80)
    print(f"Acurácia      = {global_metrics['Acuracia']:.6f}")
    print(f"Sensibilidade = {global_metrics['Sensibilidade']:.6f}")
    print(f"Especificidade= {global_metrics['Especificidade']:.6f}")
    print(f"Precisão      = {global_metrics['Precisao']:.6f}")
    print(f"F1-score      = {global_metrics['F1_score']:.6f}")
    print('-' * 80)

    if not np.isnan(mean_detection_delay):
        print(f'Tempo médio de detecção = {mean_detection_delay:.2f} amostras')
        print(f'Desvio padrão do tempo de detecção = {std_detection_delay:.2f} amostras')
    else:
        print('Tempo médio de detecção = não foi possível calcular')

    print('-' * 80)
    print(f'Gráficos salvos em: {OUT_DIR}')
    print(f'Resultados por arquivo salvos em: {results_csv}')
    print(f'Resumo global salvo em: {summary_csv}')
    print(f'Matriz de confusão salva em: {confusion_matrix_png}')
    print('=' * 80)

if __name__ == '__main__':
    main()