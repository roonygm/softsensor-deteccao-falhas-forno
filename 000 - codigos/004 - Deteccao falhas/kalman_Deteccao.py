import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# CAMINHOS
# =========================================================
# Diretório onde estão os arquivos CSV de teste
TEST_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\01 - Falha tipo bias'

# Diretório de saída para gráficos e métricas
OUT_DIR = os.path.join(TEST_DIR, 'graficos_kalman_nis_estado_aumentado')
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
# Estado:
# x(k) = [Gf(k), TL_est(k), TR_est(k)]^T
#
# Entrada:
# u(k) = [GAS(k), TL(k), TR(k)]^T
#
# Dinâmica:
# Gf(k+1)     = alpha*Gf(k) + (1-alpha)*GAS(k)
# TL_est(k+1) = TL(k)
# TR_est(k+1) = TR(k)
#
# Saída:
# y(k) = d_Gf*Gf(k) + a_TL*TL_est(k) + b_TR*TR_est(k) + c0

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
# PARÂMETROS DE DETECÇÃO
# =========================================================
# Últimas 3600 amostras são consideradas a região real de falha
FAULT_SAMPLES = 3600

# Limite qui-quadrado com 1 grau de liberdade e 95%
CHI2_LIMIT_95 = 3.841

# Nas primeiras 300 amostras não faz detecção
IGNORE_FIRST_SAMPLES = 300

# Critério: mais de 100 amostras consecutivas acima do limite
# 100 exatas -> não detecta
# 101 ou mais -> detecta
MIN_CONSECUTIVE_FAULT_SAMPLES = 100

# =========================================================
# LEITURA DOS DADOS
# =========================================================
def read_one_csv(path: str) -> pd.DataFrame:
    """
    Lê um arquivo CSV, renomeia colunas de interesse e retorna
    apenas as variáveis necessárias para o filtro.
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
# FILTRO DE KALMAN VETORIAL - ESTADO AUMENTADO
# =========================================================
def kalman_filter_augmented(df, Q, R, x0=None, P0=None):
    """
    Executa o filtro de Kalman para o modelo aumentado.

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
    nx = 3  # número de estados

    # Vetores e matrizes para armazenamento
    x_pred = np.zeros((n, nx))
    x_filt = np.zeros((n, nx))

    P_pred = np.zeros((n, nx, nx))
    P_filt = np.zeros((n, nx, nx))

    y_pred = np.zeros(n)
    y_filt = np.zeros(n)

    innovation = np.zeros(n)
    innovation_var = np.zeros(n)
    kalman_gain = np.zeros((n, nx))

    # Estado inicial
    if x0 is None:
        x0 = np.array([GAS[0], TL[0], TR[0]], dtype=float)

    # Covariância inicial
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
        # ARMAZENAMENTO
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

        # Atualiza para o próximo passo
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
# MÉTRICAS DE REGRESSÃO
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
# DETECÇÃO DE FALHA VIA NIS
# =========================================================
def detect_nis_fault_points(
    nis,
    threshold=CHI2_LIMIT_95,
    ignore_first=IGNORE_FIRST_SAMPLES,
    min_consecutive=MIN_CONSECUTIVE_FAULT_SAMPLES
):
    """
    Marca como falha os pontos pertencentes a trechos com mais de
    `min_consecutive` amostras consecutivas acima do threshold.

    Nas primeiras `ignore_first` amostras não faz detecção.

    Observação:
    Esta função marca o trecho inteiro como falha depois que o critério
    de sequência é satisfeito.
    """
    n = len(nis)
    fault_mask = np.zeros(n, dtype=bool)

    count = 0
    start_idx = None

    for k in range(n):
        if k < ignore_first:
            count = 0
            start_idx = None
            continue

        if nis[k] > threshold:
            if count == 0:
                start_idx = k
            count += 1
        else:
            if count > min_consecutive:
                fault_mask[start_idx:k] = True

            count = 0
            start_idx = None

    # Caso termine ainda dentro de um trecho acima do limite
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

    Retorna:
    - 0 se detectou logo na primeira amostra da falha
    - valor positivo se demorou algumas amostras
    - np.nan se não detectou dentro da região de falha
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
def plot_tc_and_nis(df, y_est, innovation, innovation_var, file_name, save_path):
    """
    Gera gráfico com:
    1) TC real vs TC estimado
    2) NIS + limite + pontos detectados
    """
    t = np.arange(len(df))
    nis = (innovation ** 2) / innovation_var

    fault_mask_detected = detect_nis_fault_points(
        nis,
        threshold=CHI2_LIMIT_95,
        ignore_first=IGNORE_FIRST_SAMPLES,
        min_consecutive=MIN_CONSECUTIVE_FAULT_SAMPLES
    )

    falha_inicio = max(0, len(df) - FAULT_SAMPLES)
    falha_fim = len(df) - 1

    fig, axes = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2]}
    )

    # -------------------------------------------------
    # subplot 1: TC real vs estimado
    # -------------------------------------------------
    axes[0].axvspan(
        falha_inicio, falha_fim,
        color='red', alpha=0.10,
        label='Região real de falha'
    )

    axes[0].plot(t, df['TC'].values, label='TC real')
    axes[0].plot(t, y_est, label='TC estimado (Kalman)')

    axes[0].set_ylabel('Temperatura')
    axes[0].set_title(f'Arquivo {file_name} - TC real vs estimado')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -------------------------------------------------
    # subplot 2: NIS
    # -------------------------------------------------
    axes[1].axvspan(
        falha_inicio, falha_fim,
        color='red', alpha=0.10
    )

    axes[1].plot(t, nis, label='NIS')

    axes[1].axhline(
        CHI2_LIMIT_95,
        linestyle='--',
        color='red',
        label=fr'Limite $\chi^2$ 95% = {CHI2_LIMIT_95:.3f}'
    )

    axes[1].axvline(
        IGNORE_FIRST_SAMPLES,
        linestyle='--',
        color='gray',
        alpha=0.8,
        label=f'Início da detecção (amostra {IGNORE_FIRST_SAMPLES})'
    )

    # Pontos vermelhos onde houve detecção
    axes[1].plot(
        t[fault_mask_detected],
        nis[fault_mask_detected],
        'ro',
        markersize=3,
        label='Falha detectada'
    )

    axes[1].set_xlabel('Amostra')
    axes[1].set_ylabel('NIS')
    axes[1].set_title('Estatística NIS com detecção de falhas')
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

    # -----------------------------------------------------
    # Acumuladores globais da matriz de confusão
    # -----------------------------------------------------
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
        # LEITURA E FILTRO
        # -------------------------------------------------
        df = read_one_csv(f)

        out = kalman_filter_augmented(df, Q=Q, R=R, x0=None, P0=None)

        y_true_signal = df['TC'].values
        y_est = out['y_filt']
        innovation = out['innovation']
        innovation_var = out['innovation_var']

        nis = (innovation ** 2) / innovation_var

        # -------------------------------------------------
        # DETECÇÃO DE FALHAS
        # -------------------------------------------------
        fault_mask_detected = detect_nis_fault_points(
            nis,
            threshold=CHI2_LIMIT_95,
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
        # MÉTRICAS DE REGRESSÃO
        # -------------------------------------------------
        rmse, mae, r2 = calc_regression_metrics(y_true_signal, y_est)

        # -------------------------------------------------
        # SALVA RESULTADOS POR ARQUIVO
        # -------------------------------------------------
        results_rows.append({
            'Arquivo': file_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Media_Inovacao': float(np.mean(innovation)),
            'Std_Inovacao': float(np.std(innovation)),
            'Media_NIS': float(np.mean(nis)),
            'Std_NIS': float(np.std(nis)),
            'Max_NIS': float(np.max(nis)),
            'Pct_NIS_Acima_Limite_95': float(np.mean(nis > CHI2_LIMIT_95) * 100.0),

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
        save_path_tc = os.path.join(
            OUT_DIR,
            f'kalman_tc_nis_{file_name.replace(".csv", ".png")}'
        )

        plot_tc_and_nis(
            df=df,
            y_est=y_est,
            innovation=innovation,
            innovation_var=innovation_var,
            file_name=file_name,
            save_path=save_path_tc
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
    results_csv = os.path.join(OUT_DIR, 'metricas_por_arquivo.csv')
    summary_csv = os.path.join(OUT_DIR, 'resumo_classificacao_global.csv')

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
    confusion_matrix_png = os.path.join(OUT_DIR, 'matriz_confusao.png')

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
    print('RESULTADOS GLOBAIS POR REGIÃO')
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