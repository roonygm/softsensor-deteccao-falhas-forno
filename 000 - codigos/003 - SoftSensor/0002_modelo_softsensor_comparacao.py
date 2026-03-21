import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DATA_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo'
OUT_DIR = os.path.join(DATA_DIR, 'comparacao_3_modelos_softsensor')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20  # restante = teste

# semente para embaralhar os arquivos antes da divisão
RANDOM_SEED = 42

# atraso sugerido pela correlação cruzada
GAS_DELAY_CANDIDATE = 181

# faixa de tau coerente com a escala temporal encontrada
TAU_CANDIDATES = [120, 140, 160, 180, 200, 220, 240]
TS = 1.0  # tempo de amostragem em segundos

# número máximo de pontos mostrados nos gráficos de séries
PLOT_MAX_POINTS = 1500

# =========================================================
# ARQUIVOS
# =========================================================
FILES = sorted(
    f for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

if len(FILES) < 5:
    raise RuntimeError('Poucos arquivos encontrados. Verifique DATA_DIR.')

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
# SPLIT COM EMBARALHAMENTO
# =========================================================
def split_files(files, train_ratio=0.60, val_ratio=0.20, random_seed=42):
    rng = np.random.default_rng(random_seed)
    files = list(rng.permutation(files))

    n = len(files)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    return train_files, val_files, test_files

# =========================================================
# MÉTRICAS
# =========================================================
def calc_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# =========================================================
# FILTRO DE 1ª ORDEM
# =========================================================
def first_order_filter(u, alpha):
    u = np.asarray(u, dtype=float)
    gf = np.zeros_like(u)
    gf[0] = u[0]
    for k in range(1, len(u)):
        gf[k] = alpha * gf[k-1] + (1 - alpha) * u[k]
    return gf

def tau_to_alpha(tau, Ts=1.0):
    return float(np.exp(-Ts / tau))

# =========================================================
# MATRIZES DOS MODELOS
# =========================================================
def build_matrix_model_1(df):
    # modelo instantâneo
    y = df['TC'].values
    X = np.column_stack([
        np.ones(len(df)),
        df['TL'].values,
        df['TR'].values,
        df['GAS'].values
    ])
    return X, y

def build_matrix_model_2(df, gas_delay):
    # gás com atraso puro
    max_lag = gas_delay
    n = len(df)
    if n <= max_lag + 5:
        raise ValueError('Arquivo curto demais para o atraso do gás.')

    y = df['TC'].values[max_lag:]
    tl = df['TL'].values[max_lag:]
    tr = df['TR'].values[max_lag:]
    gas = df['GAS'].values[max_lag - gas_delay : n - gas_delay]

    X = np.column_stack([np.ones(len(y)), tl, tr, gas])
    return X, y

def build_matrix_model_3(df, alpha):
    # gás filtrado por 1ª ordem
    y = df['TC'].values
    gf = first_order_filter(df['GAS'].values, alpha)
    X = np.column_stack([
        np.ones(len(df)),
        df['TL'].values,
        df['TR'].values,
        gf
    ])
    return X, y

def stack_files(files, model_type, gas_delay=None, alpha=None):
    Xs, ys = [], []
    for f in files:
        df = read_one_csv(f)

        if model_type == 1:
            X, y = build_matrix_model_1(df)
        elif model_type == 2:
            X, y = build_matrix_model_2(df, gas_delay=gas_delay)
        elif model_type == 3:
            X, y = build_matrix_model_3(df, alpha=alpha)
        else:
            raise ValueError('model_type inválido')

        Xs.append(X)
        ys.append(y)

    return np.vstack(Xs), np.hstack(ys)

# =========================================================
# AJUSTE
# =========================================================
def fit_ols(X, y):
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta

def predict(X, theta):
    return X @ theta

# =========================================================
# BUSCA DO MODELO COM FILTRO
# =========================================================
def search_best_alpha(train_files, val_files, tau_candidates):
    rows = []
    for tau in tau_candidates:
        alpha = tau_to_alpha(tau, TS)

        Xtr, ytr = stack_files(train_files, model_type=3, alpha=alpha)
        Xval, yval = stack_files(val_files, model_type=3, alpha=alpha)

        theta = fit_ols(Xtr, ytr)

        yhat_tr = predict(Xtr, theta)
        yhat_val = predict(Xval, theta)

        met_tr = calc_metrics(ytr, yhat_tr)
        met_val = calc_metrics(yval, yhat_val)

        rows.append({
            'tau_s': tau,
            'alpha': alpha,
            'c': theta[0],
            'a_tl': theta[1],
            'b_tr': theta[2],
            'd_gf': theta[3],
            'RMSE_train': met_tr['RMSE'],
            'MAE_train': met_tr['MAE'],
            'R2_train': met_tr['R2'],
            'RMSE_val': met_val['RMSE'],
            'MAE_val': met_val['MAE'],
            'R2_val': met_val['R2']
        })

    return pd.DataFrame(rows).sort_values('RMSE_val').reset_index(drop=True)

# =========================================================
# AVALIAÇÃO FINAL
# =========================================================
def fit_and_evaluate_model(dev_files, test_files, model_type, gas_delay=None, alpha=None):
    Xdev, ydev = stack_files(dev_files, model_type=model_type, gas_delay=gas_delay, alpha=alpha)
    Xte, yte   = stack_files(test_files, model_type=model_type, gas_delay=gas_delay, alpha=alpha)

    theta = fit_ols(Xdev, ydev)

    yhat_dev = predict(Xdev, theta)
    yhat_te  = predict(Xte, theta)

    met_dev = calc_metrics(ydev, yhat_dev)
    met_te  = calc_metrics(yte, yhat_te)

    return theta, met_dev, met_te

def metrics_per_file(files, theta, model_type, gas_delay=None, alpha=None):
    rows = []
    example = None

    for f in files:
        df = read_one_csv(f)

        if model_type == 1:
            X, y = build_matrix_model_1(df)
        elif model_type == 2:
            X, y = build_matrix_model_2(df, gas_delay=gas_delay)
        elif model_type == 3:
            X, y = build_matrix_model_3(df, alpha=alpha)
        else:
            raise ValueError('model_type inválido')

        yhat = predict(X, theta)
        met = calc_metrics(y, yhat)

        rows.append({
            'arquivo': os.path.basename(f),
            'RMSE': met['RMSE'],
            'MAE': met['MAE'],
            'R2': met['R2']
        })

        if example is None:
            example = {
                'arquivo': os.path.basename(f),
                'y': y,
                'yhat': yhat,
                'res': y - yhat
            }

    return pd.DataFrame(rows), example

# =========================================================
# GRÁFICOS
# =========================================================
def plot_alpha_search(search_df, save_path):
    d = search_df.sort_values('tau_s')

    plt.figure(figsize=(10, 5))
    plt.plot(d['tau_s'], d['RMSE_val'], marker='o', label='RMSE validação')

    best = search_df.iloc[0]
    plt.axvline(best['tau_s'], linestyle='--',
                label=f"melhor tau = {int(best['tau_s'])} s | alpha = {best['alpha']:.4f}")

    plt.xlabel('Constante de tempo tau (s)')
    plt.ylabel('RMSE')
    plt.title('Busca da dinâmica do filtro de 1ª ordem do gás')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_model_comparison_bar(summary_df, save_path):
    d = summary_df.copy()

    plt.figure(figsize=(9, 5))
    plt.bar(d['modelo'], d['RMSE_test'])
    plt.ylabel('RMSE teste')
    plt.title('Comparação entre modelos no conjunto de teste')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_example_predictions(examples_dict, save_path):
    plt.figure(figsize=(12, 7))

    for i, (name, ex) in enumerate(examples_dict.items(), start=1):
        ax = plt.subplot(3, 1, i)
        y = ex['y']
        yhat = ex['yhat']
        stop = min(len(y), PLOT_MAX_POINTS)

        ax.plot(np.arange(stop), y[:stop], label='TC medida')
        ax.plot(np.arange(stop), yhat[:stop], label='TC estimada')
        ax.set_title(f'{name} | exemplo: {ex["arquivo"]}')
        ax.set_xlabel('Amostra')
        ax.set_ylabel('Temperatura')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_example_residuals(examples_dict, save_path):
    plt.figure(figsize=(12, 7))

    for i, (name, ex) in enumerate(examples_dict.items(), start=1):
        ax = plt.subplot(3, 1, i)
        res = ex['res']
        stop = min(len(res), PLOT_MAX_POINTS)

        ax.plot(np.arange(stop), res[:stop])
        ax.axhline(0, linestyle='--')
        ax.set_title(f'{name} | resíduos | exemplo: {ex["arquivo"]}')
        ax.set_xlabel('Amostra')
        ax.set_ylabel('Resíduo')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_rmse_per_file(models_metrics, save_path):
    model_names = list(models_metrics.keys())
    files = models_metrics[model_names[0]]['arquivo'].tolist()

    x = np.arange(len(files))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for i, name in enumerate(model_names):
        vals = models_metrics[name]['RMSE'].values
        plt.bar(x + (i-1)*width, vals, width=width, label=name)

    plt.xticks(x, files, rotation=90)
    plt.ylabel('RMSE')
    plt.title('RMSE por arquivo no conjunto de teste')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_gas_vs_gf_example(test_files, alpha, save_path):
    if len(test_files) == 0:
        return

    df = read_one_csv(test_files[0])
    gas = df['GAS'].values
    gf = first_order_filter(gas, alpha)

    stop = min(len(gas), PLOT_MAX_POINTS)

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(stop), gas[:stop], label='GAS')
    plt.plot(np.arange(stop), gf[:stop], label='Gf (filtrado)')
    plt.xlabel('Amostra')
    plt.ylabel('Sinal')
    plt.title('Exemplo do efeito do filtro de 1ª ordem sobre o gás')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =========================================================
# MAIN
# =========================================================
def main():
    train_files, val_files, test_files = split_files(
        FILES, TRAIN_RATIO, VAL_RATIO, random_seed=RANDOM_SEED
    )
    dev_files = train_files + val_files

    print('=' * 80)
    print(f'Total de arquivos : {len(FILES)}')
    print(f'Treino            : {len(train_files)}')
    print(f'Validação         : {len(val_files)}')
    print(f'Teste             : {len(test_files)}')
    print(f'Random seed       : {RANDOM_SEED}')
    print('=' * 80)

    print('\nArquivos de treino:')
    for f in train_files:
        print('  ', os.path.basename(f))

    print('\nArquivos de validação:')
    for f in val_files:
        print('  ', os.path.basename(f))

    print('\nArquivos de teste:')
    for f in test_files:
        print('  ', os.path.basename(f))

    # -----------------------------------------------------
    # MODELO 1: totalmente instantâneo
    # -----------------------------------------------------
    theta_m1, met_dev_m1, met_te_m1 = fit_and_evaluate_model(dev_files, test_files, model_type=1)
    metrics_m1, example_m1 = metrics_per_file(test_files, theta_m1, model_type=1)

    # -----------------------------------------------------
    # MODELO 2: gás com atraso puro
    # -----------------------------------------------------
    theta_m2, met_dev_m2, met_te_m2 = fit_and_evaluate_model(
        dev_files, test_files, model_type=2, gas_delay=GAS_DELAY_CANDIDATE
    )
    metrics_m2, example_m2 = metrics_per_file(test_files, theta_m2, model_type=2, gas_delay=GAS_DELAY_CANDIDATE)

    # -----------------------------------------------------
    # MODELO 3: gás com filtro de 1ª ordem
    # busca do melhor tau/alpha em treino-validação
    # -----------------------------------------------------
    search_df = search_best_alpha(train_files, val_files, TAU_CANDIDATES)
    search_df.to_csv(
        os.path.join(OUT_DIR, '01_busca_alpha_filtro.csv'),
        sep=';', decimal=',', index=False, encoding='utf-8-sig'
    )

    best = search_df.iloc[0]
    best_tau = float(best['tau_s'])
    best_alpha = float(best['alpha'])

    theta_m3, met_dev_m3, met_te_m3 = fit_and_evaluate_model(
        dev_files, test_files, model_type=3, alpha=best_alpha
    )
    metrics_m3, example_m3 = metrics_per_file(test_files, theta_m3, model_type=3, alpha=best_alpha)

    # -----------------------------------------------------
    # TABELA COMPARATIVA FINAL
    # -----------------------------------------------------
    summary_df = pd.DataFrame([
        {
            'modelo': 'M1 - Instantâneo',
            'estrutura': 'TC = c + a*TL(k) + b*TR(k) + d*GAS(k)',
            'gas_delay': 0,
            'tau_s': np.nan,
            'alpha': np.nan,
            'c': theta_m1[0], 'a_tl': theta_m1[1], 'b_tr': theta_m1[2], 'd_gas_gf': theta_m1[3],
            'RMSE_dev': met_dev_m1['RMSE'], 'MAE_dev': met_dev_m1['MAE'], 'R2_dev': met_dev_m1['R2'],
            'RMSE_test': met_te_m1['RMSE'], 'MAE_test': met_te_m1['MAE'], 'R2_test': met_te_m1['R2']
        },
        {
            'modelo': f'M2 - GAS atraso {GAS_DELAY_CANDIDATE}',
            'estrutura': f'TC = c + a*TL(k) + b*TR(k) + d*GAS(k-{GAS_DELAY_CANDIDATE})',
            'gas_delay': GAS_DELAY_CANDIDATE,
            'tau_s': np.nan,
            'alpha': np.nan,
            'c': theta_m2[0], 'a_tl': theta_m2[1], 'b_tr': theta_m2[2], 'd_gas_gf': theta_m2[3],
            'RMSE_dev': met_dev_m2['RMSE'], 'MAE_dev': met_dev_m2['MAE'], 'R2_dev': met_dev_m2['R2'],
            'RMSE_test': met_te_m2['RMSE'], 'MAE_test': met_te_m2['MAE'], 'R2_test': met_te_m2['R2']
        },
        {
            'modelo': 'M3 - GAS filtrado',
            'estrutura': 'TC = c + a*TL(k) + b*TR(k) + d*Gf(k)',
            'gas_delay': np.nan,
            'tau_s': best_tau,
            'alpha': best_alpha,
            'c': theta_m3[0], 'a_tl': theta_m3[1], 'b_tr': theta_m3[2], 'd_gas_gf': theta_m3[3],
            'RMSE_dev': met_dev_m3['RMSE'], 'MAE_dev': met_dev_m3['MAE'], 'R2_dev': met_dev_m3['R2'],
            'RMSE_test': met_te_m3['RMSE'], 'MAE_test': met_te_m3['MAE'], 'R2_test': met_te_m3['R2']
        }
    ])

    summary_df.to_csv(
        os.path.join(OUT_DIR, '02_tabela_comparacao_modelos.csv'),
        sep=';', decimal=',', index=False, encoding='utf-8-sig'
    )

    # métricas por arquivo
    metrics_m1.to_csv(os.path.join(OUT_DIR, '03_metricas_teste_M1.csv'),
                      sep=';', decimal=',', index=False, encoding='utf-8-sig')
    metrics_m2.to_csv(os.path.join(OUT_DIR, '04_metricas_teste_M2.csv'),
                      sep=';', decimal=',', index=False, encoding='utf-8-sig')
    metrics_m3.to_csv(os.path.join(OUT_DIR, '05_metricas_teste_M3.csv'),
                      sep=';', decimal=',', index=False, encoding='utf-8-sig')

    # -----------------------------------------------------
    # GRÁFICOS
    # -----------------------------------------------------
    plot_alpha_search(search_df, os.path.join(OUT_DIR, '06_busca_tau_alpha_modelo_filtrado.png'))
    plot_model_comparison_bar(summary_df, os.path.join(OUT_DIR, '07_comparacao_rmse_modelos.png'))

    examples_dict = {
        'M1 - Instantâneo': example_m1,
        f'M2 - GAS atraso {GAS_DELAY_CANDIDATE}': example_m2,
        f'M3 - Filtrado (tau={int(best_tau)} s)': example_m3
    }
    plot_example_predictions(examples_dict, os.path.join(OUT_DIR, '08_predicoes_exemplo_teste.png'))
    plot_example_residuals(examples_dict, os.path.join(OUT_DIR, '09_residuos_exemplo_teste.png'))

    models_metrics = {
        'M1': metrics_m1,
        'M2': metrics_m2,
        'M3': metrics_m3
    }
    plot_rmse_per_file(models_metrics, os.path.join(OUT_DIR, '10_rmse_por_arquivo_teste.png'))
    plot_gas_vs_gf_example(test_files, best_alpha, os.path.join(OUT_DIR, '11_exemplo_gas_vs_gf.png'))

    print('\nMelhor busca do modelo filtrado:')
    print(best[['tau_s', 'alpha', 'RMSE_train', 'RMSE_val', 'R2_train', 'R2_val']])

    print('\nTabela comparativa final:')
    print(summary_df[['modelo', 'RMSE_dev', 'RMSE_test', 'MAE_test', 'R2_test']])

    print('\nArquivos salvos em:')
    print(OUT_DIR)

if __name__ == '__main__':
    main()