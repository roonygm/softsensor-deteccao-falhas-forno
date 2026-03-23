import os
import glob
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURAÇÕES
# =========================================================
DATA_DIR = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo'
OUT_DIR = os.path.join(DATA_DIR, 'estimativa_R_nova_arquitetura')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = sorted(
    f for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if os.path.basename(f).replace('.csv', '').isdigit()
)

# =========================================================
# PARÂMETROS DO MODELO
# =========================================================
alpha = 0.9945
beta = 1.0 - alpha

c0 = 27.40
a_tl = 0.401
b_tr = 0.551
d_gf = 0.094

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
        print(f'\nArquivo com problema: {path}')
        print('Colunas encontradas:', df.columns.tolist())
        raise ValueError(f'Colunas faltando após rename: {missing}')

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# CÁLCULO DE Gf
# =========================================================
def compute_gf(gas, alpha=0.9945, beta=0.0055, gf0=None):
    gas = np.asarray(gas, dtype=float)
    gf = np.zeros_like(gas, dtype=float)

    if len(gas) == 0:
        return gf

    if gf0 is None:
        gf[0] = gas[0]
    else:
        gf[0] = gf0

    for k in range(1, len(gas)):
        gf[k] = alpha * gf[k - 1] + beta * gas[k]

    return gf

# =========================================================
# NOVA ARQUITETURA EM ESPAÇO DE ESTADOS
# =========================================================
def compute_yhat(
    df: pd.DataFrame,
    c0: float = 27.40,
    a_tl: float = 0.401,
    b_tr: float = 0.551,
    d_gf: float = 0.094,
    alpha: float = 0.9945,
    beta: float = 0.0055,
    gf0=None
) -> pd.DataFrame:
    """
    Nova arquitetura:

    Estados:
        x(k) = [Gf(k), TL_est(k), TR_est(k)]^T

    Dinâmica:
        Gf(k+1)     = alpha * Gf(k) + beta * GAS(k)
        TL_est(k+1) = TL(k)
        TR_est(k+1) = TR(k)

    Saída:
        Yhat(k) = c0 + d_gf*Gf(k) + a_tl*TL_est(k) + b_tr*TR_est(k)
    """

    out = df.copy()

    # ---------------------------------------------
    # Estado Gf
    # ---------------------------------------------
    out['Gf'] = compute_gf(
        gas=out['GAS'].values,
        alpha=alpha,
        beta=beta,
        gf0=gf0
    )

    # ---------------------------------------------
    # Estados TL_est e TR_est
    # ---------------------------------------------
    tl = out['TL'].to_numpy(dtype=float)
    tr = out['TR'].to_numpy(dtype=float)

    tl_est = np.zeros(len(out), dtype=float)
    tr_est = np.zeros(len(out), dtype=float)

    if len(out) > 0:
        # condição inicial
        tl_est[0] = tl[0]
        tr_est[0] = tr[0]

        # atualização dos estados
        for k in range(1, len(out)):
            tl_est[k] = tl[k - 1]
            tr_est[k] = tr[k - 1]

    out['TL_est'] = tl_est
    out['TR_est'] = tr_est

    # ---------------------------------------------
    # Saída estimada
    # ---------------------------------------------
    out['Yhat'] = (
        c0
        + d_gf * out['Gf']
        + a_tl * out['TL_est']
        + b_tr * out['TR_est']
    )

    # ---------------------------------------------
    # Resíduo
    # ---------------------------------------------
    out['residuo'] = out['TC'] - out['Yhat']

    return out

# =========================================================
# ESTIMAÇÃO DE R
# =========================================================
def estimate_R_from_residual(residual, ddof=1):
    residual = np.asarray(residual, dtype=float)

    if len(residual) <= ddof:
        return np.nan

    return float(np.var(residual, ddof=ddof))

# =========================================================
# MÉTRICAS AUXILIARES
# =========================================================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return rmse, mae, r2

# =========================================================
# PROCESSAMENTO DE UM ARQUIVO
# =========================================================
def process_one_file(path: str) -> dict:
    df = read_one_csv(path)

    out = compute_yhat(
        df=df,
        c0=c0,
        a_tl=a_tl,
        b_tr=b_tr,
        d_gf=d_gf,
        alpha=alpha,
        beta=beta,
        gf0=None
    )

    R_est = estimate_R_from_residual(out['residuo'].values, ddof=1)
    rmse, mae, r2 = calc_metrics(out['TC'].values, out['Yhat'].values)

    result = {
        'Arquivo': os.path.basename(path),
        'N_amostras': len(out),
        'Media_TC': float(out['TC'].mean()),
        'Media_Yhat': float(out['Yhat'].mean()),
        'Media_residuo': float(out['residuo'].mean()),
        'Std_residuo': float(out['residuo'].std(ddof=1)) if len(out) > 1 else np.nan,
        'Var_residuo': float(out['residuo'].var(ddof=1)) if len(out) > 1 else np.nan,
        'R_estimado': R_est,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return result, out

# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
def main():
    print('=' * 80)
    print('Estimativa de R com a nova arquitetura em espaço de estados')
    print('=' * 80)

    if not FILES:
        print('Nenhum arquivo CSV numérico foi encontrado no diretório.')
        return

    results_rows = []
    residuals_all = []

    for i, path in enumerate(FILES, start=1):
        file_name = os.path.basename(path)
        print(f'[{i:02d}/{len(FILES):02d}] Processando {file_name}...')

        try:
            result, out = process_one_file(path)
            results_rows.append(result)

            residuals_all.extend(out['residuo'].tolist())

            # salva detalhamento por arquivo
            detail_path = os.path.join(
                OUT_DIR,
                f'detalhe_{file_name}'
            )
            out.to_csv(detail_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f'Erro ao processar {file_name}: {e}')

    if not results_rows:
        print('Nenhum arquivo foi processado com sucesso.')
        return

    # -----------------------------------------------------
    # resultados por arquivo
    # -----------------------------------------------------
    results_df = pd.DataFrame(results_rows)

    results_csv = os.path.join(OUT_DIR, 'estimativa_R_por_arquivo.csv')
    results_df.to_csv(results_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    # -----------------------------------------------------
    # estimativa global de R
    # -----------------------------------------------------
    residuals_all = np.asarray(residuals_all, dtype=float)

    R_global = estimate_R_from_residual(residuals_all, ddof=1)

    summary = pd.DataFrame([{
        'Qtd_arquivos_processados': len(results_df),
        'Qtd_total_amostras': int(np.sum(results_df['N_amostras'])),
        'R_medio_dos_arquivos': float(results_df['R_estimado'].mean()),
        'R_mediano_dos_arquivos': float(results_df['R_estimado'].median()),
        'R_global_todos_residuos': R_global,
        'RMSE_medio': float(results_df['RMSE'].mean()),
        'MAE_medio': float(results_df['MAE'].mean()),
        'R2_medio': float(results_df['R2'].mean())
    }])

    summary_csv = os.path.join(OUT_DIR, 'resumo_estimativa_R.csv')
    summary.to_csv(summary_csv, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    print('\n' + '=' * 80)
    print('Concluído.')
    print(f'Resultados por arquivo: {results_csv}')
    print(f'Resumo global: {summary_csv}')
    print(f'Detalhes por arquivo salvos em: {OUT_DIR}')
    print('=' * 80)

    print('\nResumo:')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
