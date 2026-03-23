import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    gf[0] = gas[0] if gf0 is None else gf0

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

    out = df.copy()

    out['Gf'] = compute_gf(
        gas=out['GAS'].values,
        alpha=alpha,
        beta=beta,
        gf0=gf0
    )

    tl = out['TL'].to_numpy(dtype=float)
    tr = out['TR'].to_numpy(dtype=float)

    tl_est = np.zeros(len(out), dtype=float)
    tr_est = np.zeros(len(out), dtype=float)

    if len(out) > 0:
        tl_est[0] = tl[0]
        tr_est[0] = tr[0]

        for k in range(1, len(out)):
            tl_est[k] = tl[k - 1]
            tr_est[k] = tr[k - 1]

    out['TL_est'] = tl_est
    out['TR_est'] = tr_est

    out['Yhat'] = (
        c0
        + d_gf * out['Gf']
        + a_tl * out['TL_est']
        + b_tr * out['TR_est']
    )

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
# PROCESSAMENTO DE UM ARQUIVO
# =========================================================
def process_one_file(path: str):
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
    std_residuo = float(out['residuo'].std(ddof=1)) if len(out) > 1 else np.nan

    result = {
        'Arquivo': os.path.basename(path),
        'R': R_est,
        'Desvio_padrao': std_residuo
    }

    return result, out

# =========================================================
# HISTOGRAMA
# =========================================================
def plot_histograma_R(valores_R, out_path):
    valores_R = np.asarray(valores_R, dtype=float)
    valores_R = valores_R[~np.isnan(valores_R)]

    if len(valores_R) == 0:
        print('Não há valores válidos de R para plotar o histograma.')
        return

    media_R = np.mean(valores_R)
    desvio_R = np.std(valores_R, ddof=1) if len(valores_R) > 1 else 0.0
    limite_2sigma = media_R + 2 * desvio_R

    plt.figure(figsize=(10, 6))
    plt.hist(valores_R, bins=15, edgecolor='black', alpha=0.7)

    plt.axvline(media_R, color='blue', linestyle='--', linewidth=2, label=f'Média = {media_R:.4f}')
    plt.axvline(limite_2sigma, color='red', linestyle='--', linewidth=2,
                label=f'Média + 2σ = {limite_2sigma:.4f}')

    plt.xlabel('R')
    plt.ylabel('Frequência')
    plt.title('Histograma dos valores de R por arquivo')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# =========================================================
# EXECUÇÃO PRINCIPAL
# =========================================================
def main():
    print('=' * 80)
    print('Estimativa de R por arquivo')
    print('=' * 80)

    if not FILES:
        print('Nenhum arquivo CSV numérico foi encontrado no diretório.')
        return

    results_rows = []

    for i, path in enumerate(FILES, start=1):
        file_name = os.path.basename(path)
        print(f'[{i:02d}/{len(FILES):02d}] Processando {file_name}...')

        try:
            result, out = process_one_file(path)
            results_rows.append(result)

            detail_path = os.path.join(OUT_DIR, f'detalhe_{file_name}')
            out.to_csv(detail_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f'Erro ao processar {file_name}: {e}')

    if not results_rows:
        print('Nenhum arquivo foi processado com sucesso.')
        return

    # -----------------------------------------------------
    # tabela final por arquivo
    # -----------------------------------------------------
    results_df = pd.DataFrame(results_rows)

    tabela_path = os.path.join(OUT_DIR, 'tabela_R_desvio_por_arquivo.csv')
    results_df.to_csv(tabela_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    # -----------------------------------------------------
    # resumo final
    # -----------------------------------------------------
    R_medio = float(results_df['R'].mean())
    desvio_padrao_medio = float(results_df['Desvio_padrao'].mean())

    resumo_df = pd.DataFrame([{
        'Qtd_arquivos_processados': len(results_df),
        'R_medio': R_medio,
        'Desvio_padrao_medio': desvio_padrao_medio
    }])

    resumo_path = os.path.join(OUT_DIR, 'resumo_R_desvio.csv')
    resumo_df.to_csv(resumo_path, sep=';', decimal=',', index=False, encoding='utf-8-sig')

    # -----------------------------------------------------
    # histograma de R
    # -----------------------------------------------------
    hist_path = os.path.join(OUT_DIR, 'histograma_R.png')
    plot_histograma_R(results_df['R'].values, hist_path)

    print('\n' + '=' * 80)
    print('Concluído.')
    print(f'Tabela por arquivo: {tabela_path}')
    print(f'Resumo final: {resumo_path}')
    print(f'Histograma: {hist_path}')
    print('=' * 80)

    print('\nTabela por arquivo:')
    print(results_df.to_string(index=False))

    print('\nResumo final:')
    print(resumo_df.to_string(index=False))

if __name__ == '__main__':
    main()
