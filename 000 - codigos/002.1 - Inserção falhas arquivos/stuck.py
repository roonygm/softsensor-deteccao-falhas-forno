from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURAÇÕES
# =========================
PASTA_ORIGEM = Path(r"C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\00 - Dados originais")
PASTA_DESTINO = Path(r"C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\01 - Testes\03 - Falha tipo congelamento")

N_AMOSTRAS_FALHA = 2000
SALVAR_GRAFICOS = True

# Quantas amostras antes do início da falha você quer mostrar no sinal com falha
AMOSTRAS_ANTES_FALHA_PLOT = 10

# =========================
# LEITURA
# =========================
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
    return df

# =========================
# FALHA DE CONGELAMENTO
# =========================
def aplicar_falha_congelamento_tc(df: pd.DataFrame, n_amostras: int = 2000):
    df_mod = df.copy()

    if 'TC' not in df_mod.columns:
        raise ValueError("A coluna 'TC' não foi encontrada no arquivo.")

    n_total = len(df_mod)
    n_falha = min(n_amostras, n_total)
    idx_inicio = n_total - n_falha

    tc_original = df_mod['TC'].copy()

    # Usa o último valor antes da falha como valor congelado
    if idx_inicio > 0:
        valor_congelado = df_mod.iloc[idx_inicio - 1]['TC']
    else:
        valor_congelado = df_mod.iloc[0]['TC']

    df_mod.loc[df_mod.index[idx_inicio:], 'TC'] = valor_congelado

    return df_mod, tc_original, idx_inicio, valor_congelado

# =========================
# PLOT
# =========================
def plot_tc_comparacao(
    nome_arquivo,
    tc_original,
    tc_modificado,
    idx_inicio_falha,
    pasta_destino=None,
    amostras_antes_falha_plot=0
):
    plt.figure(figsize=(14, 6))

    # Fundo vermelho claro na região de falha
    plt.axvspan(
        idx_inicio_falha,
        len(tc_original) - 1,
        color='red',
        alpha=0.10,
        label='Região de falha'
    )

    # TC original no gráfico inteiro (mantém a cor azul padrão)
    plt.plot(
        tc_original.values,
        label='TC original',
        linewidth=1.8
    )

    # TC congelado aparece a partir de algumas amostras antes da falha
    inicio_plot_falha = max(0, idx_inicio_falha - amostras_antes_falha_plot)

    plt.plot(
        range(inicio_plot_falha, len(tc_modificado)),
        tc_modificado.iloc[inicio_plot_falha:].values,
        label='TC congelado',
        linewidth=1.8
    )



    plt.title(f'Comparação TC - {nome_arquivo}')
    plt.xlabel('Amostra')
    plt.ylabel('TC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if pasta_destino is not None:
        nome_png = Path(nome_arquivo).stem + "_TC_congelamento.png"
        plt.savefig(pasta_destino / nome_png, dpi=150)

    plt.show()

# =========================
# PROCESSAMENTO
# =========================
def main():
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)
    arquivos_processados = 0

    for i in range(1, 51):
        nome_arquivo = f"{i:02d}.csv"
        caminho_origem = PASTA_ORIGEM / nome_arquivo

        if not caminho_origem.exists():
            print(f"[AVISO] Arquivo não encontrado: {caminho_origem}")
            continue

        try:
            df = read_one_csv(str(caminho_origem))

            df_mod, tc_original, idx_inicio_falha, valor_congelado = aplicar_falha_congelamento_tc(
                df,
                n_amostras=N_AMOSTRAS_FALHA
            )

            caminho_destino = PASTA_DESTINO / nome_arquivo
            df_mod.to_csv(caminho_destino, sep=';', decimal=',', index=False, encoding='utf-8-sig')

            plot_tc_comparacao(
                nome_arquivo=nome_arquivo,
                tc_original=tc_original,
                tc_modificado=df_mod['TC'],
                idx_inicio_falha=idx_inicio_falha,
                pasta_destino=PASTA_DESTINO if SALVAR_GRAFICOS else None,
                amostras_antes_falha_plot=AMOSTRAS_ANTES_FALHA_PLOT
            )

            arquivos_processados += 1
            print(f"[OK] {nome_arquivo} processado | valor congelado = {valor_congelado}")

        except Exception as e:
            print(f"[ERRO] {nome_arquivo}: {e}")

    print(f"\nTotal de arquivos processados: {arquivos_processados}")

if __name__ == "__main__":
    main()
