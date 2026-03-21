import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# COEFICIENTES IDENTIFICADOS
# =========================================================
alpha = 0.9945
c0 = 27.40
a_TL = 0.401
b_TR = 0.551
d_Gf = 0.094

# =========================================================
# MATRIZES ESPAÇO DE ESTADOS
# =========================================================
# Estado: x(k) = Gf(k)
# Entradas: u(k) = [GAS(k), TL(k), TR(k)]^T
# Saída: y(k) = TC estimada
A = np.array([[alpha]])
B = np.array([[1 - alpha, 0.0, 0.0]])   # [GAS, TL, TR]
C = np.array([[d_Gf]])
D = np.array([[0.0, a_TL, b_TR]])

# =========================================================
# EXIBIÇÃO DAS MATRIZES
# =========================================================
def mostrar_matrizes():
    np.set_printoptions(precision=6, suppress=True)

    print('\n' + '='*60)
    print('MODELO EM ESPAÇO DE ESTADOS')
    print('='*60)

    print('\nEquações do modelo:')
    print('x(k+1) = A x(k) + B u(k)')
    print('y(k)   = C x(k) + D u(k) + c0')

    print('\nDefinições:')
    print('x(k) = Gf(k)')
    print('u(k) = [ GAS(k)  TL(k)  TR(k) ]^T')
    print('y(k) = TC estimada')

    print('\nMatriz A:')
    print(A)

    print('\nMatriz B:')
    print(B)

    print('\nMatriz C:')
    print(C)

    print('\nMatriz D:')
    print(D)

    print('\nTermo constante:')
    print(f'c0 = {c0}')

    print('\nForma expandida:')
    print(f'x(k+1) = {alpha:.6f}·x(k) + {1-alpha:.6f}·GAS(k)')
    print(f'y(k)   = {d_Gf:.6f}·x(k) + {a_TL:.6f}·TL(k) + {b_TR:.6f}·TR(k) + {c0:.6f}')
    print('='*60 + '\n')

# =========================================================
# LEITURA DO CSV
# =========================================================
def read_one_csv(path):
    df = pd.read_csv(path, sep=';', decimal=',', engine='python')
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]

    df = df.rename(columns={
        'Intouch.U7_TT2421_UE': 'TL',
        'Intouch.U7_TT2425_UE': 'TR',
        'Intouch.U7_TT2423_UE': 'TC',
        'Intouch.FZIT_G55_2301_UE': 'GAS'
    })

    df = df[['TL', 'TR', 'TC', 'GAS']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# SOFT SENSOR ORIGINAL
# =========================================================
def softsensor(df):
    GAS = df['GAS'].values
    TL = df['TL'].values
    TR = df['TR'].values

    Gf = np.zeros(len(df))
    Gf[0] = GAS[0]

    for k in range(1, len(df)):
        Gf[k] = alpha * Gf[k - 1] + (1 - alpha) * GAS[k]

    yhat = c0 + a_TL * TL + b_TR * TR + d_Gf * Gf
    return yhat, Gf

# =========================================================
# MODELO EM ESPAÇO DE ESTADOS
# =========================================================
def state_space(df):
    GAS = df['GAS'].values
    TL = df['TL'].values
    TR = df['TR'].values

    x = np.zeros((1, len(df)))
    yhat = np.zeros(len(df))

    x[0, 0] = GAS[0]

    for k in range(1, len(df)):
        u = np.array([[GAS[k - 1]], [TL[k - 1]], [TR[k - 1]]])
        x[:, k] = (A @ x[:, [k - 1]] + B @ u).flatten()

    for k in range(len(df)):
        u = np.array([[GAS[k]], [TL[k]], [TR[k]]])
        yhat[k] = (C @ x[:, [k]] + D @ u).item() + c0

    return yhat, x.flatten()

# =========================================================
# CAMINHO DO ARQUIVO
# =========================================================
path_csv = r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo\01.csv'

# =========================================================
# EXECUÇÃO
# =========================================================
mostrar_matrizes()

df = read_one_csv(path_csv)

y_real = df['TC'].values
y_soft, gf = softsensor(df)
y_ss, x_ss = state_space(df)

residuo = y_soft - y_ss
dif_max = np.max(np.abs(residuo))

print('Diferença máxima entre modelos:', dif_max)

# =========================================================
# EIXO DO TEMPO
# =========================================================
t = np.arange(len(df))

# =========================================================
# PLOT
# =========================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(15, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# ---------------------------------------------------------
# GRÁFICO PRINCIPAL
# ---------------------------------------------------------
ax1.plot(
    t, y_real,
    color='blue',
    linewidth=1.8,
    label='TC real',
    zorder=1
)

ax1.plot(
    t, y_soft,
    color='black',
    linewidth=2.0,
    alpha=0.85,
    label='Softsensor direto',
    zorder=2
)

linha_ss, = ax1.plot(
    t, y_ss,
    color='red',
    linewidth=1.8,
    linestyle='--',
    marker='o',
    markersize=2.5,
    markevery=max(len(t) // 80, 1),
    alpha=0.95,
    label='Espaço de estados',
    zorder=3
)

# Tracejado mais espaçado
linha_ss.set_dashes([2, 20])

ax1.set_title('Comparação entre TC real e modelos do Softsensor')
ax1.set_ylabel('Temperatura')
ax1.grid(True, alpha=0.99)
ax1.legend(loc='best')

# ---------------------------------------------------------
# RESÍDUO
# ---------------------------------------------------------
ax2.plot(
    t, residuo,
    color='purple',
    linewidth=1.5,
    label='Resíduo entre modelos'
)

ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_ylim(-0.03, 0.03)

ax2.set_title('Resíduo entre os modelos')
ax2.set_xlabel('Amostra')
ax2.set_ylabel('Erro')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')

plt.tight_layout()
plt.show()
