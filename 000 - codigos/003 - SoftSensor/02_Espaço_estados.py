import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# COEFICIENTES IDENTIFICADOS
# =========================================================
alpha = 0.995
c0 = -5.050626
a_TL = 0.445554
b_TR = 0.537991
d_Gf = 0.067165

# =========================================================
# MATRIZES ESPAÇO DE ESTADOS
# =========================================================
A = alpha
B = np.array([1-alpha, 0.0, 0.0])     # GAS, TL, TR
C = d_Gf
D = np.array([0.0, a_TL, b_TR])

# =========================================================
# LEITURA
# =========================================================
def read_one_csv(path):
    df = pd.read_csv(path, sep=';', decimal=',', engine='python')
    df.columns = [str(c).replace('\ufeff','').strip() for c in df.columns]

    df = df.rename(columns={
        'Intouch.U7_TT2421_UE': 'TL',
        'Intouch.U7_TT2425_UE': 'TR',
        'Intouch.U7_TT2423_UE': 'TC',
        'Intouch.FZIT_G55_2301_UE': 'GAS'
    })

    df = df[['TL','TR','TC','GAS']].apply(pd.to_numeric, errors='coerce')
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

    for k in range(1,len(df)):
        Gf[k] = alpha*Gf[k-1] + (1-alpha)*GAS[k]

    yhat = c0 + a_TL*TL + b_TR*TR + d_Gf*Gf
    return yhat, Gf

# =========================================================
# MODELO ESPAÇO DE ESTADOS MISO
# =========================================================
def state_space(df):
    GAS = df['GAS'].values
    TL = df['TL'].values
    TR = df['TR'].values

    x = np.zeros(len(df))
    yhat = np.zeros(len(df))

    x[0] = GAS[0]

    for k in range(1,len(df)):
        u = np.array([GAS[k-1], TL[k-1], TR[k-1]])
        x[k] = A*x[k-1] + B @ u

    for k in range(len(df)):
        u = np.array([GAS[k], TL[k], TR[k]])
        yhat[k] = C*x[k] + D @ u + c0

    return yhat, x

# =========================================================
# EXECUÇÃO
# =========================================================
df = read_one_csv(
    r'C:\Dados\Usina_7_Forno\003 - Dados CSV\01 - Dados Divididos 2h\00 - Cruzeiro\00 - Treinamento Modelo\01.csv'
)

y_real = df['TC'].values

y_soft, gf = softsensor(df)
y_ss, x_ss = state_space(df)

print('Diferença máxima entre modelos:',
      np.max(np.abs(y_soft - y_ss)))

# =========================================================
# PLOT
# =========================================================
t = np.arange(len(df))

plt.figure(figsize=(14,6))
plt.plot(t, y_real, label='TC real')
plt.plot(t, y_soft, label='Softsensor direto')
plt.plot(t, y_ss, '--', label='Espaço de estados')
plt.legend()
plt.grid()
plt.title('Comparação Softsensor vs Espaço de Estados')
plt.xlabel('Amostra (s)')
plt.ylabel('Temperatura')
plt.show()

plt.figure(figsize=(14,4))
plt.plot(t, y_soft - y_ss)
plt.title('Diferença entre modelos')
plt.grid()
plt.show()
