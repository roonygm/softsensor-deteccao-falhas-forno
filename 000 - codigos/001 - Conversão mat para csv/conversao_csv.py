import scipy.io
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import re

# ===============================
# CONFIGURAÇÕES
# ===============================
pasta = r"C:\Dados\Usina_7_Forno\002 - Coleta\COLETA_2026-01-19"
saida_csv = os.path.join(pasta, "coleta_18.csv")

arquivos_mat = sorted([f for f in os.listdir(pasta) if f.endswith(".mat")])

dfs = []

# ===============================
# PROCESSAR ARQUIVOS
# ===============================
for arquivo_mat in arquivos_mat:
    caminho_mat = os.path.join(pasta, arquivo_mat)
    mat = scipy.io.loadmat(caminho_mat)
    
    variaveis = {k: v for k, v in mat.items() if not k.startswith("__")}
    nomes_variaveis = list(variaveis.keys())
    
    tags_raw = variaveis[nomes_variaveis[0]]
    dados_raw = variaveis[nomes_variaveis[1]]
    
    tags_flat = np.array(tags_raw).flatten()
    tags = []
    for tag in tags_flat:
        if isinstance(tag, np.ndarray):
            tags.append(str(tag[0]))
        else:
            tags.append(str(tag))
    
    if dados_raw.shape[0] > dados_raw.shape[1]:
        dados = dados_raw
    else:
        dados = dados_raw[1:, :]
    
    df = pd.DataFrame(dados, columns=tags)
    
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", arquivo_mat)
    data_inicial = datetime.strptime(match.group(0), "%Y-%m-%d %H-%M-%S")
    
    timestamps = [data_inicial + timedelta(seconds=i) for i in range(len(df))]
    df.insert(0, "Timestamp", timestamps)
    
    dfs.append(df)

# ===============================
# CONCATENAR E SALVAR CSV
# ===============================
df_final = pd.concat(dfs, ignore_index=True)

df_final.to_csv(
    saida_csv,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("CSV gerado com sucesso em:")
print(saida_csv)
