import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
def show_confusion( y_true, y_pred, labels, name_fig , title="Matriz de Confusão", plot_heatmap=True, cmap="Blues", figsize=(6, 5), ):
    """
    Exibe a matriz de confusão com rótulos nas linhas (reais)
    e colunas (preditos) na ordem especificada por `labels`.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos reais.
    y_pred : array-like
        Rótulos preditos.
    labels : list
        Ordem desejada das classes (ex.: ['NONE', 'WEAK', 'MODERATE', 'STRONG']).
    title : str
        Título mostrado antes da tabela / heatmap.
    plot_heatmap : bool
        Se True, desenha também um heatmap usando seaborn.
    cmap : str
        Mapa de cores para o heatmap.
    figsize : tuple
        Tamanho da figura do heatmap.
    """
    # 1. Matriz numérica
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 2. DataFrame bonito para print
    df_cm = pd.DataFrame(
        cm,
        index=[f"REAL → {lbl}" for lbl in labels],
        columns=[f"PRED → {lbl}" for lbl in labels],
    )

    # 3. Print “cru” no console
    print(f"\n{title}")
    print(df_cm)

    # 4. Heatmap opcional
    if plot_heatmap:
        plt.figure(figsize=figsize)
        sns.heatmap(
            df_cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="0.9",
        )
        plt.title(title, fontweight="bold")
        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")
        plt.tight_layout()
        plt.savefig(name_fig, dpi=300, bbox_inches='tight')
        plt.close()

def predict_ordinal(df, classifiers, features):
    preds = []
    for clf in classifiers:
        # Supondo que você queira as probabilidades da classe 1 para cada limiar
        pred = clf.predict_proba(df[features])[:, 1]
        preds.append(pred)
    
    preds = np.array(preds)  # shape: (n_threshold, n_samples)
    # Calcula as probabilidades das k classes:
    # Para a primeira classe:
    prob_class0 = 1 - preds[0]
    prob_classes = [prob_class0]
    
    # Para as classes intermediárias:
    for i in range(1, preds.shape[0]):
        # A probabilidade da classe i é a diferença entre a probabilidade da classe i-1 e a classe i
        #preds[i-1] = probabilidade da classe i-1
        #preds[i] = probabilidade da classe i
        prob = preds[i-1] - preds[i]
        prob_classes.append(prob)
    
    # Para a última classe:
    prob_class_last = preds[-1]
    prob_classes.append(prob_class_last)
    
    # Transforma em array: shape (k, n_samples)
    prob_classes = np.array(prob_classes)
    
    # Predição final: índice da classe com maior probabilidade
    pred_idx = np.argmax(prob_classes, axis=0)
    
    # Cria um dicionário inverso para mapear o código ordinal para a categoria
    mapping_inv = {v: k for k, v in choices.items()}
    pred_labels = [mapping_inv[i] for i in pred_idx]
    return pred_labels, prob_classes

def change_cathegory(df, choices):
    """
    Transforma a variável 'target' em uma classificação ordinal:
      - Cria a coluna "categoria", baseada em condições do valor contínuo;
      - Mapeia para "cat_code" usando o dicionário de escolhas, mas apenas com os rótulos presentes no dataframe;
      - Cria as colunas binárias T_1, T_2, ... T_(k-1), onde k é o número de categorias observadas.
    """
    # Definir as condições para os intervalos
    conditions = [
        (df["target"] == 0),
        ((df["target"] > 0) & (df["target"] <= 5)),
        ((df["target"] > 5) & (df["target"] <= 25)),
        (df["target"] > 25)
        # ((df["target"] > 25) & (df["target"] <= 50)),
        # (df["target"] > 50)
    ]
    
    # Aplica o np.select usando todas as categorias do dicionário base
    df["categoria"] = np.select(conditions, list(choices.keys()), default=np.nan)
    
    # Filtra as categorias que de fato aparecem no dataframe, mantendo a ordem definida em choices
    present_cats = [cat for cat in choices.keys() if (df["categoria"] == cat).any()]
    
    # Cria um novo mapeamento somente para as categorias presentes
    new_choices = {cat: i for i, cat in enumerate(present_cats)}
    
    # Mapeia os rótulos que foram de fato usados para códigos numéricos
    df["cat_code"] = df["categoria"].map(new_choices)
    
    # Número de colunas binárias a serem criadas (k - 1), onde k é o número de categorias presentes
    n = len(new_choices) - 1  
    for i in range(n):
        df[f'T_{i+1}'] = (df["cat_code"] > i).astype(int)
        
    return df, new_choices

# Inicia a medição do tempo
start_time = time.time()
out_dir = os.path.dirname(os.path.realpath(__file__))

# Carrega os dados a partir do arquivo pickle
current_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'A652.pickle')
with open(current_dir, 'rb') as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

print(f"Shapes: {X_train.shape}, {X_test.shape}, {X_val.shape}")

target = 'target'
scoring_type = 'accuracy'  # ou 'f1', 'precision', 'recall'

# Converte os arrays para DataFrames: concatena as features e o target
df_train = pd.concat([
    pd.DataFrame(X_train),
    pd.Series(y_train.ravel(), name=target)
], axis=1).reset_index(drop=True)

df_val = pd.concat([
    pd.DataFrame(X_val),
    pd.Series(y_val.ravel(), name=target)
], axis=1).reset_index(drop=True)

df_test = pd.concat([
    pd.DataFrame(X_test),
    pd.Series(y_test.ravel(), name=target)
], axis=1).reset_index(drop=True)

# Dicionário base com a ordem desejada para o target
choices = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}
label_order  = list(choices.keys())# ['NONE','WEAK','MODERATE','STRONG']
print(f"Categorias: {choices.keys()}")

# Aplica a transformação em cada DataFrame (gera também o mapeamento dinâmico new_choices)
df_train, new_choices_train = change_cathegory(df_train, choices)
df_val, new_choices_val   = change_cathegory(df_val, choices)
df_test, new_choices_test = change_cathegory(df_test, choices)

# Define as features para os modelos: removendo as colunas derivadas do target de forma dinâmica
n_threshold = len(new_choices_train) - 1  # k - 1, onde k é o número de categorias presentes
exclude_cols = [target, "categoria", "cat_code"] + [f'T_{i+1}' for i in range(n_threshold)]
features = [col for col in df_train.columns if col not in exclude_cols]

print(f"Total de features originais usadas: {len(features)}")

# ----------------------------
# Seleção do melhor k via PCA
# ----------------------------

# Defina um range de candidatos para k. Por exemplo, testar de 2 a |features| componentes (ou um range menor)
k_candidates = list(range(2, min(20, len(features)) + 1))  # usando até 20 ou o número máximo disponível

best_k = None
best_val_score = -np.inf  # ou 0
results = {}

# Usaremos o conjunto nominal para avaliação (pode ser adaptado para o ordinal também)
# Transformando a coluna target nominal via LabelEncoder:
le = LabelEncoder()
df_train["cat_nominal"] = le.fit_transform(df_train["categoria"])
df_val["cat_nominal"] = le.transform(df_val["categoria"])

for k in k_candidates:
    # Aplica PCA nos dados de treinamento e validação (apenas nas features)
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(df_train[features])
    X_val_pca   = pca.transform(df_val[features])
    
    # Treina um modelo nominal (GradientBoostingClassifier) nos dados transformados
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train_pca, df_train["cat_nominal"])
    
    # Faz predição no conjunto de validação e calcula a acurácia
    y_val_pred = clf.predict(X_val_pca)
    score = accuracy_score(df_val["cat_nominal"], y_val_pred)
    results[k] = score
    print(f"k = {k}, Acurácia de Validação = {score:.4f}")
    
    if score >= best_val_score:
        best_val_score = score
        best_k = k

print(f"\nMelhor k encontrado: {best_k} com acurácia de validação = {best_val_score:.4f}")

ks = list(results.keys())
accuracies = list(results.values())

# Plota um gráfico de linha com marcadores
plt.plot(ks, accuracies, marker='o')

plt.xlabel("Número de Componentes (k)")
plt.ylabel("Acurácia")
plt.title("Desempenho (Acurácia) em função do número de Componentes (k)")
plt.grid(True)
plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "figures", 'accuracy_k.png'))

# ----------------------------
# Criação do conjunto D2 utilizando o melhor k
# ----------------------------
pca_final = PCA(n_components=best_k)
X_train_pca_final = pca_final.fit_transform(df_train[features])
X_val_pca_final   = pca_final.transform(df_val[features])
X_test_pca_final  = pca_final.transform(df_test[features])

# Agora D2 consiste nos dataframes originais transformados via PCA
# Você pode criar DataFrames para facilitação, se necessário:
df_train_pca = pd.DataFrame(X_train_pca_final, columns=[f"PC{i+1}" for i in range(best_k)])
df_val_pca   = pd.DataFrame(X_val_pca_final, columns=[f"PC{i+1}" for i in range(best_k)])
df_test_pca  = pd.DataFrame(X_test_pca_final, columns=[f"PC{i+1}" for i in range(best_k)])

# ----------------------------
# Ajusta dois modelos: um sobre D1 (conjunto original) e outro sobre D2 (dados reduzidos via PCA)
# ----------------------------
print("== Treinamento do Modelo Nominal sobre D1 (original) ==")
clf_D1 = GradientBoostingClassifier(random_state=42)
clf_D1.fit(df_train[features], df_train["cat_nominal"])
pred_D1 = clf_D1.predict(df_test[features])
pred_D1_labels = le.inverse_transform(pred_D1)

show_confusion(df_test["categoria"], pred_D1_labels, name_fig = os.path.join(out_dir, "figures", "confusion_matrix_d1.png"), labels=label_order, title="Matriz de Confusão (D1)" )
print("\nClassification Report (D1):")
print(classification_report(df_test["categoria"], pred_D1_labels, labels=label_order))

print("\n== Treinamento do Modelo Nominal sobre D2 (PCA) ==")
clf_D2 = GradientBoostingClassifier(random_state=42)
clf_D2.fit(X_train_pca_final, df_train["cat_nominal"])
pred_D2 = clf_D2.predict(X_test_pca_final)
pred_D2_labels = le.inverse_transform(pred_D2)

show_confusion(df_test["categoria"], pred_D2_labels, name_fig = os.path.join(out_dir, "figures", "confusion_matrix_d2.png"), labels=label_order, title="Matriz de Confusão (D2)" )
print("\nClassification Report (D2):")
print(classification_report(df_test["categoria"], pred_D2_labels, labels=label_order))

# Tempo total de execução
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
