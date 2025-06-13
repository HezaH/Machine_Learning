import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

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

# Dicionário base com a ordem desejada
choices = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}#, "EXTREME": 4}

print(f"Categorias: {choices.keys()}")
# Aplica a transformação em cada DataFrame
df_train, new_choices_train = change_cathegory(df_train, choices)
df_val, new_choices_val   = change_cathegory(df_val, choices)
df_test, new_choices_test  = change_cathegory(df_test, choices)

# Define as features para os modelos: remove as colunas derivadas do target
n_threshold = len(new_choices_train) - 1  # k - 1, onde k é o número de categorias presentes
exclude_cols = [target, "categoria", "cat_code"] + [f'T_{i+1}' for i in range(n_threshold)]
features = [col for col in df_train.columns if col not in exclude_cols]

###########################
# Modelo Ordinal
###########################
ordinal_classifiers = []  # Lista para armazenar os modelos treinados
print("== Treinamento do Modelo Ordinal ==")
for i in range(1, n_threshold + 1):
    clf = GradientBoostingClassifier(random_state=42)
    # Aqui, cada target é a coluna T_i (0 ou 1)
    print(f"Treinando modelo para T_{i}, equiparado a '{list(choices.keys())[i-1]}'")
    clf.fit(df_train[features], df_train[f'T_{i}'])
    ordinal_classifiers.append(clf)

print(f"Modelos treinados: {len(ordinal_classifiers)}")

# Previsão no conjunto de teste usando o modelo ordinal
ord_pred_labels, ord_pred_numbers = predict_ordinal(df_test, ordinal_classifiers, features)

print("== Modelo Ordinal ==")
print("Matriz de Confusão (Ordinal):")
print(confusion_matrix(df_test["categoria"], ord_pred_labels))
print("\nClassification Report (Ordinal):")
print(classification_report(df_test["categoria"], ord_pred_labels))

###########################
# Modelo Nominal (Tradicional)
###########################
print("== Treinamento do Modelo Nominal ==")
# Tratando as classes como nominais, usamos a coluna "categoria" e um LabelEncoder.
le = LabelEncoder()
df_train["cat_nominal"] = le.fit_transform(df_train["categoria"])
df_val["cat_nominal"]   = le.transform(df_val["categoria"])
df_test["cat_nominal"]  = le.transform(df_test["categoria"])

nominal_clf = GradientBoostingClassifier(random_state=42)
nominal_clf.fit(df_train[features], df_train["cat_nominal"])
print("Modelo nominal treinado.")
nom_pred = nominal_clf.predict(df_test[features])
nom_pred_labels = le.inverse_transform(nom_pred)

print("== Modelo Nominal ==")
print("Matriz de Confusão (Nominal):")
print(confusion_matrix(df_test["categoria"], nom_pred_labels))
print("\nClassification Report (Nominal):")
print(classification_report(df_test["categoria"], nom_pred_labels))

# Tempo total de execução
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
