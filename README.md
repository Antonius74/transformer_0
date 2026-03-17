# Spiegazione dell'Algoritmo Transformer per la Regressione

## 1. Introduzione e Teoria

### Cos'è un Transformer?
Il Transformer è un'architettura di rete neurale originariamente progettata per il Natural Language Processing, ma che può essere adattata per problemi di regressione come la previsione di prezzi immobiliari. La sua caratteristica principale è l'**attenzione multi-testa** (Multi-Head Attention), che permette al modello di "prestare attenzione" a diverse parti dei dati di input contemporaneamente.

### Come funziona in parole semplici
Immagina di dover prevedere il prezzo di una casa. Hai diverse informazioni: numero di stanze, età della casa, posizione, ecc. Un Transformer:
1. **Guarda tutte le informazioni contemporaneamente** (non in sequenza)
2. **Decide quali informazioni sono più importanti** tra loro
3. **Combina queste informazioni** in modo intelligente
4. **Fa la previsione** finale

Esempio pratico: se stai prevedendo il prezzo di una casa a San Francisco, il modello potrebbe "prestare più attenzione" alla posizione geografica che al numero di bagni, perché a San Francisco la posizione è molto più determinante per il prezzo.

### Componenti principali nel nostro codice

#### 1. Attenzione Multi-Testa (Multi-Head Attention)
Pensa a questo come a un gruppo di esperti che analizzano lo stesso problema da prospettive diverse. Ogni "testa" (head) cerca relazioni diverse tra le features.

#### 2. Normalizzazione dei Livelli (Layer Normalization)
È come standardizzare i dati ad ogni passo per mantenere l'apprendimento stabile.

#### 3. Rete Feed-Forward
Una rete neurale semplice che trasforma ulteriormente le informazioni dopo l'attenzione.

## 2. Dati Utilizzati (Input/Output)

### Il Dataset: California Housing
Il codice utilizza il dataset **California Housing** da `sklearn.datasets`. Questo dataset contiene informazioni sulle case in California del 1990.

### Caratteristiche dell'Input (8 features)
Ogni casa è descritta da 8 caratteristiche numeriche:

1. **MedInc** - Reddito medio dei residenti del blocco
2. **HouseAge** - Età media delle case del blocco
3. **AveRooms** - Numero medio di stanze per abitazione
4. **AveBedrms** - Numero medio di camere da letto per abitazione
5. **Population** - Popolazione del blocco
6. **AveOccup** - Occupazione media delle abitazioni
7. **Latitude** - Latitudine del blocco
8. **Longitude** - Longitudine del blocco

### Output Atteso (Target)
Il target è il **valore mediano delle case** nel blocco, espresso in centinaia di migliaia di dollari.

### Esempio Numerico
Ecco come potrebbero apparire i dati grezzi per un singolo blocco:

```
MedInc: 8.3252        (reddito medio)
HouseAge: 41.0        (anni)
AveRooms: 6.9841      (stanze)
AveBedrms: 1.0238     (camere da letto)
Population: 322.0     (persone)
AveOccup: 2.5556      (persone per casa)
Latitude: 37.88       (gradi)
Longitude: -122.23    (gradi)

Target: 4.526         (≈ $452,600)
```

### Dimensionalità dei Dati
Dopo il preprocessing:
- **Training set**: (16512, 8) - 16512 esempi con 8 features ciascuno
- **Test set**: (4128, 8) - 4128 esempi con 8 features ciascuno

## 3. Analisi del Codice

### Parte 1: Preprocessing dei Dati

```python
def scarica_e_preelabora_dati():
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
```

Il codice inizia scaricando il dataset. `X` contiene le 8 features, `y` contiene i prezzi delle case.

#### Divisione Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

L'80% dei dati va nel training set, il 20% nel test set. `random_state=42` garantisce riproducibilità.

#### Standardizzazione

```python
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
```

Ogni feature viene standardizzata per avere media 0 e deviazione standard 1:

$
x_{\text{standardizzato}} = \frac{x - \mu}{\sigma}
$

Dove $\mu$ è la media e $\sigma$ è la deviazione standard della feature.

### Parte 2: Architettura del Transformer

#### Blocco Encoder del Transformer

```python
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
```

Questa funzione crea un singolo blocco encoder del Transformer. Ecco cosa fa passo dopo passo:

**Passo 1: Normalizzazione e Attenzione**

```python
x = layers.LayerNormalization(epsilon=1e-6)(inputs)
x = layers.MultiHeadAttention(
    key_dim=head_size, num_heads=num_heads, dropout=dropout
)(x, x)
```

La Layer Normalization standardizza gli input:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Dove $\gamma$ e $\beta$ sono parametri apprendibili e $\epsilon=10^{-6}$ è una piccola costante per stabilità numerica.

L'attenzione multi-testa calcola per ogni testa:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Dove $Q$ (Query), $K$ (Key), e $V$ (Value) sono trasformazioni lineari dell'input.

**Passo 2: Connessione Residuale**

```python
x = layers.Dropout(dropout)(x)
res = x + inputs
```

L'input originale viene sommato all'output dell'attenzione (connessione residuale). Questo aiuta il flusso del gradiente durante l'addestramento.

**Passo 3: Rete Feed-Forward**

```python
x = layers.LayerNormalization(epsilon=1e-6)(res)
x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
x = layers.Dropout(dropout)(x)
x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
return x + res
```

Una rete feed-forward con due layer convoluzionali 1D (che agiscono come layer densi). La funzione di attivazione ReLU è:

$$
\text{ReLU}(x) = \max(0, x)
$$

### Parte 3: Costruzione del Modello Completo

```python
def costruisci_modello(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
```

**Parametri del nostro modello:**
- `head_size=64`: Dimensione di ogni testa di attenzione
- `num_heads=4`: 4 teste di attenzione parallele
- `ff_dim=128`: Dimensione del layer feed-forward interno
- `num_transformer_blocks=2`: 2 blocchi encoder in serie
- `mlp_units=[128]`: Un layer denso finale con 128 neuroni

**Struttura del modello:**
1. **Input**: (batch_size, 8, 1) - Le 8 features reshapeate
2. **2 Blocchi Transformer**: Ogni blocco applica attenzione e feed-forward
3. **Pooling Globale**: Prende la media lungo l'asse temporale
4. **MLP Finale**: Layer denso con 128 neuroni ReLU
5. **Output**: Singolo valore (prezzo predetto)

### Parte 4: Preparazione Dati per il Transformer

```python
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
```

Trasforma la forma da (samples, 8) a (samples, 8, 1). Il Transformer si aspetta una dimensione "temporale" o "sequenziale", anche se nel nostro caso non è una vera sequenza temporale.

### Parte 5: Addestramento del Modello

```python
modello.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["mae"],
)
```

**Funzione di Loss**: Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Optimizer**: Adam con learning rate di 0.001

**Early Stopping**: Ferma l'addestramento se la loss di validazione non migliora per 10 epoche consecutive.

### Parte 6: Valutazione e Metriche

```python
y_pred_scaled = modello.predict(X_test, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)
```

Le predizioni vengono trasformate dalla scala standardizzata alla scala originale.

**Metriche Calcolate:**

1. **Mean Squared Error (MSE)**:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

2. **Mean Absolute Error (MAE)**:
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

3. **Coefficiente di Determinazione (R²)**:
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

Dove $\bar{y}$ è la media dei valori reali.

### Parte 7: Flusso Completo del Programma

1. **Preprocessing**: Scarica, divide e standardizza i dati
2. **Preparazione**: Reshape per il Transformer
3. **Costruzione**: Crea l'architettura del modello
4. **Addestramento**: Allena con early stopping
5. **Valutazione**: Calcola MSE, MAE e R²
6. **Salvataggio**: Salva il modello addestrato

Il modello finale viene salvato come `transformer_housing.h5` e può essere caricato per fare predizioni su nuovi dati senza riaddestrare.