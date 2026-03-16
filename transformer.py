import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def scarica_e_preelabora_dati():
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    return (X_train_scaled, X_test_scaled, 
            y_train_scaled, y_test_scaled,
            scaler_y, dataset.feature_names)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

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
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

def addestra_valuta_modello(X_train, X_test, y_train, y_test, scaler_y):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    input_shape = X_train.shape[1:]
    
    modello = costruisci_modello(
        input_shape=input_shape,
        head_size=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
    )
    
    modello.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["mae"],
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    
    storia = modello.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )
    
    y_pred_scaled = modello.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_orig = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return modello, storia, (mse, mae, r2)

def main():
    (X_train, X_test, 
     y_train, y_test, 
     scaler_y, feature_names) = scarica_e_preelabora_dati()
    
    print(f"Dimensionalità training set: {X_train.shape}")
    print(f"Dimensionalità test set: {X_test.shape}")
    print(f"Nomi features: {feature_names}")
    
    modello, storia, metriche = addestra_valuta_modello(
        X_train, X_test, y_train, y_test, scaler_y
    )
    
    modello.save("transformer_housing.h5")

if __name__ == "__main__":
    main()