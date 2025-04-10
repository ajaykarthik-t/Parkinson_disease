import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(file_path='parkinsons.csv'):
    """Load and preprocess the Parkinson's dataset."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop the name column as it's not a feature
    df = df.drop('name', axis=1)
    
    # Print class distribution to verify imbalance
    print("Class distribution:")
    print(df['status'].value_counts())
    
    # Extract features and target
    X = df.drop('status', axis=1).values
    y = df['status'].values
    
    # Normalize the features using MinMaxScaler (often works better for neural networks)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reshape input data for LSTM [samples, time steps, features]
    # We'll use a time window approach to create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length=3)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length=3)
    
    print(f"Training data shape after sequencing: {X_train_seq.shape}")
    print(f"Testing data shape after sequencing: {X_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler

def create_sequences(X, y, seq_length=3):
    """Create sequences for LSTM by repeating each sample to form a time series."""
    # For each data point, we'll create a sequence by repeating it
    X_seq = np.array([np.tile(x_i, (seq_length, 1)) for x_i in X])
    
    # Keep labels as they are, one per sequence
    y_seq = y
    
    return X_seq, y_seq

def create_lstm_model(input_shape):
    """Create an improved LSTM model for classification."""
    model = Sequential()
    
    # First Bidirectional LSTM layer with L2 regularization
    model.add(Bidirectional(LSTM(
        units=128, 
        return_sequences=True,
        kernel_regularizer=l2(0.001)
    ), input_shape=input_shape))  # Fixed: input_shape moved to Bidirectional wrapper
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(
        units=64, 
        return_sequences=False,
        kernel_regularizer=l2(0.001)
    )))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # First Dense layer
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Print model summary - this is now safe after proper model build
    model.summary()
    
    return model

def train_model(X_train, y_train, X_test, y_test, epochs=200, batch_size=16):
    """Train the LSTM model with improved techniques."""
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    
    # Compute class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)
    
    # Create directory for model checkpoints if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # We'll use k-fold cross validation for better generalization
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold = 1
    val_scores = []
    best_model = None
    best_score = 0
    best_history = None
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        print(f"\nTraining fold {fold}/{n_splits}")
        
        # Split data
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Create model
        model = create_lstm_model(input_shape)
        
        # Callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=30,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.00001
        )
        
        checkpoint = ModelCheckpoint(
            f'models/lstm_parkinsons_fold{fold}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, checkpoint, reduce_lr],
            class_weight=class_weight_dict,
            verbose=1
        )
        training_time = time.time() - start_time
        print(f"Training time for fold {fold}: {training_time:.2f} seconds")
        
        # Evaluate on validation set
        val_loss, val_acc, val_auc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"Fold {fold} - Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_score:
            best_score = val_auc
            best_model = model
            best_history = history
        
        val_scores.append(val_auc)
        fold += 1
    
    print(f"\nAverage validation AUC across folds: {np.mean(val_scores):.4f}")
    
    # Save the best model
    best_model.save('models/lstm_parkinsons_best_model.h5')
    
    # Final evaluation on test set
    test_loss, test_acc, test_auc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Best model - Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    return best_model, best_history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model with comprehensive metrics."""
    # Evaluate on test set
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Predictions with different threshold tuning
    y_pred_prob = model.predict(X_test).ravel()
    
    # Find best threshold for F1 score
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.7, 0.05):
        y_pred_temp = (y_pred_prob >= threshold).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred_temp)
        print(f"Threshold: {threshold:.2f}, F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
    
    # Final predictions with best threshold
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    
    # Calculate detailed metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC ROC: {auc:.4f}")
    
    # Classification report with zero_division parameter
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_prob)
    
    return y_pred, y_pred_prob

def plot_roc_curve(y_true, y_pred_prob):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

def plot_training_history(history):
    """Plot training and validation metrics."""
    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot training & validation accuracy
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation AUC
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()

def main():
    """Main function to run the training process."""
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    print("\nTraining LSTM model...")
    model, history = train_model(X_train, y_train, X_test, y_test, epochs=200, batch_size=16)
    
    print("\nEvaluating model...")
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
    
    print("\nPlotting training history...")
    if history is not None:
        plot_training_history(history)
    
    print("\nTraining complete!")
    print("Best model saved as: models/lstm_parkinsons_best_model.h5")

if __name__ == "__main__":
    main()