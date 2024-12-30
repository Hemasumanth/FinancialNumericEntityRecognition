from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np
import mlflow

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np

class F1ScoreCallback(Callback):
    def __init__(self, model, validation_data, average='micro'):
        self.model = model
        self.validation_data = validation_data
        self.average = average

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0])
        val_pred_flat = np.argmax(val_pred, axis=-1).flatten()
        val_true_flat = self.validation_data[1].flatten()

        f1_micro = f1_score(val_true_flat, val_pred_flat, average='micro')
        f1_macro = f1_score(val_true_flat, val_pred_flat, average='macro')
        f1_weighted = f1_score(val_true_flat, val_pred_flat, average='weighted')

        print(f"\nF1 Score (Micro): {f1_micro}")
        print(f"F1 Score (Macro): {f1_macro}")
        print(f"F1 Score (Weighted): {f1_weighted}")

        mlflow.log_metrics({
                    'f1_micro': f1_micro,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted
                }, step=epoch)  