
import sys
import pandas as pd
from pathlib import Path
# requirements: scikit-learn, matplotlib, pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Make local package available without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from callmefair.util.models import TabTransformer

def evaluate_tabtransformer(df: pd.DataFrame,
                            target: str = "label",
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            random_state: int = 42,
                            config: dict = None):
    """
    Splits df -> trains TabTransformerClassifier and prints metrics on test set.
    Returns the trained classifier and dictionary with results.
    """
    # 1. train/val/test split (stratified if classification labels are discrete)
    y = df[target]
    # If y is continuous, don't stratify. We assume classification here:
    stratify_arg = y if (y.dtype == object or len(np.unique(y)) < 50 and y.dtype != float) else None

    # first split train_temp and test
    X_train_temp, X_test = train_test_split(
        df, test_size=test_size, stratify=stratify_arg, random_state=random_state)

    # compute validation proportion relative to train_temp
    if val_size > 0:
        val_frac_of_total = val_size / (1.0 - test_size)
        stratify_train = X_train_temp[target] if stratify_arg is not None else None
        X_train, X_val = train_test_split(
            X_train_temp, test_size=val_frac_of_total, stratify=stratify_train, random_state=random_state)
    else:
        X_train, X_val = X_train_temp, None

    # 2. Fit model
    clf = TabTransformer(config=config, target=target)
    # pass validation_data via fit kwargs if desired (here we pass validation_split=0 since we already split)
    fit_kwargs = {}
    if X_val is not None:
        # Keras expects validation_data in fit as a tf.data.Dataset or (inputs, y) tuple; simplest is to pass validation_split=0
        # We'll just fit on X_train and evaluate on X_val manually later
        pass

    print(f"Training on {len(X_train)} rows; validating on {len(X_val) if X_val is not None else 0}; testing on {len(X_test)}")

    # Fit on training dataframe
    clf.fit(X_train)

    # 3. Predict on test set
    # predict_proba returns either 1-d array (binary) or 2-d array (multiclass)
    probs = clf.predict_proba(X_test)
    y_true = X_test[target].values

    # handle binary vs multiclass
    results = {}
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        # binary
        probs_pos = probs.ravel()
        y_pred = (probs_pos >= 0.5).astype(int)

        # if original labels were strings, map them using label mapping if available:
        if getattr(clf, "_label_mapping", None) is not None:
            # label_mapping is list of classes encountered during fit
            classes = list(clf._label_mapping)
            # Attempt to convert y_true to ints
            try:
                y_true_int = pd.factorize(y_true)[0]
                y_true = y_true_int
            except Exception:
                pass

        results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        results['precision'] = metrics.precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = metrics.recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = metrics.f1_score(y_true, y_pred, zero_division=0)
        # ROC AUC
        try:
            results['roc_auc'] = metrics.roc_auc_score(y_true, probs_pos)
        except Exception as e:
            results['roc_auc'] = None

        print("Binary classification results:")
        print(metrics.classification_report(y_true, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(metrics.confusion_matrix(y_true, y_pred))

        # ROC curve
        fpr, tpr, _ = metrics.roc_curve(y_true, probs_pos)
        plt.figure()
        plt.plot(fpr, tpr)    # do NOT set color explicitly per tool policy
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve (test)")
        plt.grid(True)
        plt.show()

    else:
        # multiclass probabilities shape = (n_samples, n_classes)
        y_pred_idx = np.argmax(probs, axis=1)
        # if label mapping exists, map indices to labels
        if getattr(clf, "_label_mapping", None) is not None:
            classes = list(clf._label_mapping)
            y_pred = np.array(classes)[y_pred_idx]
            # try to map y_true to same label space; if y_true are strings, ok; if ints, convert them
            try:
                # if y_true are not strings, attempt to map with factorize
                if not np.issubdtype(y_true.dtype, np.str_):
                    y_true_fact, uniques = pd.factorize(y_true)
                    y_true = np.array(uniques)[y_true_fact]
            except Exception:
                pass
        else:
            y_pred = y_pred_idx

        results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        results['precision_macro'] = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)

        print("Multiclass classification results:")
        print(metrics.classification_report(y_true, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(metrics.confusion_matrix(y_true, y_pred))

        # multiclass ROC AUC (one-vs-rest)
        try:
            # binarize ground truth
            from sklearn.preprocessing import label_binarize
            classes_unique = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes_unique)
            # compute per-class ROC AUC and macro average
            roc_auc_per_class = {}
            for i, cls in enumerate(classes_unique):
                roc_auc_per_class[cls] = metrics.roc_auc_score(y_true_bin[:, i], probs[:, i])
            results['roc_auc_per_class'] = roc_auc_per_class
            results['roc_auc_macro'] = metrics.roc_auc_score(y_true_bin, probs, average='macro')
            # Plot one curve per class (if small number of classes)
            plt.figure()
            for i, cls in enumerate(classes_unique):
                fpr, tpr, _ = metrics.roc_curve(y_true_bin[:, i], probs[:, i])
                plt.plot(fpr, tpr, label=f"class {cls}")
            plt.plot([0,1],[0,1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Multiclass ROC curves (test)")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            results['roc_auc_per_class'] = None

    return clf, results

if __name__ == "__main__":
    # Load dataset
    
    CSV_FILE, TARGET_COLUMN = "../data/stroke_pre/stroke.csv", "stroke"

    # load a DataFrame or pass a csv path
    df = pd.read_csv(CSV_FILE)  # or your dataframe

    # Define TabTransformer config
    # Small: fast, memory-light — good for experimentation, small datasets, or quick CI checks.
    small_config = {
        "batch_size": 256,                # larger batch allowed because model is small
        "epochs": 20,
        "learning_rate": 1e-3,
        "transformer_layers": 2,
        "transformer_heads": 4,
        "transformer_embedding_dim": 32,
        "ff_dim": 64,                     # FF per-token hidden dim
        "dropout": 0.1,
        "mlp_units": [64, 32],            # small MLP head
        "mlp_dropout": 0.2,
        "dense_activation": "gelu",
        "loss": "binary_crossentropy",
        "metrics": ["AUC"],
        "shuffle": True,
        "validation_split": 0.2,
        "seed": 42,
        "max_embedding_cardinality": None
    }

    # Medium: balanced — good default for most datasets, respectable capacity without huge memory needs.
    medium_config = {
        "batch_size": 192,                # slightly smaller to balance GPU memory and throughput
        "epochs": 30,                     # a bit more training to make use of capacity
        "learning_rate": 8e-4,
        "transformer_layers": 4,
        "transformer_heads": 8,
        "transformer_embedding_dim": 64,
        "ff_dim": 128,
        "dropout": 0.12,
        "mlp_units": [256, 128],          # wider MLP head
        "mlp_dropout": 0.25,
        "dense_activation": "gelu",
        "loss": "binary_crossentropy",
        "metrics": ["AUC"],
        "shuffle": True,
        "validation_split": 0.2,
        "seed": 42,
        "max_embedding_cardinality": None
    }

    # Large: high-capacity model — use only with plenty of GPU memory / larger datasets.
    # Consider gradient accumulation, mixed-precision, and reduced batch_size if memory is tight.
    large_config = {
        "batch_size": 128,                 # reduce batch to keep memory usage reasonable
        "epochs": 40,
        "learning_rate": 5e-4,            # smaller LR for stability on larger models
        "transformer_layers": 12,
        "transformer_heads": 16,
        "transformer_embedding_dim": 128,
        "ff_dim": 256,
        "dropout": 0.15,
        "mlp_units": [512, 256, 128],     # deep MLP head
        "mlp_dropout": 0.3,
        "dense_activation": "gelu",
        "loss": "binary_crossentropy",
        "metrics": ["AUC"],
        "shuffle": True,
        "validation_split": 0.2,
        "seed": 42,
        "max_embedding_cardinality": None
    }

    # Evaluate TabTransformer
    clf, results = evaluate_tabtransformer(df, target=TARGET_COLUMN, config=large_config)
    print("Evaluation results:", results)