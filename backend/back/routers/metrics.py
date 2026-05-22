from fastapi import APIRouter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

router = APIRouter()

@router.post("/api/v1/evaluate")
async def evaluate_models(data: dict):
    # data = {"y_true": [0,1,1,0], "y_pred_probs": [0.1, 0.9, 0.8, 0.2]}
    y_true = data["y_true"]
    y_probs = data["y_pred_probs"]
    y_pred = [1 if p > 0.5 else 0 for p in y_probs]
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
