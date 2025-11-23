from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout_rate: float = 0.3):
    """
    Create model with configurable dropout for regularization.
    Higher dropout helps prevent overfitting on small datasets.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    # Set dropout in the classifier head
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'dropout'):
        model.classifier.dropout.p = dropout_rate
    # Also set dropout in the base model if available
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'dropout'):
        model.distilbert.dropout.p = dropout_rate
    return model
