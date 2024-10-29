import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer."""
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer, device

def prepare_prompt(input_data):
    """Prepare the prompt for the model."""
    alpaca_prompt = """### Instruction:
You are a cyber security assistant, your job is to check for suspicious packages that i will send to you, tell me what kind of package you are seeing and check if the package is dangerous or not.

### Input:
Analyze the network traffic pattern for suspicious behavior:

{input}

### Response:
"""
    return alpaca_prompt.format(input=json.dumps(input_data, indent=2))

def get_model_prediction(model, tokenizer, prompt, device):
    """Get prediction from the model."""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        response_text = response.split("### Response:")[-1].strip()
        response_dict = json.loads(response_text)
        return response_dict.get("decision", "NORMAL")
    except:
        # Fallback: Check if response contains "SUSPICIOUS"
        return "SUSPICIOUS" if "SUSPICIOUS" in response.upper() else "NORMAL"

def create_confusion_matrix(dataset_path, model_path):
    """Create and visualize confusion matrix from dataset."""
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    true_labels = []
    predicted_labels = []
    responses = 1
    
    for data in dataset:
        # Get true label from dataset
        true_label = data["output"]["decision"]
        true_labels.append(true_label)
        
        # Get model prediction
        prompt = prepare_prompt(data["input"])
        predicted_label = get_model_prediction(model, tokenizer, prompt, device)
        predicted_labels.append(predicted_label)
        print("response: ", responses)
        responses +=1
    
    labels = ["NORMAL", "SUSPICIOUS"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    report = classification_report(true_labels, predicted_labels, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    return cm, report, plt

def evaluate_model(dataset_path, model_path):
    """Main function to evaluate the model."""
    cm, report, plt_figure = create_confusion_matrix(dataset_path, model_path)
    
    print("\nClassification Report:")
    print(report)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nAdditional Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plt_figure.show()
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

# Example usage
if __name__ == "__main__":
    dataset_path = "C:/tareas/LLM/hackathon/validation.json"
    model_path = "C:/tareas/models/trained_model"
    results = evaluate_model(dataset_path, model_path)