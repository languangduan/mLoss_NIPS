def format_accuracy(data):
    formatted_values = []
    for dataset, metrics in data.items():
        if isinstance(metrics, dict) and "top1" in metrics:
            formatted_values.append(f"{metrics['top1'] * 100:.2f}")
    
    if "average_accuracy" in data:
        formatted_values.append(f"{data['average_accuracy'] * 100:.2f}")
    
    return " & ".join(formatted_values)

# Example input
data = {
    "MNIST": {
        "top1": 0.9226
    },
    "DTD": {
        "top1": 0.527127659574468
    },
    "EuroSAT": {
        "top1": 0.7381481481481481
    },
    "GTSRB": {
        "top1": 0.6161520190023753
    },
    "SUN397": {
        "top1": 0.6523378235483426
    },
    "SVHN": {
        "top1": 0.731138598647818
    },
    "average_accuracy": 0.6979173748201921
}
# Output result
print(format_accuracy(data))