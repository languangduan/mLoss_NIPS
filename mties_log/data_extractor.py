DATA = {
    "MNIST": {
        "top1": 0.9882
    },
    "DTD": {
        "top1": 0.5792553191489361
    },
    "EuroSAT": {
        "top1": 0.7527777777777778
    },
    "GTSRB": {
        "top1": 0.7705463182897863
    },
    "SUN397": {
        "top1": 0.616063629258425
    },
    "SVHN": {
        "top1": 0.9048478795328826
    },
    "average_accuracy": 0.7686151540013012
}






def format_accuracy(data):
    formatted_values = []
    for dataset, metrics in data.items():
        if isinstance(metrics, dict) and "top1" in metrics:
            formatted_values.append(f"{metrics['top1'] * 100:.2f}")
    
    if "average_accuracy" in data:
        formatted_values.append(f"{data['average_accuracy'] * 100:.2f}")
    if "avg_accuracy" in data:
        formatted_values.append(f"{data['avg_accuracy'] * 100:.2f}")
    
    return " & ".join(formatted_values)

# Example input
data = DATA

# Output result
print(format_accuracy(data))