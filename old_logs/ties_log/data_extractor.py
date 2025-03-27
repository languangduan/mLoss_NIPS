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
        "top1": 0.988
    },
    "DTD": {
        "top1": 0.5824468085106383
    },
    "EuroSAT": {
        "top1": 0.7155555555555555
    },
    "GTSRB": {
        "top1": 0.7798099762470309
    },
    "SUN397": {
        "top1": 0.6273734540940646
    },
    "SVHN": {
        "top1": 0.9094575906576521
    },
    "average_accuracy": 0.7671072308441569
}

# Output result
print(format_accuracy(data))
