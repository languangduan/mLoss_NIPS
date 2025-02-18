DATA = {
    "MNIST": {
        "top1": 0.9843
    },
    "DTD": {
        "top1": 0.5617021276595745
    },
    "EuroSAT": {
        "top1": 0.7179629629629629
    },
    "GTSRB": {
        "top1": 0.7802850356294537
    },
    "SUN397": {
        "top1": 0.6181324996551882
    },
    "SVHN": {
        "top1": 0.8864474492931776
    },
    "average_accuracy": 0.7581383458667261
}

def format_accuracy(data):
    formatted_values = []
    for dataset, metrics in data.items():
        if isinstance(metrics, dict) and "top1" in metrics:
            formatted_values.append(f"{metrics['top1'] * 100:.2f}")
    
    if "average_accuracy" in data:
        formatted_values.append(f"{data['average_accuracy'] * 100:.2f}")
    
    return " & ".join(formatted_values)

# Example input
data = DATA
# Output result
print(format_accuracy(data))