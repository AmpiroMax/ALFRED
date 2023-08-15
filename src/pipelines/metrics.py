from torchmetrics import CharErrorRate

CER_METRIC = CharErrorRate()


def count_acc(prediction: list[str], labels: list[str]) -> float:
    cleaned = [
        pred.replace("<|endoftext|>", "")
        for pred in prediction
    ]
    answers = [
        pred.split("?")[1].strip()
        if len(pred.split("?")) > 1 else ""
        for pred in cleaned
    ]

    count = [
        1 if ans == label else 0
        for ans, label in zip(answers, labels)
    ]

    acc = sum(count) / len(count)
    # print(answers)
    # print(labels)
    # print(acc)
    return acc


def count_cer(prediction: list[str], labels: list[str]) -> float:
    cleaned = [
        pred.replace("<|endoftext|>", "")
        for pred in prediction
    ]
    answers = [
        pred.split("?")[1].strip()
        if len(pred.split("?")) > 1 else ""
        for pred in cleaned
    ]

    cer = CER_METRIC(answers, labels)
    # print(answers)
    # print(labels)
    # print(cer)
    return cer


def count_metrics(prediction: list[str], labels: list[str]) -> dict:
    cleaned = [
        pred.replace("<|endoftext|>", "")
        for pred in prediction
    ]
    answers = [
        pred.split("?")[1].strip()
        if len(pred.split("?")) > 1 else ""
        for pred in cleaned
    ]

    count = [
        1 if ans == label else 0
        for ans, label in zip(answers, labels)
    ]

    acc = sum(count) / len(count)

    cer = CER_METRIC(answers, labels).numpy()
    # print(answers)
    # print(labels)
    # print(cer)
    # print(acc)
    return {
        "accuracy": acc,
        "cer": cer
    }
