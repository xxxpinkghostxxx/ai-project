with open('pylint errors.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 20:
            print(f"{i+1}: {line.strip()}")
        else:
            break