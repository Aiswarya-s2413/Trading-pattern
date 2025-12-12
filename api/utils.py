def relevance(item, query):
    symbol = item["symbol"].lower()
    name = item["name"].lower()

    # Highest priority: symbol starts with query
    if symbol.startswith(query):
        return (0, symbol)

    # Next: symbol contains query
    if query in symbol:
        return (1, symbol)

    # Next: name contains query
    if query in name:
        return (2, symbol)

    # Lowest priority
    return (3, symbol)