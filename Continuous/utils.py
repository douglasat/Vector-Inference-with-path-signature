def shorten_name(text: str, max_length: int = 10) -> str:
    """Truncate text that are too long

    Args:
        text (str): The text to truncate
        max_length (int, optional): Maximum length of the text. Defaults to 10.

    Returns:
        str: The truncated text
    """
    if not isinstance(text, str):
        text = str(text)
    return text if len(text) <= max_length else text[:max_length] + "..."