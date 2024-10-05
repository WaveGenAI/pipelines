import hashlib


def hash_url(url: str) -> str:
    """Function to hash the url

    Args:
        url (str): URL to hash

    Returns:
        str: Hashed URL
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()
