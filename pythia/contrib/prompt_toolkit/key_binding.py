from .keys import Keys

class KeyPress:
    def __init__(self, key, data=None):
        if data is None:
            data = key.value if isinstance(key, Keys) else key
        self.key = key
        self.data = data
