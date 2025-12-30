def unquote(s):
    """
    Remove surrounding single or double quotes from a string, if present.
    """
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s

def parse_inline(v):
    """
    Parse an inline YAML value: flow‑style list or plain scalar.
    """
    v = v.strip()
    # flow list
    if v.startswith('[') and v.endswith(']'):
        items = v[1:-1].split(',')
        return [parse_inline(it.strip()) for it in items if it.strip()]
    # plain scalar (remove quotes if any)
    return unquote(v)

def loads(stream):
    """
    Parse a YAML document from a string or file‑like object.
    Returns a Python dict or list.
    """
    if isinstance(stream, str):
        lines = stream.splitlines()
    else:
        lines = stream.readlines()

    # ----- preprocess: keep (indent, stripped_line), skip empty/comments -----
    processed = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(line) - len(stripped)
        processed.append((indent, stripped))

    if not processed:
        return None

    # ----- determine root type (dict or list) -----
    first_indent, first_content = processed[0]
    if first_content.startswith('-'):
        root = []
        root_type = 'list'
    else:
        root = {}
        root_type = 'dict'

    # stack holds current containers; each entry: container, its indent, type
    stack = [{'container': root, 'indent': -1, 'type': root_type}]

    i = 0
    while i < len(processed):
        indent, content = processed[i]

        # pop stack until we find the container that owns this line
        while stack[-1]['indent'] >= indent:
            stack.pop()

        top = stack[-1]

        # ----- dictionary context -----
        if top['type'] == 'dict':
            if ':' not in content:
                raise ValueError(f"Expected 'key: value' in dict, got: {content}")

            key, rest = content.split(':', 1)
            key = key.strip()
            key = unquote(key)
            rest = rest.strip()

            if rest:
                # inline value (scalar or flow list)
                top['container'][key] = parse_inline(rest)
            else:
                # value is a nested block
                next_indent = next_content = None
                if i + 1 < len(processed):
                    next_indent, next_content = processed[i+1]

                if next_indent is not None and next_indent > indent:
                    # block exists: decide list or dict by looking at the first line
                    if next_content.startswith('-'):
                        new_container = []
                        new_type = 'list'
                    else:
                        new_container = {}
                        new_type = 'dict'
                    top['container'][key] = new_container
                    stack.append({'container': new_container,
                                  'indent': indent,
                                  'type': new_type})
                else:
                    # no nested lines → null value
                    top['container'][key] = None

        # ----- list context -----
        elif top['type'] == 'list':
            if not content.startswith('-'):
                raise ValueError(f"Expected list item starting with '-', got: {content}")

            item_content = content[1:].lstrip()

            if item_content:
                # item with inline content
                if item_content.startswith('[') and item_content.endswith(']'):
                    # flow list
                    value = parse_inline(item_content)
                    top['container'].append(value)
                elif item_content.startswith('{') and item_content.endswith('}'):
                    # flow map – not fully implemented, keep as string
                    value = item_content
                    top['container'].append(value)
                elif ':' in item_content:
                    # mapping start (first key‑value pair on the same line)
                    item_dict = {}
                    key, val_rest = item_content.split(':', 1)
                    key = key.strip()
                    val_rest = val_rest.strip()
                    if val_rest:
                        item_dict[key] = parse_inline(val_rest)
                    else:
                        item_dict[key] = None
                    top['container'].append(item_dict)

                    # if there are deeper lines, push this dict to allow more keys
                    if i + 1 < len(processed) and processed[i+1][0] > indent:
                        stack.append({'container': item_dict,
                                      'indent': indent,
                                      'type': 'dict'})
                else:
                    # plain scalar
                    value = parse_inline(item_content)
                    top['container'].append(value)
            else:
                # empty item (just '-')
                if i + 1 < len(processed) and processed[i+1][0] > indent:
                    next_indent, next_content = processed[i+1]
                    if next_content.startswith('-'):
                        new_container = []
                        new_type = 'list'
                    else:
                        new_container = {}
                        new_type = 'dict'
                    top['container'].append(new_container)
                    stack.append({'container': new_container,
                                  'indent': indent,
                                  'type': new_type})
                else:
                    top['container'].append(None)

        else:
            raise RuntimeError("Unexpected container type")

        i += 1

    return root
