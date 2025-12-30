from typing import Optional, TypedDict, Union, get_type_hints
from dataclasses import dataclass

@dataclass
class Span:
    value: str
    start: int
    end: int

class NotPresent: pass

def extract_struct(haystack: str, ty) -> dict:
    def _get_default_value(ty, k):
        if hasattr(ty, k):
            v = getattr(ty, k)
        else:
            raise AttributeError(
                f"extract: {ty.__name__}: {repr(k)}"
            )
    d = dict()
    th = get_type_hints(ty)
    for k, ty in th.items():
        if ty is Optional[str]:
            span = extract_tagged_span(haystack, k)
            if span is None:
                v = _get_default_value(ty, k)
            else:
                v = span.value
        elif ty is Optional[bool]:
            v = extract_tagged_maybe_bool(haystack, k)
            if isinstance(v, NotPresent):
                v = _get_default_value(ty, k)
        elif ty is Optional[int]:
            v = extract_tagged_maybe_int(haystack, k)
            if isinstance(v, NotPresent):
                v = _get_default_value(ty, k)
        else:
            raise NotImplementedError
        d[k] = v
    return d

def extract_tagged_span(haystack: str, tag: str) -> Optional[Span]:
    needle = haystack
    start_tag = f"<{tag}>"
    start = needle.rfind(start_tag)
    if start < 0:
        return None
    start += len(start_tag)
    needle = needle[start:]
    end_tag = f"</{tag}>"
    end = needle.find(end_tag)
    if end < 0:
        return Span(needle, start, start + len(needle))
    needle = needle[:end]
    return Span(needle, start, start + end)

def extract_tagged_maybe_bool(haystack: str, tag: str) -> Union[NotPresent, Optional[bool]]:
    span = extract_tagged_span(haystack, tag)
    if span is None:
        return NotPresent()
    if span.value == "True":
        return True
    if span.value == "False":
        return False
    return None

def extract_tagged_maybe_int(haystack: str, tag: str) -> Union[NotPresent, Optional[int]]:
    span = extract_tagged_span(haystack, tag)
    if span is None:
        return NotPresent()
    try:
        value = int(span.value)
        return value
    except ValueError:
        return None

if __name__ == "__main__":
    class Foo(TypedDict):
        hello: Optional[str]
        world: Optional[bool]

    class Bar(Foo):
        goodbye: Optional[int] = None

    class MockClient:
        pass

    class Baz(Bar):
        @staticmethod
        def hi(client: MockClient, messages: list) -> "Baz":
            pass

    print(issubclass(Foo, dict))
    # print(issubclass(Foo, TypedDict))

    th = get_type_hints(Foo)
    print(th)

    th = get_type_hints(Bar)
    print(th)

    print(Bar.goodbye)
    # print(Bar.world)

    th = get_type_hints(Baz)
    print(th)
    print(th["world"] is Optional[bool])

    pat = "<hello>world</hello><world>True</world>"
    d = extract(pat, Foo)
    print(d)

    pat = "<hello>world</hello><world>True</world>"
    d = extract(pat, Bar)
    print(d)

    pat = "<hello>world</hello><goodbye>2</goodbye><world>True</world>"
    d = extract(pat, Bar)
    print(d)

    pat = "<hello>world</hello>"
    d = extract(pat, Foo)
    print(d)
