from typing import Any, Optional, TypedDict, Union
from dataclasses import dataclass
from io import StringIO

class MessagePart(TypedDict):
    type: str
    text: Optional[str]
    thinking: Optional[str]
    signature: Optional[str]

class Message(TypedDict):
    role: str
    content: Union[str, list[MessagePart]]

    @staticmethod
    def get_text(message: dict) -> Optional[str]:
        content = message.get("content", None)
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text = None
            for part in content:
                if part["type"] == "thinking":
                    pass
                elif part["type"] == "text":
                    if text is not None:
                        raise ValueError
                    text = part["text"]
                else:
                    raise NotImplementedError
            return text
        else:
            raise NotImplementedError

    @staticmethod
    def get_thinking(message: dict) -> Optional[str]:
        content = message.get("content", None)
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            thinking = None
            for part in content:
                if part["type"] == "thinking":
                    if thinking is not None:
                        raise ValueError
                    thinking = part["thinking"]
                elif part["type"] == "text":
                    pass
                else:
                    raise NotImplementedError
            return thinking
        else:
            raise NotImplementedError

    @staticmethod
    def get_thinking_part(message: dict) -> Optional[dict]:
        content = message.get("content", None)
        if isinstance(content, str):
            return None
        elif isinstance(content, list):
            thinking_part = None
            for part in content:
                if part["type"] == "thinking":
                    if thinking_part is not None:
                        raise ValueError
                    thinking_part = part
                elif part["type"] == "text":
                    pass
                else:
                    raise NotImplementedError
            return thinking_part
        else:
            raise NotImplementedError

class TextBlock(TypedDict):
    start: int
    end: int
    text: str

class MarkdownTextBlock(TextBlock):
    pass

@dataclass
class MarkdownHeaderIndex:
    _pivots: dict[int, list[tuple[int, str]]]

    @classmethod
    def new(cls, haystack: str) -> "MarkdownHeaderIndex":
        pivots = dict()
        headers = []
        code = False
        line_start = 0
        line_end = -1
        while True:
            line_end = haystack.find("\n", line_start)
            og_line_end = line_end
            if line_end < 0:
                line_end = len(haystack)
            line_len = line_end - line_start
            if line_len >= 3 and haystack[line_start:line_start+3] == "```":
                code = not code
            append_header = False
            if line_len >= 1 and haystack[line_start] == "#" and not code:
                level = 1
                while level < line_len and haystack[line_start+level] == "#":
                    level += 1
                text = haystack[line_start+level:line_end].strip()
                while headers and headers[-1][0] >= level:
                    headers.pop()
                headers.append((level, text))
                append_header = True
            line_end += 1
            if line_end < len(haystack) and haystack[line_end] == "\r":
                line_end += 1
            if append_header:
                pivots[(line_start, line_end)] = headers.copy()
            if og_line_end < 0:
                break
            line_start = line_end
        return cls(pivots)

    def find(self, pos: int) -> list[tuple[int, str]]:
        prev_headers = []
        for (hstart, _), headers in self._pivots.items():
            if pos < hstart:
                return prev_headers
            prev_headers = headers
        return prev_headers

    def extract_prelude_text(self, haystack: str, start: int, end: int) -> MarkdownTextBlock:
        prev_hstart = 0
        prev_hend = 0
        prev_headers = []
        for (hstart, hend), headers in self._pivots.items():
            if end <= hstart:
                break
            prev_hstart = hstart
            prev_hend = hend
            prev_headers = headers
        start = max(start, prev_hend)
        text = haystack[start:end]
        return {
            "start": start,
            "end": end,
            "text": text,
        }

class MarkdownCodeBlock(MarkdownTextBlock):
    lang: Optional[str]

    @staticmethod
    def extract_next(haystack: str, start: Optional[int] = None) -> Optional["MarkdownCodeBlock"]:
        return extract_next_markdown_code_block(haystack, start)

def extract_next_markdown_code_block(haystack: str, start: Optional[int] = None) -> Optional[MarkdownCodeBlock]:
    if start is not None:
        start_pos = haystack.find("```", start)
    else:
        start_pos = haystack.find("```")
    if start_pos < 0:
        return None
    end_pos = haystack.find("```", start_pos + 3)
    if end_pos < 0:
        # FIXME(20250824): could be truncated.
        return None
    start_eol_pos = haystack.find("\n", start_pos + 3)
    if start_eol_pos < 0:
        return None
    if start_pos + 3 < start_eol_pos:
        lang = haystack[start_pos+3:start_eol_pos]
    else:
        lang = None
    start_eol_pos += 1
    if haystack[start_eol_pos] == "\r":
        start_eol_pos += 1
    text = haystack[start_eol_pos:end_pos]
    return {
        "start": start_pos,
        "end": end_pos + 3,
        "lang": lang,
        "text": text,
    }

def extract_first_markdown_code_block(haystack: str) -> Optional[MarkdownCodeBlock]:
    return extract_next_markdown_code_block(haystack)

def extract_last_markdown_code_block(haystack: str) -> Optional[MarkdownCodeBlock]:
    end_pos = haystack.rfind("```")
    if end_pos < 0:
        return None
    start_pos = haystack.rfind("```", end_pos)
    if start_pos < 0:
        # FIXME(20250824): could be truncated.
        # start_pos = end_pos
        return None
    start_eol_pos = haystack.find("\n", start_pos + 3)
    if start_eol_pos < 0:
        return None
    if start_pos + 3 < start_eol_pos:
        lang = haystack[start_pos+3:start_eol_pos]
    else:
        lang = None
    if haystack[start_eol_pos] == "\r":
        start_eol_pos += 1
    text = haystack[start_eol_pos:end_pos]
    return {
        "start": start_pos,
        "end": end_pos + 3,
        "lang": lang,
        "text": text,
    }
