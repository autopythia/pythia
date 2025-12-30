from typing import Optional, TypedDict
from dataclasses import dataclass
from io import StringIO
import ast
import token
import tokenize

class Token(TypedDict):
    start: int
    end: int
    text: str
    ty: int
    xty: Optional[int]
    start_row: int
    start_col: int
    end_row: int
    end_col: int

    @staticmethod
    def type_name(tok: "Token") -> str:
        return token.tok_name[tok["ty"]]

    @staticmethod
    def exact_type_name(tok: "Token") -> Optional[str]:
        if tok["xty"] is not None:
            return token.tok_name[tok["xty"]]
        else:
            return None

def extract_tokens(code_text: str) -> list[Token]:
    toks = []
    for tok in generate_tokens(code_text):
        toks.append(tok)
    return toks

def generate_tokens(code_text: str):
    tok_start = 0
    for tok in tokenize.generate_tokens(StringIO(code_text).readline):
        tok_len = len(tok.string)
        tok_end = tok_start + tok_len
        if tok.type != tok.exact_type:
            tok_xty = tok.exact_type
        else:
            tok_xty = None
        new_tok = {
            "start": tok_start,
            "end": tok_end,
            "text": tok.string,
            "ty": tok.type,
            "xty": tok_xty,
            "start_row": tok.start[0] - 1,
            "start_col": tok.start[1],
            "end_row": tok.end[0] - 1,
            "end_col": tok.end[1],
        }
        yield new_tok
        tok_start = tok_end

class Ast(TypedDict):
    pass

    @staticmethod
    def from_root_node(root) -> "Ast":
        assert isinstance(root, ast.Module)
        for node in root.body:
            # node.lineno
            pass

    @staticmethod
    def from_node(node) -> "Ast":
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    pass
        else:
            raise NotImplementedError(
                f"ast node type: {type(node).__name__}"
            )

def parse_ast(code_text: str):
    root = ast.parse(code_text)
    ast = Ast.from_root_node(root)

@dataclass
class CodeIndex:
    pass

    @classmethod
    def from_text(cls, code_text: str):
        this = cls()
        comments = dict()
        elifs = dict()
        elses = dict()
        top_stms = dict()
        for tok in generate_tokens(code_text):
            if tok["ty"] == 60:
                assert tok["start_row"] not in comments
                comments[tok["start_row"]] = tok
            elif tok["ty"] == 1:
                if tok["text"] == "elif":
                    assert tok["start_row"] not in elifs
                    elifs[tok["start_row"]] = tok
                elif tok["text"] == "else":
                    assert tok["start_row"] not in elses
                    elses[tok["start_row"]] = tok
        root = ast.parse(code_text)
        assert isinstance(root, ast.Module)
        def top_level_if_stm(node):
            if (
                len(node.orelse) == 1 and
                isinstance(node.orelse[0], ast.If)
            ):
                stm_type = "elif"
                node = node.orelse[0]
                yield stm_type, node
                top_level_if_stm(node)
            else:
                stm_type = "else"
                # FIXME
                node = node.orelse[0]
                yield stm_type, node
        def top_level_stms(body):
            for node in body:
                stm_type = None
                if isinstance(node, ast.If):
                    stm_type = "if"
                yield stm_type, node
                if isinstance(node, ast.If):
                    yield from top_level_if_stm(node)
        for stm_type, node in top_level_stms(root.body):
            start_row = node.lineno - 1
            start_col_offset = node.col_offset
            if stm_type == "elif":
                new_start_row = None
                # FIXME
                new_start_col_offset = 0
                for new_start_row in elifs:
                    if new_start_row < start_row:
                        pass
                    else:
                        break
                assert new_start_row is not None
                start_row = new_start_row
                start_col_offset = new_start_col_offset
            elif stm_type == "else":
                new_start_row = None
                # FIXME
                new_start_col_offset = 0
                for new_start_row in elses:
                    if new_start_row < start_row:
                        pass
                    else:
                        break
                assert new_start_row is not None
                start_row = new_start_row
                start_col_offset = new_start_col_offset
            end_row = node.end_lineno - 1
            end_col_offset = node.end_col_offset
            assert start_row not in top_stms
            top_stms[start_row] = {
                "stm_num": None,
                "stm_type": stm_type,
                "node_type": type(node).__name__,
                "start_row": start_row,
                "start_col_offset": start_col_offset,
                "end_row": end_row,
                "end_col_offset": end_col_offset,
            }
        this.comments = comments
        this.top_stms = top_stms
        for start_row, stm in this.top_stms.items():
            if start_row - 1 in this.comments:
                print(
                    start_row,
                    repr(this.comments[start_row-1]["text"]),
                    repr(stm["node_type"]),
                )
        return this
