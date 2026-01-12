from pythia.python_utils import _py_version

if _py_version >= (3, 11):
    from typing import Generic, TypeVar, TypedDict
    T = TypeVar("T")
    E = TypeVar("E")
    class Result(TypedDict, Generic[T, E]):
        ok:  T
        err: E

else:
    from typing import Any, Literal, Union
    Result = dict[Union[Literal["ok"], Literal["err"]], Any]
