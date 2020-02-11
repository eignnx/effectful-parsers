from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Coroutine,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Text,
    Tuple,
    TypeVar,
    Union,
    cast,
)

Eff = TypeVar("Eff")
Resp = TypeVar("Resp")
T = TypeVar("T")


class ErrDescribe(ABC):
    @abstractmethod
    def err_describe(self) -> str:
        pass


@dataclass
class ParseResult(Generic[T]):
    SUCCESSFUL: ClassVar[bool] = True

    @property
    def failed(self) -> bool:
        return not self.SUCCESSFUL

    @property
    def succeeded(self) -> bool:
        return self.SUCCESSFUL


@dataclass
class Success(ParseResult[T]):
    parsed: T
    rest: str = ""

    SUCCESSFUL: ClassVar[bool] = True


@dataclass
class Failure(ParseResult[T], ErrDescribe):
    rest: str
    reason: Optional[ErrDescribe] = field(default=None)
    trace: List[Tuple[str, ParserFactory]] = field(default_factory=list)

    SUCCESSFUL: ClassVar[bool] = False

    def err_describe(self) -> str:
        preamble = f"Failure to parse at input '{self.rest}'"
        if self.reason is None:
            return preamble
        else:
            context = "\n".join(
                (
                    f"\t* During attempted parsing with `{parser!s}` at '{txt}'..."
                    for txt, parser in self.trace
                )
            )
            main_error = self.reason.err_describe()
            return f"{preamble} because:\n{context}\n\t- {main_error}\n"

    def add_parser_context(self, txt: str, parser_factory: ParserFactory) -> Failure[T]:
        self.trace.append((txt, parser_factory))
        return self

    def __str__(self) -> str:
        return self.err_describe()


ParserCoro = Coroutine[Eff, Optional[Resp], T]


# TODO: Make it a dataclass once https://github.com/python/mypy/issues/5485 is
# resolved.
class ParserFactory(Generic[Eff, Resp, T]):
    """
    Allows a `ParserCoro` coroutine to be constructed repeatedly. This class's
    constructor accepts a "thunk" (a function of no arguments) which produces
    a `ParserCoro[Eff, Resp, T]`.
    """

    def __init__(self, factory: Callable[..., ParserCoro[Eff, Resp, T]], args, kwargs):
        self.factory = factory
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(factory={self.factory!r})"

    def make(self) -> ParserCoro[Eff, Resp, T]:
        """
        Returns a new ParserCoro coroutine produced by the `self.factory` function.
        """
        return self.factory(*self.args, **self.kwargs)

    def __call__(self):
        """
        Since `ParserThunk[T]` is part of this module's API, `ParserFactory`
        objects have a __call__ method defined on them in order to justify
        users' potential intuition that a `ParserThunk` (which is an alias to
        `ParserFactory`) is callable.
        """
        return self.make()

    def __await__(self):
        """
        If a ParserFactory is awaited directly, a new ParserCoro coroutine will
        be instantiated, and it's `__await__` method will be invoked, returning
        an iterator.
        """
        # I don't understand this syntax. See: https://stackoverflow.com/a/33420721/9045161
        return self.make().__await__()

    def __str__(self) -> str:
        factory_name = self.factory.__name__

        def maybe_repr(x) -> str:
            return str(x) if isinstance(x, ParserFactory) else repr(x)

        args = ", ".join((maybe_repr(arg) for arg in self.args))
        kwargs = ", ".join((f"{name}={value!r}" for name, value in self.kwargs.items()))
        args_and_kwargs = f"{args}, {kwargs}" if kwargs else args
        return f"{factory_name}({args_and_kwargs})"


def parser_factory(
    f: Callable[..., ParserCoro[Eff, Resp, T]]
) -> Callable[..., ParserFactory[Eff, Resp, T]]:
    """
    Remembers the arguments you pass to `f` when you call it. Allows the
    coroutine to be restarted because a thunk (`factory`) is created on calling
    `f`.
    """

    def factory_builder(*args, **kwargs) -> ParserFactory[Eff, Resp, T]:
        return ParserFactory(f, args, kwargs)

    return factory_builder


"""A `ParserThunk[T]` is a user-friendly parser-factory type-alias."""
ParserThunk = ParserFactory[Any, Any, T]


class Effect(Generic[T], ABC):
    """
    An `Effect` is a message object which knows how to be awaited. When it is
    awaited, it simply returns a reference to itself to the runtime. The runtime
    then switches on the child-class type and behaves according to the semantics
    of the child-class.
    """

    def __await__(self):
        return (yield self)

    @abstractmethod
    def perform(self, txt: str) -> ParseResult[T]:
        pass


@dataclass
class Exactly(Effect[str], ErrDescribe):
    """
    Succeeds if the input begins with `target`, and fails otherwise. Returns
    `target` when it succeeds.
    """

    target: str

    def perform(self, txt: str) -> ParseResult[str]:
        target = self.target
        prefix, parsed, rest = txt.partition(target)
        if prefix != "" or parsed != target:
            return Failure(rest, reason=self)
        else:
            return Success(rest=rest, parsed=parsed)

    def err_describe(self) -> str:
        return f"Expected exactly '{self.target}'"


@parser_factory
async def exactly(target: str):
    return await Exactly(target)


@dataclass
class Many(Effect[List[T]], Generic[Eff, Resp, T]):
    """Parses 0 or more instances of `parser`. Returns a list of results."""

    parser: ParserFactory[Eff, Resp, T]

    def perform(self, txt: str) -> ParseResult[List[T]]:
        collected = []
        while True:
            res: ParseResult[T] = run_parser(self.parser, txt)
            if isinstance(res, Success):
                collected.append(res.parsed)
                txt = res.rest
            else:
                break
        return Success(rest=txt, parsed=collected)


@parser_factory
async def many(parser: ParserFactory[Eff, Resp, T]):
    return await Many(parser)


# TODO: Make it a dataclass once https://github.com/python/mypy/issues/5485 is
# resolved.
class TakeWhile(Effect[str], ErrDescribe):
    """Consumes input while the char -> bool `predicate` holds."""

    def __init__(
        self,
        predicate: Callable[[str], bool],
        pred_name: Optional[str] = None,
        allow_empty: bool = True,
    ):
        self.predicate = predicate
        self.pred_name = pred_name
        self.allow_empty = allow_empty

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(predicate={self.predicate!r})"

    def perform(self, txt: str) -> ParseResult[str]:
        i = 0  # Default in case txt is empty
        for i, ch in enumerate(txt):
            if not self.predicate(ch):
                break
        if not self.allow_empty and i == 0:
            return Failure(txt, reason=self)
        else:
            return Success(rest=txt[i:], parsed=txt[:i])

    def err_describe(self) -> str:
        pred_name = self.pred_name if self.pred_name else repr(self.predicate)
        return (
            f"The predicate '{pred_name}' given to "
            f"`take_nonempty_while` never matched any of the input."
        )


@parser_factory
async def take_while(predicate: Callable[[str], bool]) -> str:
    """
    Consumes characters one at a time, and stops when `predicate` applied on
    the next character is `False`. Be aware that this parser will succeed when
    given the empty string, or if the first character encountered does NOT
    satisfy `predicate`.
    """
    return await TakeWhile(predicate)


@parser_factory
async def take_nonempty_while(
    predicate: Callable[[str], bool], pred_name: Optional[str] = None
) -> str:
    """
    The `pred_name` parameter is optional, but when provided, can give better
    error reporting upon parse failure.
    """
    return await TakeWhile(predicate, pred_name=pred_name, allow_empty=False)


@dataclass
class Either(Effect[T], Generic[Eff, Resp, T], ErrDescribe):
    """Returns the output of the first parser in `parsers` that succeeds."""

    parsers: Iterable[ParserFactory[Eff, Resp, T]]

    def perform(self, txt: str) -> ParseResult[T]:
        for sub_parser in self.parsers:
            res = run_parser(sub_parser, txt)
            if res.succeeded:
                return res
        else:
            # If none of the parsers succeed...
            return Failure(txt, reason=self)

    def err_describe(self) -> str:
        ps = "\n".join((f"\t\t- {parser}" for parser in self.parsers))
        return f"None of the following parsers succeeded:\n{ps}"


@parser_factory
async def either(*parsers: ParserFactory[Eff, Resp, T]) -> T:
    return await Either(parsers)


@dataclass
class Matches(Effect[str], ErrDescribe):
    re: re.Pattern

    def perform(self, txt: str) -> ParseResult[str]:
        match = self.re.match(txt)
        if match is None:
            return Failure(txt, reason=self)
        else:
            return Success(parsed=match.group(), rest=txt[match.end() :])

    def err_describe(self) -> str:
        return f"The regex '{self.re.pattern}' did not match the beginning of the input text."


@parser_factory
async def matches(re_src: Text, *compile_args, **compile_kwargs):
    return await Matches(re.compile(re_src, *compile_args, **compile_kwargs))


Default = TypeVar("Default")


@dataclass
class Optional_(Effect[Union[T, Default]], Generic[Eff, Resp, T, Default]):
    parser: ParserFactory[Eff, Resp, T]
    default: Default

    def perform(self, txt: str) -> ParseResult[Union[T, Default]]:
        res = run_parser(self.parser, txt)
        if isinstance(res, Success):
            return res
        elif isinstance(res, Failure):
            return Success(parsed=self.default, rest=txt)
        else:
            raise Exception("Unreachable")


@parser_factory
async def optional(parser: ParserThunk[T], default=None) -> Union[T, Default]:
    return await Optional_(parser, default)


def run_parser(parser_factory: ParserFactory[Eff, Resp, T], txt: str) -> ParseResult[T]:
    parser = parser_factory.make()

    send_value: Optional[Resp] = None

    while True:
        try:
            got = parser.send(send_value)
        except StopIteration as e:
            return Success(rest=txt, parsed=e.value)

        if isinstance(got, Effect):
            res = got.perform(txt)
            if isinstance(res, Success):
                txt, send_value = res.rest, res.parsed
            elif isinstance(res, Failure):
                return res.add_parser_context(txt, parser_factory)
        else:
            raise Exception(f"Expected parser object, got {got}")


@parser_factory
async def many1(parser: ParserFactory[Eff, Resp, T]) -> List[T]:
    first = await parser
    rest = await many(parser)
    return [first] + rest


@parser_factory
async def preceded(prefix: ParserFactory, target: ParserFactory):
    await prefix
    return await target


@parser_factory
async def separated_nonempty_list(
    item: ParserFactory[Eff, Resp, T], sep: ParserFactory
) -> List[T]:
    first = await item
    rest = await many(preceded(sep, item))
    return [first] + rest


@parser_factory
async def py_int() -> int:
    """
    Parses a Python int.
    """
    # See https://docs.python.org/3/library/re.html#simulating-scanf
    digits = await matches(r"[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)")
    return int(digits)


@parser_factory
async def py_float(allow_special_values: bool = False) -> float:
    """
    Parses a Python float. By default, this parser accepts any float literal
    that is valid in a Python source file. Note that this excludes the special
    values 'nan', 'inf', and 'Infinity', as those values are constructed
    repectively by `float('nan')`, `float('inf')`, and `float('Infinity')`. If
    the `allow_special_values` flag is enabled, then the special values 'nan',
    'inf' and 'Infinity' will be accepted, emulating the behavior of the builtin
    `float` function.
    """
    if allow_special_values:
        raise NotImplementedError()
    # TODO: parse according to actual python float literal syntax.
    # See https://docs.python.org/3/library/re.html#simulating-scanf
    digits = await matches(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
    if "_" in digits:
        raise NotImplementedError()
    return float(digits)
