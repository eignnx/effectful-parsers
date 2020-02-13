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
U = TypeVar("U")


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
    trace: List[Tuple[str, Combinator]] = field(default_factory=list)

    SUCCESSFUL: ClassVar[bool] = False

    def err_describe(self) -> str:
        preamble = f"Failure to parse at input '{self.rest}'"
        if self.reason is None:
            return preamble
        else:
            context = "\n".join(
                (
                    f"\t* During attempted parsing with `{parser!s}` at '{txt}'..."
                    for txt, parser in reversed(self.trace)
                )
            )
            main_error = self.reason.err_describe()
            return f"{preamble} because:\n{context}\n\t- {main_error}\n"

    def add_parser_context(self, txt: str, parser_factory: Combinator) -> Failure[T]:
        self.trace.append((txt, parser_factory))
        return self

    def __str__(self) -> str:
        return self.err_describe()


ParserCoro = Coroutine[Eff, Optional[Resp], T]


class Combinator(Generic[T], ABC):
    @abstractmethod
    def __await__(self):
        pass

    @abstractmethod
    def make(self) -> ParserCoro[Any, Any, T]:
        pass

    def map(self, f: Callable[[T], U]):
        return Map(f, self)

    def __or__(self, other: Combinator[T]) -> Combinator[T]:
        return Either((self, other))

    def __lshift__(self, other: Combinator[U]) -> Combinator[T]:
        return cast(Combinator[T], terminated(self, other))

    def __rshift__(self, other: Combinator[U]) -> Combinator[U]:
        return cast(Combinator[U], preceded(self, other))


# TODO: Make it a dataclass once https://github.com/python/mypy/issues/5485 is
# resolved.
class ParserFactory(Generic[Eff, Resp, T], Combinator[T]):
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
            return str(x) if isinstance(x, (ParserFactory, Effect)) else repr(x)

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


class Effect(Generic[T], Combinator[T], ABC):
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

    def make(self):
        return self.__await__()


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

    parsers: Iterable[Combinator[T]]

    def __post_init__(self):
        # Speeds up runtime parsing execution by flattening an
        # `either(either(a, b), c)` into an `either(a, b, c)`.
        flattened: List[Combinator[T]] = []
        for sub_parser in self.parsers:
            if isinstance(sub_parser, Either):
                for sub_sub_parser in sub_parser.parsers:
                    flattened.append(sub_sub_parser)
            else:
                flattened.append(sub_parser)
        self.parsers = flattened

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

    def __str__(self):
        return " | ".join((str(parser) for parser in self.parsers))


def either(*parsers: Combinator[T]) -> Combinator[T]:
    return Either(parsers)


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
    parser: Combinator[T]
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
async def optional(parser: Combinator[T], default=None) -> Union[T, Default]:
    return await Optional_(parser, default)


class Map(Effect[U], Generic[Eff, Resp, T, U]):
    def __init__(self, f: Callable[[T], U], parser: Combinator[T]):
        self.f = f
        self.parser = parser

    def perform(self, txt: str) -> ParseResult[U]:
        res = run_parser(self.parser, txt)
        if isinstance(res, Success):
            return Success(parsed=self.f(res.parsed), rest=res.rest)
        else:
            return cast(ParseResult[U], res)

    def __str__(self):
        return f"{self.parser}.map({self.f!r})"


@dataclass
class Recognize(Effect[str]):
    parser: ParserFactory

    def perform(self, txt: str) -> ParseResult[str]:
        res = run_parser(self.parser, txt)
        if isinstance(res, Success):
            diff = len(txt) - len(res.rest)
            return Success(parsed=txt[:diff], rest=res.rest)
        else:
            return cast(ParseResult[str], res)


@parser_factory
async def recognize(parser):
    """
    Runs the sub-pareser, but discards the result and instead returns the string
    that was accepted by the sub-parser.
    """
    return await Recognize(parser)


@parser_factory
async def preceded(prefix: Combinator[T], target: Combinator[U]) -> U:
    await prefix
    return await target


@parser_factory
async def terminated(target: Combinator[T], suffix: Combinator[U]) -> T:
    x = await target
    await suffix
    return x


def run_parser(parser_factory: Combinator[T], txt: str) -> ParseResult[T]:
    parser = parser_factory.make()

    send_value: Optional[Any] = None

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
