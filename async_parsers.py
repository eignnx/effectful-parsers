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
                    for txt, parser in reversed(self.trace)
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

    def map(self, f: Callable[[T], U]) -> ParserFactory[Eff, Resp, U]:
        @parser_factory
        async def map(f, p) -> U:
            return await Map(f, p)

        return map(f, self)

    def __or__(self, other):
        return either(self, other)

    def __lshift__(self, other):
        return terminated(self, other)

    def __rshift__(self, other):
        return preceded(self, other)


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
async def either(*parsers: ParserFactory[Any, Any, T]) -> T:
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


class Map(Effect[U], Generic[Eff, Resp, T, U]):
    def __init__(self, f: Callable[[T], U], parser: ParserFactory[Eff, Resp, T]):
        self.f = f
        self.parser = parser

    def perform(self, txt: str) -> ParseResult[U]:
        res = run_parser(self.parser, txt)
        if isinstance(res, Success):
            return Success(parsed=self.f(res.parsed), rest=res.rest)
        else:
            return cast(ParseResult[U], res)


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
async def terminated(target: ParserFactory, suffix: ParserFactory):
    x = await target
    await suffix
    return x


@parser_factory
async def sequence(*parsers: ParserFactory):
    result = []
    for parser in parsers:
        result.append(await parser)
    return result


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
    # FROM https://docs.python.org/3.9/reference/lexical_analysis.html#integer-literals
    # integer      ::=  decinteger | bininteger | octinteger | hexinteger
    # decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
    # bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
    # octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
    # hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
    # nonzerodigit ::=  "1"..."9"
    # digit        ::=  "0"..."9"
    # bindigit     ::=  "0" | "1"
    # octdigit     ::=  "0"..."7"
    # hexdigit     ::=  digit | "a"..."f" | "A"..."F"

    @dataclass
    class Base:
        n: int

        def __call__(self, digits: str) -> int:
            return int(digits.lstrip("_"), self.n)

    dec_integer = matches(r"[1-9](_?[0-9])*|0+(_?0)*").map(Base(10))
    hex_integer = preceded(
        matches(r"0[xX]"), matches(r"(_?[0-9a-fA-F])+").map(Base(16))
    )
    oct_integer = preceded(matches(r"0[oO]"), matches(r"(_?[0-7])+").map(Base(8)))
    bin_integer = preceded(matches(r"0[bB]"), matches(r"(_?[01])+").map(Base(2)))

    sign = await matches(r"[+-]?").map(lambda s: -1 if s == "-" else 1)
    return await either(hex_integer, oct_integer, bin_integer, dec_integer).map(
        lambda x: sign * x
    )


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

    # FROM https://docs.python.org/3.9/library/functions.html?highlight=float#float
    # sign           ::=  "+" | "-"
    # infinity       ::=  "Infinity" | "inf"
    # nan            ::=  "nan"
    # numeric_value  ::=  floatnumber | infinity | nan
    # numeric_string ::=  [sign] numeric_value

    # From https://docs.python.org/3.9/reference/lexical_analysis.html#floating-point-literals
    # floatnumber   ::=  pointfloat | exponentfloat
    # pointfloat    ::=  [digitpart] fraction | digitpart "."
    # exponentfloat ::=  (digitpart | pointfloat) exponent
    # digitpart     ::=  digit (["_"] digit)*
    # fraction      ::=  "." digitpart
    # exponent      ::=  ("e" | "E") ["+" | "-"] digitpart

    digits = matches(r"\d[_\d]*")

    point_float = either(
        sequence(digits, exactly("."), digits),
        sequence(exactly("."), digits),
        sequence(digits, exactly(".")),
    )

    @parser_factory
    async def exponent():
        await matches(r"[eE]")
        await matches(r"[+-]?")
        await digits

    exponent_float = sequence(either(point_float, digits), exponent())

    special_values = (
        [exactly("nan"), exactly("inf"), exactly("Infinity")]
        if allow_special_values
        else []
    )

    @parser_factory
    async def py_float_format() -> None:
        await matches(r"[+-]?")
        await either(exponent_float, point_float, *special_values)

    float_text = await recognize(py_float_format())
    return float(float_text)
