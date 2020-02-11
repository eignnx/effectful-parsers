from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

Eff = TypeVar("Eff")
Resp = TypeVar("Resp")
T = TypeVar("T")

PRes = Optional[Tuple[str, T]]
ParserCoro = Coroutine[Eff, Optional[Resp], T]


# TODO: Make it a dataclass once https://github.com/python/mypy/issues/5485 is
# resolved.
class ParserFactory(Generic[Eff, Resp, T]):
    """
    Allows a `ParserCoro` coroutine to be constructed repeatedly. This class's
    constructor accepts a "thunk" (a function of no arguments) which produces
    a `ParserCoro[Eff, Resp, T]`.
    """

    def __init__(self, factory: Callable[[], ParserCoro[Eff, Resp, T]]):
        self.factory = factory

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(factory={self.factory!r})"

    def make(self) -> ParserCoro[Eff, Resp, T]:
        """
        Returns a new ParserCoro coroutine produced by the `self.factory` function.
        """
        return self.factory()

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


def parser_factory(
    f: Callable[..., ParserCoro[Eff, Resp, T]]
) -> Callable[..., ParserFactory[Eff, Resp, T]]:
    """
    Remembers the arguments you pass to `f` when you call it. Allows the
    coroutine to be restarted because a thunk (`factory`) is created on calling
    `f`.
    """

    def factory_builder(*args, **kwargs) -> ParserFactory[Eff, Resp, T]:
        @wraps(f)
        def thunk():
            return f(*args, **kwargs)

        return ParserFactory(thunk)

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
    def perform(self, txt: str) -> PRes[T]:
        pass


@dataclass
class Exactly(Effect[str]):
    """
    Succeeds if the input begins with `target`, and fails otherwise. Returns
    `target` when it succeeds.
    """

    target: str

    def perform(self, txt: str) -> PRes[str]:
        target = self.target
        prefix, parsed, rest = txt.partition(target)
        if prefix != "" or parsed != target:
            return None
        else:
            send_value = parsed
            txt = rest
            return (txt, send_value)


@parser_factory
async def exactly(target: str):
    return await Exactly(target)


@dataclass
class Many(Effect[List[T]], Generic[Eff, Resp, T]):
    """Parses 0 or more instances of `parser`. Returns a list of results."""

    parser: ParserFactory[Eff, Resp, T]

    def perform(self, txt: str) -> PRes[List[T]]:
        collected = []
        while True:
            res: PRes[T] = run_parser(self.parser, txt)
            if res:
                rest, parsed = res
                collected.append(parsed)
                txt = rest
            else:
                break
        send_value = collected
        return (txt, send_value)


@parser_factory
async def many(parser: ParserFactory[Eff, Resp, T]):
    return await Many(parser)


# TODO: Make it a dataclass once https://github.com/python/mypy/issues/5485 is
# resolved.
class TakeWhile(Effect[str]):
    """Consumes input while the char -> bool `predicate` holds."""

    def __init__(self, predicate: Callable[[str], bool]):
        self.predicate = predicate

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(predicate={self.predicate!r})"

    def perform(self, txt: str) -> PRes[str]:
        i = 0  # Default in case txt is empty
        for i, ch in enumerate(txt):
            if not self.predicate(ch):
                break
        send_value, txt = txt[:i], txt[i:]
        return (txt, send_value)


@parser_factory
async def take_while(predicate: Callable[[str], bool]):
    return await TakeWhile(predicate)


@dataclass
class Either(Effect[T], Generic[Eff, Resp, T]):
    """Returns the output of the first parser in `parsers` that succeeds."""

    parsers: Iterable[ParserFactory[Eff, Resp, T]]

    def perform(self, txt: str) -> PRes[T]:
        for sub_parser in self.parsers:
            res = run_parser(sub_parser, txt)
            if res:
                rest, parsed = res
                txt = rest
                send_value = parsed
                return (txt, send_value)
        else:
            # If none of the parsers succeed...
            return None


@parser_factory
async def either(*parsers: ParserFactory[Eff, Resp, T]) -> ParserCoro[Eff, Resp, T]:
    return await Either(parsers)


def run_parser(parser_factory: ParserFactory[Eff, Resp, T], txt: str) -> PRes[T]:
    parser = parser_factory.make()

    send_value: Optional[Resp] = None

    while True:
        try:
            got = parser.send(send_value)
        except StopIteration as e:
            return (txt, e.value)

        if isinstance(got, Effect):
            res = got.perform(txt)
            if res:
                txt, send_value = res
            else:
                return None
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
async def nat() -> int:
    digits = await take_while(lambda ch: ch.isdigit())
    return int(digits)
