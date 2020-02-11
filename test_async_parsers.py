from typing import List, Tuple, TypeVar

from async_parsers import (
    ParserThunk,
    Success,
    Failure,
    either,
    exactly,
    nat,
    parser_factory,
    run_parser,
    separated_nonempty_list,
)

T = TypeVar("T")


@parser_factory
async def list_of_xs() -> List[str]:
    await exactly("[")
    xs = await separated_nonempty_list(item=exactly("X"), sep=exactly(", "))
    await exactly("]")
    return xs


p = list_of_xs()
assert run_parser(p, "[X]") == Success(["X"])
assert run_parser(p, "[X, X, X]") == Success(["X", "X", "X"])


@parser_factory
async def n_bangs():
    count = await nat()
    for _ in range(count):
        await exactly("!")
    return count


assert run_parser(n_bangs(), "12!!!!!!!!!!!!") == Success(12)
assert run_parser(n_bangs(), "3!").failed
print(run_parser(n_bangs(), "3!"))

A = TypeVar("A")
B = TypeVar("B")


@parser_factory
async def separated_pair(
    first: ParserThunk[A], sep, second: ParserThunk[B]
) -> Tuple[A, B]:
    fst = await first
    await sep
    snd = await second
    return (fst, snd)


@parser_factory
async def delimited(left, target: ParserThunk[T], right) -> T:
    await left
    value = await target
    await right
    return value


pair_parser = delimited(
    exactly("("),
    separated_pair(exactly("1"), exactly(", "), exactly("2")),
    exactly(")"),
)

assert run_parser(pair_parser, "(1, 2)") == Success(("1", "2"))


foo_bar_or_baz = either(exactly("foo"), exactly("bar"), exactly("baz"))
assert run_parser(foo_bar_or_baz, "foo") == Success("foo")
assert run_parser(foo_bar_or_baz, "bar") == Success("bar")
assert run_parser(foo_bar_or_baz, "baz") == Success("baz")
assert run_parser(foo_bar_or_baz, "qux").failed
print(run_parser(foo_bar_or_baz, "qux"))


print("✅ All tests passed.")
