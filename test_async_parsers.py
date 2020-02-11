from typing import List, Tuple, TypeVar, cast
import math

from async_parsers import (
    Failure,
    ParserThunk,
    Success,
    either,
    exactly,
    matches,
    optional,
    parser_factory,
    preceded,
    py_float,
    py_int,
    recognize,
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
assert run_parser(p, "[X, Z, X]").failed
print(run_parser(p, "[X, Z, X]"))


@parser_factory
async def n_bangs():
    count = await py_int()
    for _ in range(count):
        await exactly("!")
    return count


assert run_parser(n_bangs(), "12!!!!!!!!!!!!") == Success(12)
assert run_parser(n_bangs(), "3!").failed
print(run_parser(n_bangs(), "3!"))


assert run_parser(py_int(), "123").succeeded
assert run_parser(py_int(), "asdf").failed
print(run_parser(py_int(), "asdf"))


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
    exactly("("), separated_pair(py_int(), exactly(", "), py_int()), exactly(")"),
)

assert run_parser(pair_parser, "(1, 2)") == Success((1, 2))
assert run_parser(pair_parser, "(1, 2").failed
print(run_parser(pair_parser, "(1, 2"))
assert run_parser(pair_parser, "(1, x)").failed
print(run_parser(pair_parser, "(1, x)"))


foo_bar_or_baz = either(exactly("foo"), exactly("bar"), exactly("baz"))
assert run_parser(foo_bar_or_baz, "foo") == Success("foo")
assert run_parser(foo_bar_or_baz, "bar") == Success("bar")
assert run_parser(foo_bar_or_baz, "baz") == Success("baz")
assert run_parser(foo_bar_or_baz, "qux").failed
print(run_parser(foo_bar_or_baz, "qux"))


@parser_factory
async def foo_parser():
    await exactly("<")
    foo = await matches(r"foo+")
    await exactly(">")
    return foo


assert run_parser(foo_parser(), "<fooooo>").succeeded
assert run_parser(foo_parser(), "<foxooo>").failed
print(run_parser(foo_parser(), "<foxooo>"))
assert run_parser(foo_parser(), "<fooxoo>").failed
print(run_parser(foo_parser(), "<fooxoo>"))


assert run_parser(py_int(), "-123") == Success(-123)
assert run_parser(py_int(), "0") == Success(0)
assert run_parser(py_int(), "-0") == Success(0)
assert run_parser(py_int(), "-").failed

assert run_parser(py_float(), "-123.456") == Success(-123.456)
assert run_parser(py_float(), "-0.456") == Success(-0.456)
assert run_parser(py_float(), "-.456") == Success(-0.456)
assert run_parser(py_float(), "-123.") == Success(-123.0)
assert run_parser(py_float(), "123.456") == Success(123.456)
assert run_parser(py_float(), "0.456") == Success(0.456)
assert run_parser(py_float(), ".456") == Success(0.456)
assert run_parser(py_float(), "123.") == Success(123.0)
assert run_parser(py_float(), "123.4e5") == Success(123.4e5)
assert run_parser(py_float(), ".4e5") == Success(0.4e5)
assert run_parser(py_float(), "1_2_3.4e5") == Success(1_2_3.4e5)
assert run_parser(py_float(), "1_2_3.4_4e1_1") == Success(1_2_3.4_4e1_1)
assert run_parser(py_float(), "123").failed
for x in ["inf", "-inf", "Infinity", "-Infinity"]:
    assert run_parser(py_float(allow_special_values=True), x) == Success(float(x))
assert math.isnan(
    cast(Success, run_parser(py_float(allow_special_values=True), "nan")).parsed
)
assert math.isnan(
    cast(Success, run_parser(py_float(allow_special_values=True), "-nan")).parsed
)


assert run_parser(optional(exactly("foo")), "foo") == Success("foo")
assert run_parser(optional(exactly("foo")), "bar") == Success(None, rest="bar")
assert run_parser(optional(exactly("foo"), default=-1), "bar") == Success(
    -1, rest="bar"
)

assert run_parser(py_int().map(lambda x: x * x), "12") == Success(144)
assert run_parser(py_int().map(lambda x: x * x), "x").failed
print(run_parser(py_int().map(lambda x: x * x), "x"))

assert run_parser(recognize(preceded(exactly("#"), py_int())), "#123") == Success(
    "#123"
)

print()
print("--------------------")
print("✅ All tests passed.")
print("--------------------")
