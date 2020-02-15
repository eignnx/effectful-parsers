import math
from typing import List, Tuple, TypeVar, cast

from effectful_parsers import (
    Combinator,
    Failure,
    Success,
    either,
    exactly,
    fail,
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

errors_so_far = 0


def test(parser, input, expected, print_failure=False):
    try:
        actual = run_parser(parser, input)
        if expected is Failure:
            assert actual.failed
            if print_failure:
                print(
                    f"GOOD PARSE FAILURE ON run_parser({parser}, {input!r}) BECAUSE ```"
                )
                print(actual)
                print("```")
        elif expected is Success:
            assert actual.succeeded
        else:
            assert actual == expected
    except AssertionError:
        global errors_so_far
        errors_so_far += 1
        print()
        print(f"~~~~ASSERTION ERROR #{errors_so_far}~~~~")
        print(f"run_parser({parser}, {input!r}) != {expected}")
        print(f"PARSE RESULT:```")
        print(actual)
        print("```")
        print()


@parser_factory
async def list_of_xs() -> List[str]:
    await exactly("[")
    xs = await separated_nonempty_list(item=exactly("X"), sep=exactly(", "))
    await exactly("]")
    return xs


p = list_of_xs()
test(p, "[X]", Success(["X"]))
test(p, "[X, X, X]", Success(["X", "X", "X"]))
test(p, "[X, Z, X]", Failure)


@parser_factory
async def n_bangs():
    count = await py_int()
    for _ in range(count):
        await exactly("!")
    return count


test(n_bangs(), "12!!!!!!!!!!!!", Success(12))
test(n_bangs(), "3!", Failure)


test(py_int(), "123", Success)
test(py_int(), "asdf", Failure)


A = TypeVar("A")
B = TypeVar("B")


@parser_factory
async def separated_pair(
    first: Combinator[A], sep, second: Combinator[B]
) -> Tuple[A, B]:
    fst = await first
    await sep
    snd = await second
    return (fst, snd)


@parser_factory
async def delimited(left, target: Combinator[T], right) -> T:
    await left
    value = await target
    await right
    return value


pair_parser = delimited(
    exactly("("), separated_pair(py_int(), exactly(", "), py_int()), exactly(")"),
)

test(pair_parser, "(1, 2)", Success((1, 2)))
test(pair_parser, "(1, 2", Failure)
test(pair_parser, "(1, x)", Failure)


foo_bar_or_baz = either(exactly("foo"), exactly("bar"), exactly("baz"))
test(foo_bar_or_baz, "foo", Success("foo"))
test(foo_bar_or_baz, "bar", Success("bar"))
test(foo_bar_or_baz, "baz", Success("baz"))
test(foo_bar_or_baz, "qux", Failure, print_failure=True)


@parser_factory
async def foo_parser():
    await exactly("<")
    foo = await matches(r"foo+")
    await exactly(">")
    return foo


test(foo_parser(), "<fooooo>", Success)
test(foo_parser(), "<fo*ooo>", Failure, print_failure=True)
test(foo_parser(), "<foo*oo>", Failure, print_failure=True)


test(py_int(), "-123", Success(-123))
test(py_int(), "0", Success(0))
test(py_int(), "-0", Success(0))
test(py_int(), "-", Failure)

test(py_float(), "-123.456", Success(-123.456))
test(py_float(), "-0.456", Success(-0.456))
test(py_float(), "-.456", Success(-0.456))
test(py_float(), "-123.", Success(-123.0))
test(py_float(), "123.456", Success(123.456))
test(py_float(), "0.456", Success(0.456))
test(py_float(), ".456", Success(0.456))
test(py_float(), "123.", Success(123.0))
test(py_float(), "123.4e5", Success(123.4e5))
test(py_float(), ".4e5", Success(0.4e5))
test(py_float(), "1_2_3.4e5", Success(1_2_3.4e5))
test(py_float(), "1_2_3.4_4e1_1", Success(1_2_3.4_4e1_1))
test(py_float(), "123", Failure)
for x in ["inf", "-inf", "Infinity", "-Infinity"]:
    test(py_float(allow_special_values=True), x, Success(float(x)))

assert math.isnan(
    cast(Success, run_parser(py_float(allow_special_values=True), "nan")).parsed
)
assert math.isnan(
    cast(Success, run_parser(py_float(allow_special_values=True), "-nan")).parsed
)


test(optional(exactly("foo")), "foo", Success("foo"))
test(optional(exactly("foo")), "bar", Success(None, rest="bar"))
test(optional(exactly("foo"), default=-1), "bar", Success(-1, rest="bar"))

test(py_int().map(lambda x: x * x), "12", Success(144))
test(py_int().map(lambda x: x * x), "x", Failure, print_failure=True)

test(recognize(preceded(exactly("#"), py_int())), "#123", Success("#123"))

int_passing_examples = [
    "0",
    "00000",
    "0_0_0_000_0",
    "-00000",
    "1",
    "1234",
    "123_532_25",
    "-1",
    "-0",
    "0x2_3Fac_E",
    "0o45_6701_20",
    "0b_1001_1",
    "-0B_1101",
    "-0O_77171",
    "-0X_DEADBEEF",
]
for example in int_passing_examples:
    test(py_int(), example, Success(eval(example)))


test(exactly("[") >> py_int() << exactly("]"), "[123]", Success(123))
test(exactly("fooooo") | exactly("foo") | exactly("f"), "fooooo", Success("fooooo"))
test(exactly("fooooo") | exactly("foo") | exactly("f").map(ord), "f", Success(ord("f")))
test(py_int().map(lambda x: x * x).map(lambda y: y + 1), "12", Success(12 * 12 + 1))


@parser_factory
async def square_integer():
    n = await py_int()
    if n < 0 or math.sqrt(n) * math.sqrt(n) != n:
        await fail(f"The integer {n} is not a square.")
    else:
        return n


test(square_integer(), "5", Failure)
test(square_integer(), "16", Success(16))
test(square_integer(), "-1", Failure, print_failure=True)
test(square_integer(), "144", Success(144))

if errors_so_far == 0:
    print()
    print("--------------------")
    print("✅ All tests passed.")
    print("--------------------")
else:
    print()
    print("!!!!!!!!!!!!!!!!!!!!")
    print(f"❌ {errors_so_far} tests failed!")
    print("!!!!!!!!!!!!!!!!!!!!")
