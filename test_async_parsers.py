from async_parsers import (
    either,
    exactly,
    nat,
    parser_factory,
    run_parser,
    separated_nonempty_list,
)


@parser_factory
async def list_of_xs():
    await exactly("[")
    xs = await separated_nonempty_list(item=exactly("X"), sep=exactly(", "))
    await exactly("]")
    return xs


p = list_of_xs()
assert run_parser(p, "[X]") == ("", ["X"])
assert run_parser(p, "[X, X, X]") == ("", ["X", "X", "X"])


@parser_factory
async def n_bangs():
    count = await nat()
    for _ in range(count):
        await exactly("!")
    return count


assert run_parser(n_bangs(), "12!!!!!!!!!!!!") == ("", 12)
assert run_parser(n_bangs(), "3!") == None


@parser_factory
async def separated_pair(first, sep, second):
    fst = await first
    await sep
    snd = await second
    return (fst, snd)


@parser_factory
async def delimited(left, target, right):
    await left
    value = await target
    await right
    return value


pair_parser = delimited(
    exactly("("),
    separated_pair(exactly("1"), exactly(", "), exactly("2")),
    exactly(")"),
)

assert run_parser(pair_parser, "(1, 2)") == ("", ("1", "2"))


p = either(exactly("foo"), exactly("bar"), exactly("baz"))
assert run_parser(p, "foo") == ("", "foo")
assert run_parser(p, "bar") == ("", "bar")
assert run_parser(p, "baz") == ("", "baz")
assert run_parser(p, "qux") == None


print("✅ All tests passed.")
