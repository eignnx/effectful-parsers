from dataclasses import dataclass
from typing import List

from effectful_parsers.lib import (
    Combinator,
    Eff,
    Resp,
    T,
    U,
    either,
    exactly,
    many,
    matches,
    parser_factory,
    recognize,
)


@parser_factory
async def many1(parser: Combinator[T]) -> List[T]:
    first = await parser
    rest = await many(parser)
    return [first] + rest


@parser_factory
async def sequence(*parsers: Combinator) -> List:
    result = []
    for parser in parsers:
        result.append(await parser)
    return result


@parser_factory
async def separated_nonempty_list(item: Combinator[T], sep: Combinator) -> List[T]:
    first = await item
    rest = await many(sep >> item)
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
    class ToBase:
        n: int

        def __call__(self, digits: str) -> int:
            return int(digits.lstrip("_"), self.n)

        def __repr__(self):
            return f"to_base({self.n})"

    dec_integer = matches(r"[1-9](_?[0-9])*|0+(_?0)*").map(ToBase(10))
    hex_integer = matches(r"0[xX]") >> matches(r"(_?[0-9a-fA-F])+").map(ToBase(16))

    oct_integer = matches(r"0[oO]") >> matches(r"(_?[0-7])+").map(ToBase(8))
    bin_integer = matches(r"0[bB]") >> matches(r"(_?[01])+").map(ToBase(2))

    sign = await matches(r"[+-]?").map(lambda s: -1 if s == "-" else 1)
    return await (hex_integer | oct_integer | bin_integer | dec_integer).map(
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

    point_float = (
        (digits >> exactly(".") >> digits)
        | (exactly(".") >> digits)
        | (digits << exactly("."))
    )

    @parser_factory
    async def exponent():
        await matches(r"[eE]")
        await matches(r"[+-]?")
        await digits

    exponent_float = sequence(point_float | digits, exponent())

    special_values = (
        [exactly("nan"), exactly("inf"), exactly("Infinity")]
        if allow_special_values
        else []
    )

    @parser_factory
    async def py_float_format():
        await matches(r"[+-]?")
        await either(exponent_float, point_float, *special_values)

    float_text = await recognize(py_float_format())
    return float(float_text)
