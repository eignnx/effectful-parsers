# effectful-parsers

A Python 3 parser-combinator library which uses coroutines to sequence parsers.

**WORK IN PROGRESS!**

## The Idea

A coroutine is defined by it's runtime. If you pass a coroutine to an event loop, it has the ability to perform asynchronous IO operations. This library provides a runtime that knows how to parse text as guided by the coroutine.

The API for this library is heavily inspired by [Rust's `nom` library](https://github.com/Geal/nom).

## Example

### Parsing Number Pair

In this example, `pair_parser` can parse a pair of numbers, like `(123, 987)`.

```python
from effectful_parsers import parser_factory, exactly, py_int, run_parser, Success

@parser_factory
async def pair_parser():
    await exactly("(")
    first = await py_int()
    await exactly(", ")
    second = await py_int()
    await exactly(")")
    return (first, second)

assert run_parser(pair_parser, "(1, 2)") == Success((1, 2))
```

More general parsers can be built by accepting sub-parsers as arguments. Here's another way that `pair_parser` could have been implemented:

```python
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
    exactly("("), separated_pair(py_int(), exactly(", "), py_int()), exactly(")"),
)
```

### Parse Number!

This parser will first parse a number, then attempts to parse that number of exclamation points.

```python
from effectful_parsers import parser_factory, py_int, exactly, run_parser, Success

@parser_factory
async def n_bangs():
    count = await py_int()
    for _ in range(count):
        await exactly("!")
    return count

assert run_parser(n_bangs(), "12!!!!!!!!!!!!") == Success(12)
assert run_parser(n_bangs(), "3!").failed
```

### Combining Parsers

You can combine parsers with the `either`, `preceded`, and `terminated` functions:

```python
from effectful_parsers import either, exactly, preceded, terminated, run_parser, Success

abc = either(exactly("A"), exactly("B"), exactly("C"))
bracketed_abc = preceded(exactly("["), terminated(abc, exactly("]")))

assert run_parser(bracketed_abc, "[A]") == Success("A")
assert run_parser(bracketed_abc, "[B]") == Success("C")
assert run_parser(bracketed_abc, "[C]") == Success("C")
```

There is also short-hand syntax for those three operations:

- `either(A, B) == A | B`
- `preceded(A, B) == A >> B` (Mnemonic: the result of `B` is being kept)
- `terminated(A, B) == A << B` (Mnemonic: the result of `A` is being kept)

So the previous example could be rewritten as:

```python
abc = exactly("A") | exactly("B") | exactly("C")
bracketed_abc = exactly("[") >> abc << exactly("]")
```

### Many

The `many` and `many1` combinators repeatedly try to use the parser they're provided and return a list or results:

```python
from effectful_parsers import many, exactly, run_parser, Success

list_of_xs = exactly("[") >> many(exactly("X")) << exactly("]")

assert run_parser(list_of_xs, "[]") == Success("")
assert run_parser(list_of_xs, "[X]") == Success("X")
assert run_parser(list_of_xs, "[XXXX]") == Success("XXXX")
```

Note that `many` parses **0 or more** repetitions of its parser, while `many1` parses **1 or more** repetitions.

### Mapping Results

Every `Combinator` has a `map` method which accepts a function of one argument. If the combinator successfully parses a result, then this function will be applied to that result, and that mapped value will be returned instead.

```python
from effectful_parsers import py_int, run_parser, Success

parser = py_int().map(lambda x: x * x).map(lambda y: y + 1)
assert run_parser(parser, "12") == Success((12 * 12) + 1)
assert run_parser(parser, "asdfasdf").failed
```

### Recognize

The `recognize` combinator ignores the result of the parser its given and instead returns the **text** that was parsed:

```python
from effectful_parsers import exactly, py_float, recognize, run_parser, Success

parser = exactly("@@@") >> py_float() << exactly("@@@")
assert run_parser(parser, "@@@-12.4e5@@@") == Success(-12.4e5)
assert run_parser(recognize(parser), "@@@-12.4e5@@@") == Success("@@@-12.4e5@@@")
```
