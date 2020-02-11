# effectful-parsers

A Python 3 parser-combinator library which uses coroutines to sequence parsers.

**WORK IN PROGRESS!**

## The Idea

A coroutine is defined by it's runtime. If you pass a coroutine to an event loop, it has the ability to perform asynchronous IO operations. This library provides a runtime that knows how to parse text as guided by the coroutine.

## Example

### Parsing Number Pair

In this example, `pair_parser` can parse a pair of numbers, like `(123, 987)`.

```python
from async_parsers import parser_factory, exactly, py_int, run_parser, Success

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
from async_parsers import parser_factory, py_int, exactly, run_parser, Success

@parser_factory
async def n_bangs():
    count = await py_int()
    for _ in range(count):
        await exactly("!")
    return count

assert run_parser(n_bangs(), "12!!!!!!!!!!!!") == Success(12)
assert run_parser(n_bangs(), "3!").failed
```
