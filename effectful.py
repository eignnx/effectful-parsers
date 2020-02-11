from dataclasses import dataclass
from typing import TypeVar, List, Callable, Optional, Tuple, Generator, Generic, Union, Any, cast
from types import GeneratorType

T = TypeVar("T")
U = TypeVar("U")
Effect = TypeVar("Effect")
SendType = TypeVar("SendType")

PRes = Optional[Tuple[str, T]]
Parser = Generator[Any, Any, PRes[T]]

@dataclass
class StartsWith:
    sep: str


def exactly(target: str) -> Generator[StartsWith, str, PRes[str]]:
    def parser():
        found = yield StartsWith(target)
        return found

    return parser()


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def separated_pair(
    first: Parser[A],
    sep: Parser[B],
    second: Parser[C]
) -> Parser[Tuple[A, C]]:

    def new_parser():
        fst = yield first
        _ = yield sep
        snd = yield second
        return (fst, snd)

    return new_parser()


class Alt(Generic[T]):
    def __init__(self, *parsers: Parser[T]):
        self.parsers = parsers

    def __repr__(self):
        joined = " ,".join((repr(p) for p in self.parsers))
        return f"Alt({joined})"
    

def run_parser(parser: Generator[Union[StartsWith, Parser[U], Any], Any, PRes[T]], txt: str) -> PRes[T]:
    send_value = None

    while True:
        try:
            got = parser.send(send_value)
            print("GOT", got)
        except StopIteration as e:
            return (txt, e.value)

        if isinstance(got, StartsWith):
            starts_with_request = got
            err, parsed, rest = txt.partition(starts_with_request.sep)
            if err:
                return None
            else:
                txt = rest
                send_value = parsed
                continue

        elif isinstance(got, Alt):
            sub_parsers = got.parsers
            for sub_parser in sub_parsers:
                res = run_parser(sub_parser, txt) # Must be a recursive call.
                if res:
                    rest, parsed = res
                    txt = rest,
                    send_value = parsed
                    break
            else: # If all sub-parsers failed, entire parse fails.
                return None

        elif isinstance(got, GeneratorType):
            sub_parser = cast(Parser[Any], got)
            res = run_parser(sub_parser, txt)  # Must be a recursive call.
            if res:
                rest, parsed = res
                txt = rest
                send_value = parsed
                continue
            else:
                return None

        else:
            raise Exception(f"Unknown effect message: {got}")



if __name__ == '__main__':


    # p: Parser[Tuple[str, str]] = separated_pair(exactly("foo"), exactly("-"), exactly("bar"))
    # print("FINAL PARSED VALUE:", run_parser(p, "foo-bar"))

    # def a_or_b():
    #     x = yield Alt(exactly("A"), exactly("B"))
    #     return x

    # p: Parser[Tuple[str, str]] = a_or_b()
    # print("FINAL PARSED VALUE:", run_parser(p, "V"))

    # def list_of_ones():
    #     ones = []
    #     yield exactly("[")
    #     yield Try(exactly("1"), lambda first: ones.append(first))
    #     async with While():
    #         res = yield Try(exactly(", "))
    #         if res is None:
    #             break
    #         else:
    #             one = yield exactly("1")

    #     yield exactly("]")


    ####################################
    # PROCEDURE:
    ####################################
    # txt = "foo-bar"
    # foo = p.send(None)
    # starts_with_foo = foo.send(None)
    # txt = "-bar"
    # try:
    #     foo.send("foo")
    #     raise Exception("BAD")
    # except StopIteration as e:
    #     res = e.value

    # dash = p.send(res)
    # starts_with_dash = dash.send(None)
    # txt = "bar"
    # try:
    #     dash.send("-")
    #     raise Exception("BAD")
    # except StopIteration as e:
    #     res = e.value

    # bar = p.send(res)
    # starts_with_bar = bar.send(None)
    # txt = ""
    # try:
    #     bar.send("bar")
    #     raise Exception("BAD")
    # except StopIteration as e:
    #     res = e.value

    # try:
    #     p.send(res)
    #     raise Exception("BAD")
    # except StopIteration as e:
    #     res = e.value

    # print(res)
