from typing import Generic, TypeVar, List, Optional, Callable, Tuple, cast

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


# def and_then(x: Optional[T], f: Callable[[T], Optional[U]]) -> Optional[U]:
#     """The monadic bind operation."""
#     if x is None:
#         return None
#     else:
#         return f(x)


PRes = Optional[Tuple[str, T]]
Parser = Callable[[str], PRes[T]]


def exactly(target: str) -> Parser[str]:
    def p(txt: str):
        err, parsed, rest = txt.partition(target)
        if not err:
            return (rest, parsed)
        else:
            None

    return p


def many(p: Parser[T]) -> Parser[List[T]]:
    def new_p(txt: str):
        collected: List[T] = []
        while True:
            res = p(txt)
            if res is None:
                return (txt, collected)
            else:
                (rest, parsed) = res
                collected.append(parsed)
                txt = rest

    return new_p


def alt(*ps: Parser[T]) -> Parser[T]:
    def new_p(txt: str) -> PRes[T]:
        for p in ps:
            res = p(txt)
            if res is None:
                continue
            else:
                return res
        return None

    return new_p


def and_then(p: Parser[T], f: Callable[[T], Parser[U]]) -> Parser[U]:
    def new_p(txt: str) -> PRes[U]:
        res = p(txt)
        if res is None:
            return None
        else:
            (rest, parsed) = res
            return f(parsed)(rest)

    return new_p


def lift(x: T) -> Parser[T]:
    """Consumes no input."""
    def new_p(txt: str):
        return (txt, x)

    return new_p


def separated_pair(first: Parser[T], sep: Parser[U], second: Parser[V]) -> Parser[Tuple[T, V]]:
    return (
        and_then(first,  lambda fst:
        and_then(sep,    lambda _:
        and_then(second, lambda snd:
        lift((fst, snd))
    ))))

def delimited(left: Parser[T], target: Parser[U], right: Parser[V]) -> Parser[U]:
    return (
        and_then(left,   lambda _:
        and_then(target, lambda tgt:
        and_then(right,  lambda _:
        lift(tgt)
    ))))


def many_joined(p: Parser[str]) -> Parser[str]:
    return (
        and_then(many(p), lambda results:
        lift("".join(results))
    ))


def matches_char(pred: Callable[[str], bool]) -> Parser[str]:
    return lambda txt: (txt[1:], txt[0]) if txt and pred(txt[0]) else None


word = many_joined(matches_char(lambda ch: ch.isalpha()))
python_identifier: Parser[str] = (
    and_then(matches_char(lambda ch: ch.isalpha() or ch == "_"), lambda fst:
    and_then(many(matches_char(lambda ch: ch.isalnum() or ch == "_")), lambda rest:
    lift("".join([cast(str, fst)] + rest)))))


def separated_nonempty_list(item: Parser[T], sep: Parser[U]) -> Parser[List[T]]:
    sep_item: Parser[T] = (
        and_then(sep, lambda _:
        and_then(item, lambda i:
        lift(i)
    )))

    return (
        and_then(item, lambda first:
        and_then(many(sep_item), lambda rest:
        lift([cast(T, first)] + rest)
    )))
        

if __name__ == "__main__":
    # p: Parser[Tuple[str, str]] = (
    #     and_then(exactly("first"), lambda fst:
    #     and_then(exactly(", "), lambda _:
    #     and_then(exactly("second"), lambda snd:
    #     lift((fst, snd))
    # ))))

    # p = separated_pair(exactly("first"), exactly(", "), exactly("second"))

    p: Parser[Tuple[str, List[str]]] = (
        and_then(python_identifier, lambda head:
        and_then(exactly(" --> "), lambda _:
        and_then(separated_nonempty_list(python_identifier, exactly(", ")), lambda body:
        and_then(exactly(";"), lambda _:
        lift((head, body))
    )))))


    print(p("start --> first, second, third;"))
