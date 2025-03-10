from dad.types import Matcher


def load_roma_matcher() -> Matcher:
    from romatch import roma_outdoor

    roma_matcher = roma_outdoor("cuda")
    roma_matcher.symmetric = False
    return roma_matcher
