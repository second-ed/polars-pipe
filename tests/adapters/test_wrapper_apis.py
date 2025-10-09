import inspect

import pytest

from src.polars_pipe.adapters.io_pl import FakeIOWrapper, IOBase, IOWrapper


@pytest.mark.parametrize(
    ("obj", "protocol"),
    [
        pytest.param(IOWrapper(), IOBase),
        pytest.param(FakeIOWrapper(), IOBase),
    ],
)
def test_protocols(obj, protocol):
    assert isinstance(obj, protocol)


class SanityCheck:
    def method_a(self, a: int, b: float) -> float:
        return a * b


class FakeSanityCheck:
    def method_a(self, a: int, b: float) -> float:
        return a * b

    def some_test_helper_method(self, *, c: bool) -> bool:
        return c


class FakeMissingMethod:
    pass


class FakeMismatchingSignature:
    def method_a(self, a: int) -> int:
        return a


@pytest.mark.parametrize(
    ("real", "fake"),
    [
        pytest.param(
            SanityCheck(),
            FakeSanityCheck(),
            id="ensure matching public methods pass",
        ),
        pytest.param(
            SanityCheck(),
            FakeMissingMethod(),
            id="ensure fails if fake missing method",
            marks=pytest.mark.xfail(
                reason="ensure fails if fake missing method",
                strict=True,
            ),
        ),
        pytest.param(
            SanityCheck(),
            FakeMismatchingSignature(),
            id="ensure fails if fake not matching signature",
            marks=pytest.mark.xfail(
                reason="ensure fails if fake not matching signature",
                strict=True,
            ),
        ),
        pytest.param(
            IOWrapper(),
            FakeIOWrapper(),
            id="ensure IO wrapper matches fake",
        ),
        pytest.param(
            IOWrapper,
            IOBase,
            id="ensure IO wrapper matches protocol",
        ),
    ],
)
def test_api_match(real: object, fake: object) -> None:
    def get_methods(obj: object) -> dict[str, inspect.Signature]:
        return {
            name: inspect.signature(fn)
            for name, fn in inspect.getmembers(obj, inspect.isroutine)
            if not (name.startswith("__") and name.endswith("__"))
            and not name.startswith("_")  # ignore private methods
        }

    real_methods = get_methods(real)
    fake_methods = get_methods(fake)

    # all methods in the real are in the fake
    assert set(real_methods) - set(fake_methods) == set()

    # all methods in the real have the same signature as those in the fake
    mismatches = [
        {"method": key, "real": method, "fake": fake_methods[key]}
        for key, method in real_methods.items()
        if fake_methods[key] != method
    ]
    assert mismatches == []
