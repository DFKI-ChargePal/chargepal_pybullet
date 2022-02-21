import sys
from typing import (overload, Literal)

ENV_LIST = [
    'foo1',
    'foo2',
]


@overload
def make(id_: Literal['foo1']) -> None: ...


@overload
def make(id_: Literal['foo2']) -> None: ...


@overload
def make(id_: str) -> None: ...


def make(id_: str) -> None:
    if id_ not in ENV_LIST:
        sys.exit(f'Environment with id >> {id_} << not found!')
    else:
        print(f'Create {id_}')


make('foo1')
make('not_listed')
