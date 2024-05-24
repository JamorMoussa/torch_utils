from .exceptions import insure_type, MissingAttrError

from typing import Any


__all__ = ["ConfigsBase", ]


class ConfigsBase:

    types: list[type] = None

    def __setattr__(self, name: str, value: Any) -> None:
        
        if self.types == None:
            raise MissingAttrError(
                """A 'types' attributes must be specified. Which is a list of types that config's attributes must match.\nFor example:\n\ttypes: list[type] = field(default=(int, int, int, bool, torch.device))""")

        super().__setattr__(name, value)
        for index, attr in enumerate(self.__dict__):
            if attr != "types": insure_type(varname= attr, value=self.__dict__[attr], Type= self.types[index])
