from typing import Any


__all__ = ["MissingAttrError", "insure_type"]


class MissingAttrError(Exception):
    pass 

def insure_type(varname: Any, Type: type):
    if not isinstance(varname, Type):
        raise TypeError(f"Expected '{Type.__name__}' for integer_field, got '{type(varname).__name__}'")
    

