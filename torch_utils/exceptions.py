from typing import Any


__all__ = ["MissingAttrError", "insure_type"]


class MissingAttrError(Exception):
    pass 

def insure_type(varname: Any, value: Any, Type: type):
    if not isinstance(value, Type):
        raise TypeError(f"Expected type for '{varname}' attribute is '{Type.__name__}', got '{type(value).__name__}'")
    

