from llvmlite import ir # pyright: ignore[reportMissingTypeStubs]
from typing import Optional

class Environment:
    def __init__(self, records: Optional[dict[str, tuple[ir.Value, ir.Type]]] = None, parent: Optional['Environment'] = None, name: str = "global") -> None:
        self.records = records if records is not None else {}
        self.parent = parent
        self.name = name
    
    def define(self, name: str, value: ir.Value, typ: ir.Type) -> ir.Value:
        self.records[name] = (value, typ)
        return value
    
    def lookup(self, name: str) -> tuple[Optional[ir.Value], Optional[ir.Type]]:
        return self.__resolve(name)

    def __resolve(self, name: str) -> tuple[Optional[ir.Value], Optional[ir.Type]]:
        if name in self.records:
            return self.records[name]
        elif self.parent:
            return self.parent.__resolve(name)
        else:
            return None, None