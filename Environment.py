from llvmlite import ir # pyright: ignore[reportMissingTypeStubs]
from typing import Optional
from dataclasses import dataclass

@dataclass
class EnumMetadata:
    name: str
    variants: list[str]

class Environment:
    def __init__(
        self, 
        records: Optional[dict[str, tuple[ir.Value, ir.Type]]] = None,
        structs: Optional[dict[str, tuple[ir.BaseStructType, list[str], list[ir.Type]]]] = None,
        enums: Optional[dict[str, EnumMetadata]] = None,
        parent: Optional['Environment'] = None, 
        name: str = "global"
    ) -> None:
        self.records = records if records is not None else {}
        self.structs = structs if structs is not None else {}
        self.enums = enums if enums is not None else {}
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

    def define_struct(self, name: str, llvm_struct: ir.BaseStructType, field_names: list[str], field_types: list[ir.Type]) -> ir.BaseStructType:
        self.structs[name] = (llvm_struct, field_names, field_types)
        return llvm_struct

    def lookup_struct(self, name: str) -> tuple[Optional[ir.BaseStructType], Optional[list[str]], Optional[list[ir.Type]]]:
        if name in self.structs:
            return self.structs[name]
        elif self.parent:
            return self.parent.lookup_struct(name)
        else:
            return None, None, None
        
    def define_enum(
        self,
        name: str,
        variants: list[str],
    ) -> None:
        self.enums[name] = EnumMetadata(name, variants)
    
    def lookup_enum(self, name: str) -> Optional[EnumMetadata]:
        if name in self.enums:
            return self.enums[name]
        elif self.parent:
            return self.parent.lookup_enum(name)
        else:
            return None