from llvmlite import ir # pyright: ignore[reportMissingTypeStubs]
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class EnumMetadata:
    name: str
    variants: list[str]

@dataclass
class UnionMetadata:
    name: str
    variant_names: list[str]
    variant_types: list[Optional[ir.Type]]
    llvm_struct: ir.BaseStructType

@dataclass
class Environment:
    records: dict[str, tuple[ir.Value, ir.Type]] = field(default_factory=lambda: {})
    structs: dict[str, tuple[ir.BaseStructType, list[str], list[ir.Type]]] = field(default_factory=lambda: {})
    enums: dict[str, EnumMetadata] = field(default_factory=lambda: {})
    unions: dict[str, UnionMetadata] = field(default_factory=lambda: {})
    parent: Optional['Environment'] = field(default=None)
    name: str = field(default_factory=lambda: "global")
    
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
        
    def define_union(
        self,
        name: str,
        variant_names: list[str],
        variant_types: list[Optional[ir.Type]],
        llvm_struct: ir.BaseStructType,
    ) -> UnionMetadata:
        self.unions[name] = UnionMetadata(
            name,
            variant_names,
            variant_types,
            llvm_struct,
        )
        return self.unions[name]
    
    def lookup_union(self, name: str) -> Optional[UnionMetadata]:
        if name in self.unions:
            return self.unions[name]
        elif self.parent:
            return self.parent.lookup_union(name)
        else:
            return None