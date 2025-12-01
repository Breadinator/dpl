from abc import abstractmethod
from pathlib import Path
from typing import Optional
import logging

from AST import *
from Exceptions import *

class Environment:
    def __init__(self) -> None:
        logging.warning("environment not implemented")
        self.types: dict[str, Type] = {}
        pass

class Type:
    @property
    @abstractmethod
    def name(self) -> str:
        ...
    
    def __eq__(self, value: object) -> bool:
        if type(self) != type(value):
            return False
        
        if isinstance(self, StructType):
            raise NotImplementedError
        elif isinstance(self, EnumType):
            raise NotImplementedError
        else:
            return self.name == value.name # type: ignore

class Void(Type):
    @property
    def name(self) -> str:
        return "void"

class I32(Type):
    @property
    def name(self) -> str:
        return "i32"
    
class F32(Type):
    @property
    def name(self) -> str:
        return "f32"

class Str(Type):
    @property
    def name(self) -> str:
        return "str"
    
class StructType(Type):
    @property
    def name(self) -> str:
        raise NotImplementedError

class EnumType(Type):
    @property
    def name(self) -> str:
        raise NotImplementedError

TYPE_MAP: dict[str, Type] = {
    "void": Void(),
    "i32": I32(),
    "f32": F32(),
    "str": Str(),
}

class TypeChecker:
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.modules: dict[str, Program] = {}
        self.env = Environment()

    def type_check(self, node: Node) -> Optional[Type]:
        if isinstance(node, ImportStatement):
            raise NotImplementedError("import not implemented")
        elif isinstance(node, Program):
            for child in node.statements:
                self.type_check(child)
        elif isinstance(node, FunctionStatement):
            ty = self.__type_from_literal(node.return_type)
            self.__assert_type_is(node.body, ty) # TODO: check early returns
        elif isinstance(node, LetStatement):
            ty = self.__type_from_literal(node.value_type)
            self.__assert_type_is(node.value, ty)
        elif isinstance(node, ExpressionStatement):
            self.type_check(node.expr)
        elif isinstance(node, BlockExpression):
            for stmt in node.statements:
                self.type_check(stmt)
            if node.return_expression is None:
                return Void()
            else:
                return self.type_check(node.return_expression)
        elif isinstance(node, IfExpression):
            ty = self.__assert_type_check(node.consequence)
            ty2 = self.type_check(node.alternative) if node.alternative else None
            if isinstance(ty, Void):
                return ty
            if ty2 is None and not isinstance(ty, Void):
                raise BranchMismatchError(f"{ty.name} is not void")
            if ty != ty2 and ty2 is not None:
                raise BranchMismatchError(f"{ty.name} not equal to {ty2.name}")
            return ty
        elif isinstance(node, I32Literal):
            return I32()
        elif isinstance(node, StringLiteral):
            return Str()
        else:
            logging.warning(f"type checking for {node.type().value} not implemented")
    
    def __assert_type_check(self, x: Node) -> Type:
        ty = self.type_check(x)
        if ty is None:
            raise CouldntInferTypeError(f"{x.type().value}'s type couldn't be inferred")
        return ty

    def __assert_type_is(self, x: Expression, ty: Type):
        ty2 = self.__assert_type_check(x)
        if ty != ty2:
            raise MismatchedTypesError(f"expected {ty.name}, got {ty2.name}")
    
    def __type_from_literal(self, ty: str) -> Type:
        if ty in TYPE_MAP:
            return TYPE_MAP[ty]
        elif ty in self.env.types:
            return self.env.types[ty]
        else:
            raise TypeNotFoundError(f"couldn't find type `{ty}`")
        