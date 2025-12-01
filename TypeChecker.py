from abc import abstractmethod
from pathlib import Path
from typing import Optional

from AST import *
from Exceptions import *

class Environment:
    def __init__(self, parent: Optional['Environment'] = None) -> None:
        self.parent = parent
        self.types: dict[str, Type] = {}
        self.functions: dict[str, tuple[list[Type], Type]] = {}

    def lookup_type(self, name: str) -> Optional['Type']:
        if name in self.types:
            return self.types[name]
        elif self.parent is not None:
            return self.parent.lookup_type(name)
        else:
            return None
    
    def lookup_function(self, name: str) -> Optional[tuple[list['Type'], 'Type']]:
        if name in self.functions:
            return self.functions[name]
        elif self.parent is not None:
            return self.parent.lookup_function(name)
        else:
            return None

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

        # STATEMENTS
        elif isinstance(node, FunctionStatement):
            ty = self.__type_from_literal(node.return_type)
            param_tys = [self.__type_from_literal(t.value_type) for t in node.params]
            self.env.functions[node.name.value] = (param_tys, ty)
            self.__assert_type_is(node.body, ty) # TODO: check returns
        elif isinstance(node, LetStatement):
            ty = self.__type_from_literal(node.value_type)
            self.env.types[node.name.value] = ty
            self.__assert_type_is(node.value, ty)
        elif isinstance(node, ExpressionStatement):
            self.type_check(node.expr)

        # EXPRESSIONS
        elif isinstance(node, BlockExpression):
            prev_env = self.env
            self.env = Environment(self.env)

            for stmt in node.statements:
                self.type_check(stmt)

            if node.return_expression is None:
                typ = Void()
            else:
                typ = self.type_check(node.return_expression)
            
            self.env = prev_env
            return typ
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
        elif isinstance(node, CallExpression):
            name = node.function.value
            if name == "printf":
                return I32()

            fn = self.env.lookup_function(name)
            if fn is None:
                raise FunctionNotFoundError(f"function `{name}` doesn't exist")
            (param_types, return_type) = fn
            if len(param_types) != len(node.args):
                raise IncorrectArgumentCountError(f"expected {len(node.args)} arguments in function `{name}`, got {len(param_types)}")

            for param, expected_ty in zip(node.args, param_types):
                self.__assert_type_is(param, expected_ty)

            return return_type
        
        # LITERALS
        elif isinstance(node, I32Literal):
            return I32()
        elif isinstance(node, F32Literal):
            return F32()
        elif isinstance(node, StringLiteral):
            return Str()
        elif isinstance(node, IdentifierLiteral):
            var = self.env.lookup_type(node.value)
            if var is None:
                raise VariableNotFoundError(f"variable `{node.value}` not found")
            return var
        else:
            raise NotImplementedError(f"type checking for {node.type().value} not implemented")
    
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
        else:
            t = self.env.lookup_type(ty)
            if t is None:
                raise TypeNotFoundError(f"couldn't find type `{ty}`")
            else:
                return t
        