from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

class NodeType(Enum):
    Program = "Program"

    # Statements
    ExpressionStatement = "ExpressionStatement"
    LetStatement = "LetStatement"
    ConstStatement = "ConstStatement"
    FunctionStatement = "FunctionStatement"
    ReturnStatement = "ReturnStatement"
    AssignStatement = "AssignStatement"
    WhileStatement = "WhileStatement"
    ForStatement = "ForStatement"
    ContinueStatement = "ContinueStatement"
    BreakStatement = "BreakStatement"
    ImportStatement = "ImportStatement"
    StructStatement = "StructStatement"
    EnumStatement = "EnumStatement"
    UnionStatement = "UnionStatement"

    # Expressions
    InfixExpression = "InfixExpression"
    BlockExpression = "BlockExpression"
    IfExpression = "IfExpression"
    CallExpression = "CallExpression"
    PrefixExpression = "PrefixExpression"
    NewStructExpression = "NewStructExpression"
    FieldAccessExpression = "FieldAccessExpression"
    EnumVariantAccessExpression = "EnumVariantAccessExpression"
    MatchExpression = "MatchExpression"

    # Literals
    I32Literal = "I32Literal"
    F32Literal = "F32Literal"
    IdentifierLiteral = "IdentifierLiteral"
    BooleanLiteral = "BooleanLiteral"
    StringLiteral = "StringLiteral"

    # Helper
    FunctionParameter = "FunctionParameter"

class Node(ABC):
    @abstractmethod
    def type(self) -> NodeType:
        pass

    @abstractmethod
    def json(self) -> dict[str, Any]:
        pass

class Statement(Node):
    pass

class Expression(Node):
    pass

class Program(Node):
    def __init__(self) -> None:
        self.statements: list[Statement] = []

    def type(self) -> NodeType:
        return NodeType.Program
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "statements": [{stmt.type().value: stmt.json()} for stmt in self.statements]
        }

# region Helpers
class FunctionParameter(Expression):
    def __init__(self, name: str, value_type: str) -> None:
        self.name = name
        self.value_type = value_type

    def type(self) -> NodeType:
        return NodeType.FunctionParameter
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name,
            "value_type": self.value_type,
        }
# endregion

# region Statements
class ExpressionStatement(Statement):
    def __init__(self, expr: Expression) -> None:
        self.expr = expr
    
    def type(self) -> NodeType:
        return NodeType.ExpressionStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "expr": self.expr.json()
        }
    
class LetStatement(Statement):
    def __init__(self, name: 'IdentifierLiteral', value: Expression, value_type: str, const: bool = False) -> None:
        self.name = name
        self.value = value
        self.value_type = value_type
        self.const = const

    def type(self) -> NodeType:
        return NodeType.LetStatement

    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name.json(),
            "value": self.value.json(),
            "value_type": self.value_type,
            "const": self.const,
        }
   
class ReturnStatement(Statement):
    def __init__(self, return_value: Expression) -> None:
        self.return_value = return_value
    
    def type(self) -> NodeType:
        return NodeType.ReturnStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "return_value": self.return_value.json(),
        }
    
class FunctionStatement(Statement):
    def __init__(
        self,
        params: list[FunctionParameter],
        body: 'BlockExpression',
        name: 'IdentifierLiteral',
        return_type: str
    ) -> None:
        self.params = params
        self.body = body
        self.name = name
        self.return_type = return_type

    def type(self) -> NodeType:
        return NodeType.FunctionStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name.json(),
            "return_type": self.return_type,
            "parameters": [p.json() for p in self.params],
            "body": self.body.json(),
        }
    
class WhileStatement(Statement):
    def __init__(self, condition: Expression, body: 'BlockExpression') -> None:
        self.condition = condition
        self.body = body
    
    def type(self) -> NodeType:
        return NodeType.WhileStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "condition": self.condition.json(),
            "body": self.body.json(),
        }

class BreakStatement(Statement):
    def __init__(self) -> None:
        pass

    def type(self) -> NodeType:
        return NodeType.BreakStatement
    
    def json(self) -> dict[str, Any]:
        return { "type": self.type().value }

class ContinueStatement(Statement):
    def __init__(self) -> None:
        pass

    def type(self) -> NodeType:
        return NodeType.ContinueStatement
    
    def json(self) -> dict[str, Any]:
        return { "type": self.type().value }
    
class ForStatement(Statement):
    def __init__(
        self,
        var_declaration: LetStatement,
        condition: Expression,
        action: 'AssignExpression',
        body: 'BlockExpression',
    ) -> None:
        self.var_declaration = var_declaration
        self.condition = condition
        self.action = action
        self.body = body

    def type(self) -> NodeType:
        return NodeType.ForStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "var_declaration": self.var_declaration.json(),
            "condition": self.condition.json(),
            "action": self.action.json(),
            "body": self.body.json(),
        }

class ImportStatement(Statement):
    def __init__(self, path: str) -> None:
        self.path = path
    
    def type(self) -> NodeType:
        return NodeType.ImportStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "path": self.path,
        }
    
class StructStatement(Statement):
    def __init__(self, ident: 'IdentifierLiteral', fields: list[tuple[str, str]]) -> None:
        self.ident = ident
        self.fields = fields

    def type(self) -> NodeType:
        return NodeType.StructStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "ident": self.ident.json(),
            "fields": [[field[0], field[1]] for field in self.fields],
        }

class EnumStatement(Statement):
    def __init__(self, name: 'IdentifierLiteral', variants: list['IdentifierLiteral']) -> None:
        self.name = name
        self.variants = variants
    
    def type(self) -> NodeType:
        return NodeType.EnumStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name.json(),
            "variants": [variant.json() for variant in self.variants]
        }

class UnionStatement(Statement):
    def __init__(self, name: 'IdentifierLiteral', variants: list[tuple['IdentifierLiteral', Optional[str]]]) -> None:
        self.name = name
        self.variants = variants

    def type(self) -> NodeType:
        return NodeType.UnionStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name.json(),
            "variants": [[variant[0].json(), variant[1]] for variant in self.variants]
        }
# endregion

# region Expressions
class InfixExpression(Expression):
    def __init__(self, left_node: Expression, operator: str, right_node: Optional[Expression] = None) -> None:
        self.left_node = left_node
        self.operator = operator
        self.right_node = right_node

    def type(self) -> NodeType:
        return NodeType.InfixExpression

    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "left_node": self.left_node.json(),
            "operator": self.operator,
            "right_node": self.right_node.json() if self.right_node is not None else {}
        }
    
class AssignExpression(Expression):
    def __init__(self, lh: Expression, rh: Expression, operator: str = "") -> None:
        self.lh = lh
        self.rh = rh
        self.operator = operator
    
    def type(self) -> NodeType:
        return NodeType.AssignStatement
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "lh": self.lh.json(),
            "rh": self.rh.json(),
            "operator": self.operator,
        }
    
class BlockExpression(Expression):
    def __init__(self) -> None:
        self.statements: list[Statement] = []
        self.return_expression: Optional[Expression] = None
    
    def type(self) -> NodeType:
        return NodeType.BlockExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "statements": [stmt.json() for stmt in self.statements],
            "return_expression": self.return_expression.json() if self.return_expression is not None else {}
        }
    
class IfExpression(Expression):
    def __init__(
        self,
        condition: Expression,
        consequence: 'BlockExpression',
        alternative: Optional['BlockExpression'] = None
    ) -> None:
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative

    def type(self) -> NodeType:
        return NodeType.IfExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "condition": self.condition.json(),
            "consequence": self.consequence.json(),
            "alternative": self.alternative.json() if self.alternative is not None else "None"
        }

class CallExpression(Expression):
    def __init__(self, function: 'IdentifierLiteral', args: Optional[list[Expression]] = None) -> None:
        self.function = function
        self.args = args or []

    def type(self) -> NodeType:
        return NodeType.CallExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "function": self.function.json(),
            "args": [arg.json() for arg in self.args]
        } 
    
class PrefixExpression(Expression):
    def __init__(self, operator: str, right_node: Expression) -> None:
        self.operator = operator
        self.right_node = right_node

    def type(self) -> NodeType:
        return NodeType.PrefixExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "operator": self.operator,
            "right_node": self.right_node.json()
        }

class NewStructExpression(Expression):
    def __init__(
        self, 
        struct_ident: 'IdentifierLiteral', 
        fields: list[tuple['IdentifierLiteral', Expression]]
    ) -> None:
        self.struct_ident = struct_ident
        self.fields = fields

    def type(self) -> NodeType:
        return NodeType.NewStructExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "struct_ident": self.struct_ident.json(),
            "fields": [[field[0].json(), field[1].json()] for field in self.fields]
        }
    
class FieldAccessExpression(Expression):
    def __init__(self, base: Expression, field: 'IdentifierLiteral') -> None:
        self.base = base
        self.field = field
    
    def type(self) -> NodeType:
        return NodeType.FieldAccessExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "base": self.base.json(),
            "field": self.field.json(),
        }
    
class EnumVariantAccessExpression(Expression):
    def __init__(self, name: 'IdentifierLiteral', variant: 'IdentifierLiteral', value: Optional[Expression] = None) -> None:
        self.name = name
        self.variant = variant
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.EnumVariantAccessExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "name": self.name.json(),
            "variant": self.variant.json(),
            "value": self.value.json() if self.value is not None else None,
        }

class MatchExpression(Expression):
    def __init__(self, match: Expression, cases: list[tuple[EnumVariantAccessExpression, BlockExpression]]) -> None:
        self.match = match
        self.cases = cases

    def type(self) -> NodeType:
        return NodeType.MatchExpression
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "match": self.match.json(),
            "cases": [[case[0].json(), case[1].json()] for case in self.cases]
        }
# endregion

# region Literals
class I32Literal(Expression):
    def __init__(self, value: int) -> None:
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.I32Literal
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "value": self.value
        }
    
class F32Literal(Expression):
    def __init__(self, value: float) -> None:
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.F32Literal
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "value": self.value
        }
    
class IdentifierLiteral(Expression):
    def __init__(self, value: str) -> None:
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.IdentifierLiteral
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "value": self.value
        }
    
class BooleanLiteral(Expression):
    def __init__(self, value: bool) -> None:
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.BooleanLiteral
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "value": self.value
        }

class StringLiteral(Expression):
    def __init__(self, value: str) -> None:
        self.value = value
    
    def type(self) -> NodeType:
        return NodeType.StringLiteral
    
    def json(self) -> dict[str, Any]:
        return {
            "type": self.type().value,
            "value": self.value
        }
# endregion