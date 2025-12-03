from Lexer import Lexer
from Token import Token, TokenType
from Exceptions import *
from AST import *

from typing import Callable, Optional
from enum import Enum, auto

# Precedence Types
class PrecedenceType(Enum):
    P_LOWEST = 0
    P_EQUALS = auto()
    P_LESSGREATER = auto()
    P_SUM = auto()
    P_PRODUCT = auto()
    P_EXPONENT = auto()
    P_PREFIX = auto()
    P_CALL = auto()
    P_INDEX = auto()

# Precedence mapping
PRECEDENCES: dict[TokenType, PrecedenceType] = {
    TokenType.PLUS: PrecedenceType.P_SUM,
    TokenType.MINUS: PrecedenceType.P_SUM,
    TokenType.SLASH: PrecedenceType.P_PRODUCT,
    TokenType.ASTERISK: PrecedenceType.P_PRODUCT,
    TokenType.MODULUS: PrecedenceType.P_PRODUCT,
    TokenType.POW: PrecedenceType.P_EXPONENT,
    TokenType.EQ_EQ: PrecedenceType.P_EQUALS,
    TokenType.NOT_EQ: PrecedenceType.P_EQUALS,
    TokenType.LT: PrecedenceType.P_LESSGREATER,
    TokenType.GT: PrecedenceType.P_LESSGREATER,
    TokenType.LT_EQ: PrecedenceType.P_LESSGREATER,
    TokenType.GT_EQ: PrecedenceType.P_LESSGREATER,
    TokenType.LPAREN: PrecedenceType.P_CALL,
    TokenType.PIPE: PrecedenceType.P_CALL,
    TokenType.DOT: PrecedenceType.P_CALL, # TODO: decide
    TokenType.DOUBLE_COLON: PrecedenceType.P_CALL, # TODO: decide
    TokenType.EQ: PrecedenceType.P_EQUALS,
    TokenType.PLUS_EQ: PrecedenceType.P_EQUALS,
    TokenType.MINUS_EQ: PrecedenceType.P_EQUALS,
    TokenType.MUL_EQ: PrecedenceType.P_EQUALS,
    TokenType.DIV_EQ: PrecedenceType.P_EQUALS,
    TokenType.AS: PrecedenceType.P_CALL,
}

class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer

        self.current_token: Token = self.lexer.next_token()
        self.peek_token: Token = self.lexer.next_token()

        self.prefix_parse_fns: dict[TokenType, Callable[[], Expression]] = {
            TokenType.IDENT: self.__parse_ident,
            TokenType.I32: self.__parse_i32_literal,
            TokenType.F32: self.__parse_f32_literal,
            TokenType.LPAREN: self.__parse_grouped_expression,
            TokenType.IF: self.__parse_if_expression,
            TokenType.TRUE: self.__parse_boolean,
            TokenType.FALSE: self.__parse_boolean,
            TokenType.NULL: lambda: NullLiteral(),
            TokenType.STRING: self.__parse_string_literal,
            TokenType.MINUS: self.__parse_prefix_expression,
            TokenType.BANG: self.__parse_prefix_expression,
            TokenType.NEW: self.__parse_new_struct_expression,
            TokenType.MATCH: self.__parse_match_expression,
            TokenType.ASTERISK: self.__parse_prefix_expression,
            TokenType.AMPERSAND: self.__parse_prefix_expression,
        }
        self.infix_parse_fns: dict[TokenType, Callable[[Expression], Expression]] = {
            TokenType.PLUS: self.__parse_infix_expression,
            TokenType.MINUS: self.__parse_infix_expression,
            TokenType.ASTERISK: self.__parse_infix_expression,
            TokenType.SLASH: self.__parse_infix_expression,
            TokenType.POW: self.__parse_infix_expression,
            TokenType.MODULUS: self.__parse_infix_expression,
            TokenType.EQ_EQ: self.__parse_infix_expression,
            TokenType.NOT_EQ: self.__parse_infix_expression,
            TokenType.LT: self.__parse_infix_expression,
            TokenType.GT: self.__parse_infix_expression,
            TokenType.LT_EQ: self.__parse_infix_expression,
            TokenType.GT_EQ: self.__parse_infix_expression,
            TokenType.LPAREN: self.__parse_call_expression, # pyright: ignore[reportAttributeAccessIssue]
            TokenType.PIPE: self.__parse_pipe_call_expression,
            TokenType.DOT: self.__parse_field_access_expression,
            TokenType.DOUBLE_COLON: self.__parse_enum_variant_access_expression,
            TokenType.EQ: self.__parse_assignment_expression,
            TokenType.PLUS_EQ: self.__parse_assignment_expression,
            TokenType.MINUS_EQ: self.__parse_assignment_expression,
            TokenType.MUL_EQ: self.__parse_assignment_expression,
            TokenType.DIV_EQ: self.__parse_assignment_expression,
            TokenType.AS: self.__parse_cast_expression,
        }
    
    # region Parser Helpers
    def __next_token(self) -> None:
        self.current_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def __current_token_is(self, tt: TokenType) -> bool:
        return self.current_token.type == tt

    def __peek_token_is(self, tt: TokenType) -> bool:
        return self.peek_token.type == tt
    
    def __expect_peek(self, tt: TokenType):
        if self.__peek_token_is(tt):
            self.__next_token()
        else:
            self.__peek_error(tt)
            raise self.peek_token.add_exception_info(ExpectedTokenError(f"expected next token to be {tt}, got {self.peek_token.type}"))
        
    def __expect_peek_type_name(self):
        if self.__peek_token_is(TokenType.AMPERSAND):
            self.__next_token()
            if self.__peek_token_is(TokenType.TYPE) or self.__peek_token_is(TokenType.IDENT):
                self.__next_token()
                self.current_token.literal = '&' + self.current_token.literal
            else:
                self.__peek_error(TokenType.TYPE)
        else:
            if self.__peek_token_is(TokenType.TYPE) or self.__peek_token_is(TokenType.IDENT):
                self.__next_token()
            else:
                self.__peek_error(TokenType.TYPE)
        
    def __current_precedence(self) -> PrecedenceType:
        prec: Optional[PrecedenceType] = PRECEDENCES.get(self.current_token.type)
        return prec if prec is not None else PrecedenceType.P_LOWEST
        
    def __peek_precedence(self) -> PrecedenceType:
        prec: Optional[PrecedenceType] = PRECEDENCES.get(self.peek_token.type)
        return prec if prec is not None else PrecedenceType.P_LOWEST
    
    def __peek_error(self, tt: TokenType) -> None:
        raise self.peek_token.add_exception_info(ExpectedTokenError(f"expected next token to be {tt}, got {self.peek_token.type}"))
    
    def __current_error(self, tt: TokenType) -> None:
        raise self.peek_token.add_exception_info(ExpectedTokenError(f"expected current token to be {tt}, got {self.current_token.type}"))

    def __no_prefix_parse_fn_error(self, token: Token) -> None:
        raise token.add_exception_info(MissingPrefixParseFunction(f"no prefix parse function for {token.type} found"))
    # endregion

    def parse_program(self) -> Program:
        program = Program()
        
        while self.current_token.type != TokenType.EOF:
            stmt = self.__parse_statement()
            program.statements.append(stmt)

            self.__next_token()
        
        return program

    # region Statement Methods
    def __parse_statement(self) -> Statement:
        match self.current_token.type:
            case TokenType.LET:
                return self.__parse_let_statement()
            case TokenType.CONST:
                return self.__parse_const_statement()
            case TokenType.FN:
                return self.__parse_function_statement()
            case TokenType.RETURN:
                return self.__parse_return_statement()
            case TokenType.WHILE:
                return self.__parse_while_statement()
            case TokenType.FOR:
                return self.__parse_for_statement()
            case TokenType.CONTINUE:
                return self.__parse_continue_statement()
            case TokenType.BREAK:
                return self.__parse_break_statement()
            case TokenType.IMPORT:
                return self.__parse_import_statement()
            case TokenType.STRUCT:
                return self.__parse_struct_statement()
            case TokenType.ENUM:
                return self.__parse_enum_statement()
            case TokenType.UNION:
                return self.__parse_union_statement()
            case _:
                return self.__parse_expression_statement()
    
    def __parse_expression_statement(self) -> ExpressionStatement:
        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        if self.__peek_token_is(TokenType.SEMICOLON):
            self.__next_token()
        
        return ExpressionStatement(expr)

    def __parse_let_statement(self) -> LetStatement:
        # let a: i32 = 10;
        self.__expect_peek(TokenType.IDENT)
        stmt_name = IdentifierLiteral(self.current_token.literal)

        self.__expect_peek(TokenType.COLON)
        
        self.__expect_peek_type_name()
        stmt_value_type = self.current_token.literal

        self.__expect_peek(TokenType.EQ)
        self.__next_token()
        
        stmt_value = self.__parse_expression(PrecedenceType.P_LOWEST)
        while not self.__current_token_is(TokenType.SEMICOLON) and not self.__current_token_is(TokenType.EOF):
            self.__next_token()
        
        return LetStatement(stmt_name, stmt_value, stmt_value_type)
    
    def __parse_const_statement(self) -> LetStatement:
        # const a: i32 = 10;
        self.__expect_peek(TokenType.IDENT)
        stmt_name = IdentifierLiteral(self.current_token.literal)

        self.__expect_peek(TokenType.COLON)
        
        self.__expect_peek_type_name()
        stmt_value_type = self.current_token.literal

        self.__expect_peek(TokenType.EQ)
        self.__next_token()
        
        stmt_value = self.__parse_expression(PrecedenceType.P_LOWEST)
        while not self.__current_token_is(TokenType.SEMICOLON) and not self.__current_token_is(TokenType.EOF):
            self.__next_token()
        
        return LetStatement(stmt_name, stmt_value, stmt_value_type, True)
    
    def __parse_function_statement(self) -> FunctionStatement:
        # fn foo() -> i32 { return 10; }
        self.__expect_peek(TokenType.IDENT)
        name = IdentifierLiteral(self.current_token.literal)

        self.__expect_peek(TokenType.LPAREN)
        
        params = self.__parse_function_parameters()

        self.__expect_peek(TokenType.ARROW)
        
        self.__expect_peek_type_name()
        return_type = self.current_token.literal

        if self.__peek_token_is(TokenType.FATARROW):
            # fn foo() -> i32 => 10;
            self.__next_token()
            self.__next_token()
            expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            body = BlockExpression()
            body.return_expression = expr
            while not self.__current_token_is(TokenType.SEMICOLON) and not self.__current_token_is(TokenType.EOF):
                self.__next_token()
        else:
            self.__expect_peek(TokenType.LBRACE)
        
            body = self.__parse_block_expression()
            
        return FunctionStatement(params, body, name, return_type)

    def __parse_function_parameters(self) -> list[FunctionParameter]:
        params: list[FunctionParameter] = []

        if self.__peek_token_is(TokenType.RPAREN):
            self.__next_token()
            return params
        
        self.__next_token()

        first_param = FunctionParameter(self.current_token.literal, "")
        self.__expect_peek(TokenType.COLON)
        self.__next_token()
        first_param.value_type = self.current_token.literal
        params.append(first_param)

        while self.__peek_token_is(TokenType.COMMA):
            self.__next_token()
            self.__next_token()

            param = FunctionParameter(self.current_token.literal, "")
            self.__expect_peek(TokenType.COLON)
            self.__next_token()
            param.value_type = self.current_token.literal
            params.append(param)
        
        self.__expect_peek(TokenType.RPAREN)

        return params

    def __parse_return_statement(self) -> ReturnStatement:
        self.__next_token()
        stmt_return_value = self.__parse_expression(PrecedenceType.P_LOWEST)
        self.__expect_peek(TokenType.SEMICOLON)
        return ReturnStatement(stmt_return_value)
   
    def __parse_while_statement(self) -> WhileStatement:
        self.__next_token()
        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        self.__expect_peek(TokenType.LBRACE)
        body = self.__parse_block_expression()
        return WhileStatement(condition, body)
    
    def __parse_for_statement(self) -> ForStatement:
        if self.__peek_token_is(TokenType.LPAREN):
            self.__next_token()
        self.__expect_peek(TokenType.LET)
        var_declaration = self.__parse_let_statement()
        self.__next_token()
        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        self.__expect_peek(TokenType.SEMICOLON)
        self.__next_token() # skip ;
        action = self.__parse_assignment_expression(self.__parse_expression(PrecedenceType.P_LOWEST))
        if self.__peek_token_is(TokenType.RPAREN):
            self.__next_token()
        self.__expect_peek(TokenType.LBRACE)
        body = self.__parse_block_expression()
        return ForStatement(var_declaration, condition, action, body)
    
    def __parse_break_statement(self) -> BreakStatement:
        self.__next_token()
        return BreakStatement()
    
    def __parse_continue_statement(self) -> ContinueStatement:
        self.__next_token()
        return ContinueStatement()
    
    def __parse_import_statement(self) -> ImportStatement:
        self.__expect_peek(TokenType.STRING)
        stmt = ImportStatement(self.current_token.literal)
        self.__expect_peek(TokenType.SEMICOLON)
        return stmt
    
    def __parse_struct_statement(self) -> StructStatement:
        self.__expect_peek(TokenType.IDENT)
        ident = IdentifierLiteral(self.current_token.literal)
        self.__expect_peek(TokenType.LBRACE)
        fields: list[tuple[str, str]] = []
        self.__next_token()

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            if not self.__current_token_is(TokenType.IDENT):
                self.__current_error(TokenType.IDENT)
            field_name = self.current_token.literal
            self.__expect_peek(TokenType.COLON)
            self.__next_token()
            field_type = self.__parse_type()
            fields.append((field_name, field_type))
            self.__expect_peek(TokenType.SEMICOLON)
            self.__next_token()

        if not self.__current_token_is(TokenType.RBRACE):
            self.__expect_peek(TokenType.RBRACE)
        return StructStatement(ident, fields)

    def __parse_enum_statement(self) -> EnumStatement:
        self.__expect_peek(TokenType.IDENT)
        name = IdentifierLiteral(self.current_token.literal)

        self.__expect_peek(TokenType.LBRACE)

        variants: list[IdentifierLiteral] = []
        while not self.__peek_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            self.__expect_peek(TokenType.IDENT)
            variants.append(IdentifierLiteral(self.current_token.literal))
            self.__expect_peek(TokenType.COMMA)
        self.__next_token()

        return EnumStatement(name, variants)

    def __parse_union_statement(self) -> UnionStatement:
        self.__expect_peek(TokenType.IDENT)
        name = IdentifierLiteral(self.current_token.literal)
        
        self.__expect_peek(TokenType.LBRACE)
        self.__next_token()

        variants: list[tuple[IdentifierLiteral, Optional[str]]] = []

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            if not self.__current_token_is(TokenType.IDENT):
                self.__peek_error(TokenType.IDENT)

            variant_ident = IdentifierLiteral(self.current_token.literal)
            variant_type = None

            if self.__peek_token_is(TokenType.LPAREN):
                self.__next_token()
                self.__expect_peek_type_name()
                variant_type = self.current_token.literal
                self.__expect_peek(TokenType.RPAREN)
            
            variants.append((variant_ident, variant_type))

            if self.__peek_token_is(TokenType.COMMA):
                self.__next_token()
            self.__next_token()
        
        return UnionStatement(name, variants)
    # endregion

    # region Expression Methods
    def __parse_expression(self, precedence: PrecedenceType) -> Expression:
        if self.__current_token_is(TokenType.LBRACE):
            return self.__parse_block_expression()

        prefix_fn = self.prefix_parse_fns.get(self.current_token.type)
        if prefix_fn is None:
            self.__no_prefix_parse_fn_error(self.current_token)
            raise Exception
        
        left_expr = prefix_fn()
        while not self.__peek_token_is(TokenType.SEMICOLON) and not self.__peek_token_is(TokenType.RBRACE) and precedence.value < self.__peek_precedence().value:
            infix_fn = self.infix_parse_fns.get(self.peek_token.type)
            if infix_fn is None:
                return left_expr
            
            self.__next_token()

            left_expr = infix_fn(left_expr)
        
        return left_expr

    def __parse_infix_expression(self, left_node: Expression) -> Expression:
        operator = self.current_token.literal
        precedence = self.__current_precedence()
        self.__next_token()
        right_node = self.__parse_expression(precedence)
        return InfixExpression(left_node, operator, right_node)
 
    def __parse_assignment_expression(self, lh: Expression) -> AssignExpression:
        operator = self.current_token.literal
        precedence = self.__current_precedence()
        self.__next_token()
        rh = self.__parse_expression(precedence)
        return AssignExpression(lh, rh, operator)

    def __parse_grouped_expression(self) -> Expression:
        self.__next_token()
        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        self.__expect_peek(TokenType.RPAREN)
        return expr
    
    def __parse_block_expression(self) -> BlockExpression:
        expr = BlockExpression()
        self.__next_token()

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            if self.__current_token_is(TokenType.SEMICOLON):
                self.__next_token()
                continue

            stmt = self.__parse_statement()

            if isinstance(stmt, ExpressionStatement):
                if self.__current_token_is(TokenType.SEMICOLON):
                    expr.statements.append(stmt)
                    self.__next_token()
                    continue
                else:
                    expr.return_expression = stmt.expr
                    break
            elif isinstance(stmt, ReturnStatement):
                expr.statements.append(stmt)
                if self.__current_token_is(TokenType.SEMICOLON):
                    self.__next_token()
                break
            else:
                expr.statements.append(stmt)
                if self.__current_token_is(TokenType.SEMICOLON):
                    self.__next_token()

        if not self.__current_token_is(TokenType.RBRACE):
            self.__expect_peek(TokenType.RBRACE)

        return expr

    def __parse_if_expression(self) -> IfExpression:
        self.__next_token()
        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        self.__expect_peek(TokenType.LBRACE)
        consequence = self.__parse_block_expression()
        
        if self.__peek_token_is(TokenType.ELSE):
            self.__next_token()
            self.__expect_peek(TokenType.LBRACE)
            alternative = self.__parse_block_expression()
        else:
            alternative = None
        
        return IfExpression(condition, consequence, alternative)

    def __parse_call_expression(self, function: IdentifierLiteral) -> CallExpression:
        expr = CallExpression(function)
        args = self.__parse_expression_list(TokenType.RPAREN)
        expr.args = args
        return expr
    
    def __parse_pipe_call_expression(self, lhs: Expression) -> Expression:
        self.__expect_peek(TokenType.IDENT)
        func_ident = IdentifierLiteral(self.current_token.literal)

        if self.__peek_token_is(TokenType.LPAREN):
            self.__next_token()
            call_expr = self.__parse_call_expression(func_ident)
            call_expr.args.insert(0, lhs)
            return call_expr
        else:
            call_expr = CallExpression(func_ident)
            call_expr.args = [lhs]
            return call_expr

    
    def __parse_expression_list(self, end: TokenType) -> list[Expression]:
        exprs: list[Expression] = []

        if self.__peek_token_is(end):
            self.__next_token() # R paren
            return exprs
        self.__next_token() # L paren

        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        exprs.append(expr)

        while self.__peek_token_is(TokenType.COMMA):
            self.__next_token()
            self.__next_token()
            expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            exprs.append(expr)

        self.__expect_peek(end)
        return exprs
    
    def __parse_prefix_expression(self) -> PrefixExpression:
        operator = self.current_token.literal  
        self.__next_token()
        right_node = self.__parse_expression(PrecedenceType.P_PREFIX)
        return PrefixExpression(operator, right_node)

    def __parse_new_struct_expression(self) -> NewStructExpression:
        if not (self.__peek_token_is(TokenType.TYPE) or self.__peek_token_is(TokenType.IDENT)):
            self.__peek_error(TokenType.TYPE)
        self.__next_token()
        struct_ident = IdentifierLiteral(self.current_token.literal)

        self.__expect_peek(TokenType.LBRACE)

        self.__next_token()

        fields: list[tuple[IdentifierLiteral, Expression]] = []

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            if self.__current_token_is(TokenType.SEMICOLON):
                self.__next_token()
                continue

            if not self.__current_token_is(TokenType.IDENT):
                raise self.current_token.add_exception_info(ExpectedTokenError(
                    f"expected field IDENT inside struct literal, got {self.current_token.type}"
                ))
            field_name = IdentifierLiteral(self.current_token.literal)

            if not self.__peek_token_is(TokenType.EQ):
                if self.__peek_token_is(TokenType.SEMICOLON):
                    raise self.peek_token.add_exception_info(ExpectedTokenError(
                        f"expected '=' and an initializer for field `{field_name.value}`, got semicolon. "
                        "Did you forget `= expr`? Or did a nested expression (like a `|>` pipe) fail to consume tokens?"
                    ))
                else:
                    raise self.peek_token.add_exception_info(ExpectedTokenError(
                        f"expected '=' after field `{field_name.value}`, got {self.peek_token.type}"
                    ))

            self.__next_token()
            self.__next_token()

            init_expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            fields.append((field_name, init_expr))

            if self.__peek_token_is(TokenType.SEMICOLON):
                self.__next_token()
                self.__next_token()
                continue

            if self.__peek_token_is(TokenType.RBRACE):
                self.__next_token()
                break

            self.__next_token()

        if not self.__current_token_is(TokenType.RBRACE):
            self.__expect_peek(TokenType.RBRACE)

        return NewStructExpression(struct_ident=struct_ident, fields=fields)

    def __parse_field_access_expression(self, lhs: Expression) -> FieldAccessExpression:
        self.__expect_peek(TokenType.IDENT)
        field_ident = IdentifierLiteral(self.current_token.literal)
        return FieldAccessExpression(lhs, field_ident)
    
    def __parse_enum_variant_access_expression(self, lhs: Expression) -> EnumVariantAccessExpression:
        if not isinstance(lhs, IdentifierLiteral):
            raise ValueError("Left side of enum/union variant access must be an identifier")

        self.__expect_peek(TokenType.IDENT)
        variant_ident = IdentifierLiteral(self.current_token.literal)

        value_expr: Optional[Expression] = None
        if self.__peek_token_is(TokenType.LPAREN):
            self.__next_token()
            self.__next_token() 
            value_expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            self.__expect_peek(TokenType.RPAREN)

        return EnumVariantAccessExpression(lhs, variant_ident, value_expr)

    def __parse_match_expression(self) -> MatchExpression:
        self.__next_token()
        match_expr = self.__parse_expression(PrecedenceType.P_LOWEST)

        self.__expect_peek(TokenType.LBRACE)
        self.__next_token()

        cases: list[tuple[EnumVariantAccessExpression, BlockExpression]] = []

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            if self.__current_token_is(TokenType.COMMA):
                self.__next_token()
                continue

            if not self.__current_token_is(TokenType.IDENT):
                raise self.current_token.add_exception_info(
                    ExpectedTokenError(f"expected IDENT got {self.current_token}")
                )

            lhs = IdentifierLiteral(self.current_token.literal)
            self.__expect_peek(TokenType.DOUBLE_COLON)
            self.__next_token()
            if not self.__current_token_is(TokenType.IDENT):
                raise self.current_token.add_exception_info(
                    ExpectedTokenError(f"expected variant IDENT got {self.current_token}")
                )
            rhs = IdentifierLiteral(self.current_token.literal)

            value_expr: Optional[IdentifierLiteral] = None
            if self.__peek_token_is(TokenType.LPAREN):
                self.__next_token()
                self.__next_token()
                if self.__current_token_is(TokenType.IDENT):
                    value_expr = IdentifierLiteral(self.current_token.literal)
                    if self.__peek_token_is(TokenType.RPAREN):
                        self.__next_token()
                    else:
                        self.__peek_error(TokenType.RPAREN)
                else:
                    raise self.current_token.add_exception_info(
                        ExpectedTokenError(f"expected IDENT inside parentheses for variant receiver")
                    )

            enum_access = EnumVariantAccessExpression(lhs, rhs, value_expr)

            self.__expect_peek(TokenType.FATARROW)
            self.__next_token()

            block = self.__parse_block_expression()
            cases.append((enum_access, block))

            if self.__peek_token_is(TokenType.COMMA):
                self.__next_token()

            self.__next_token()

        if not self.__current_token_is(TokenType.RBRACE):
            self.__expect_peek(TokenType.RBRACE)

        self.__next_token()
        return MatchExpression(match_expr, cases)
    
    def __parse_cast_expression(self, lhs: Expression) -> CastExpression:
        self.__next_token() # skip as
        typ = self.__parse_type()
        return CastExpression(lhs, typ)
    
    def __parse_type(self) -> str:
        if self.__current_token_is(TokenType.TYPE) or self.__current_token_is(TokenType.IDENT):
            field_type = self.current_token.literal
        elif self.__current_token_is(TokenType.AMPERSAND):
            self.__next_token()
            field_type = "&" + self.__parse_type()
        else:
            raise ParserException(f"couldn't parse token `{self.current_token}` as type")
        
        return field_type
    # endregion

    # region Prefix Methods
    def __parse_i32_literal(self) -> Expression:
        try:
            return I32Literal(int(self.current_token.literal))
        except ValueError:
            raise self.current_token.add_exception_info(LiteralParseError(f"Could not parse `{self.current_token.literal}` as an integer"))
        
    def __parse_f32_literal(self) -> Expression:
        try:
            return F32Literal(float(self.current_token.literal))
        except ValueError:
            raise self.current_token.add_exception_info(LiteralParseError(f"Could not parse `{self.current_token.literal}` as a float"))
        
    def __parse_ident(self) -> IdentifierLiteral:
        return IdentifierLiteral(self.current_token.literal)
    
    def __parse_boolean(self) -> BooleanLiteral:
        return BooleanLiteral(value=self.__current_token_is(TokenType.TRUE))
    
    def __parse_string_literal(self) -> StringLiteral:
        return StringLiteral(self.current_token.literal)
    # endregion