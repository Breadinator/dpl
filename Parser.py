from Lexer import Lexer
from Token import Token, TokenType

from AST import Statement, Expression, Program
from AST import FunctionParameter
from AST import ExpressionStatement, LetStatement, FunctionStatement, ReturnStatement, AssignStatement, ImportStatement, StructStatement
from AST import WhileStatement, BreakStatement, ContinueStatement, ForStatement
from AST import InfixExpression, BlockExpression, IfExpression, CallExpression, PrefixExpression, NewStructExpression, FieldAccessExpression
from AST import I32Literal, F32Literal, IdentifierLiteral, BooleanLiteral, StringLiteral

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
}

class Parser:
    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer

        self.errors: list[str] = []

        self.current_token: Token = self.lexer.next_token()
        self.peek_token: Token = self.lexer.next_token()

        self.prefix_parse_fns: dict[TokenType, Callable[[], Optional[Expression]]] = {
            TokenType.IDENT: self.__parse_ident,
            TokenType.I32: self.__parse_i32_literal,
            TokenType.F32: self.__parse_f32_literal,
            TokenType.LPAREN: self.__parse_grouped_expression,
            TokenType.IF: self.__parse_if_expression,
            TokenType.TRUE: self.__parse_boolean,
            TokenType.FALSE: self.__parse_boolean,
            TokenType.STRING: self.__parse_string_literal,
            TokenType.MINUS: self.__parse_prefix_expression,
            TokenType.BANG: self.__parse_prefix_expression,
            TokenType.NEW: self.__parse_new_struct_expression,
        }
        self.infix_parse_fns: dict[TokenType, Callable[[Expression], Optional[Expression]]] = {
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
        }
    
    # region Parser Helpers
    def __next_token(self) -> None:
        self.current_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def __current_token_is(self, tt: TokenType) -> bool:
        return self.current_token.type == tt

    def __peek_token_is(self, tt: TokenType) -> bool:
        return self.peek_token.type == tt
    
    def __peek_token_is_assignment(self) -> bool:
        assignment_operators = [
            TokenType.EQ,
            TokenType.PLUS_EQ,
            TokenType.MINUS_EQ,
            TokenType.MUL_EQ,
            TokenType.DIV_EQ,
        ]
        return self.peek_token.type in assignment_operators

    def __expect_peek(self, tt: TokenType) -> bool:
        if self.__peek_token_is(tt):
            self.__next_token()
            return True
        else:
            self.__peek_error(tt)
            return False
        
    def __expect_peek_type_name(self) -> bool:
        if self.__peek_token_is(TokenType.TYPE) or self.__peek_token_is(TokenType.IDENT):
            self.__next_token()
            return True
        else:
            self.__peek_error(TokenType.TYPE)
            return False
        
    def __current_precedence(self) -> PrecedenceType:
        prec: Optional[PrecedenceType] = PRECEDENCES.get(self.current_token.type)
        return prec if prec is not None else PrecedenceType.P_LOWEST
        
    def __peek_precedence(self) -> PrecedenceType:
        prec: Optional[PrecedenceType] = PRECEDENCES.get(self.peek_token.type)
        return prec if prec is not None else PrecedenceType.P_LOWEST
    
    def __peek_error(self, tt: TokenType) -> None:
        self.errors.append(f"expected next token to be {tt}, got {self.peek_token.type}")

    def __no_prefix_parse_fn_error(self, tt: TokenType) -> None:
        self.errors.append(f"no prefix parse function for {tt} found")
    # endregion

    def parse_program(self) -> Program:
        program = Program()
        
        while self.current_token.type != TokenType.EOF:
            stmt = self.__parse_statement()
            if stmt is not None:
                program.statements.append(stmt)

            self.__next_token()
        
        return program

    # region Statement Methods
    def __parse_statement(self) -> Optional[Statement]:
        if self.__current_token_is(TokenType.IDENT) and self.__peek_token_is_assignment():
            return self.__parse_assignment_statement()

        match self.current_token.type:
            case TokenType.LET:
                return self.__parse_let_statement()
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
            case _:
                return self.__parse_expression_statement()
    
    def __parse_expression_statement(self) -> Optional[ExpressionStatement]:
        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        if expr is None:
            return None

        if self.__peek_token_is(TokenType.SEMICOLON):
            self.__next_token()
        
        return ExpressionStatement(expr)

    def __parse_let_statement(self) -> Optional[LetStatement]:
        # let a: i32 = 10;
        if not self.__expect_peek(TokenType.IDENT):
            return None
        stmt_name = IdentifierLiteral(self.current_token.literal)

        if not self.__expect_peek(TokenType.COLON):
            return None
        
        if not self.__expect_peek_type_name():
            return None
        stmt_value_type = self.current_token.literal

        if not self.__expect_peek(TokenType.EQ):
            return None
        self.__next_token()
        
        stmt_value = self.__parse_expression(PrecedenceType.P_LOWEST)
        if stmt_value is None:
            return None
        while not self.__current_token_is(TokenType.SEMICOLON) and not self.__current_token_is(TokenType.EOF):
            self.__next_token()
        
        return LetStatement(stmt_name, stmt_value, stmt_value_type)
    
    def __parse_function_statement(self) -> Optional[FunctionStatement]:
        # fn foo() -> i32 { return 10; }
        if not self.__expect_peek(TokenType.IDENT):
            return None
        name = IdentifierLiteral(self.current_token.literal)

        if not self.__expect_peek(TokenType.LPAREN):
            return None
        
        params = self.__parse_function_parameters()
        if params is None:
            return None

        if not self.__expect_peek(TokenType.ARROW):
            return None
        
        if not self.__expect_peek_type_name():
            return None
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
            if not self.__expect_peek(TokenType.LBRACE):
                return None
        
            body = self.__parse_block_expression()
            if body is None:
                return None
            
        return FunctionStatement(params, body, name, return_type)

    def __parse_function_parameters(self) -> Optional[list[FunctionParameter]]:
        params: list[FunctionParameter] = []

        if self.__peek_token_is(TokenType.RPAREN):
            self.__next_token()
            return params
        
        self.__next_token()

        first_param = FunctionParameter(self.current_token.literal, "")
        if not self.__expect_peek(TokenType.COLON):
            return None
        self.__next_token()
        first_param.value_type = self.current_token.literal
        params.append(first_param)

        while self.__peek_token_is(TokenType.COMMA):
            self.__next_token()
            self.__next_token()

            param = FunctionParameter(self.current_token.literal, "")
            if not self.__expect_peek(TokenType.COLON):
                return None
            self.__next_token()
            param.value_type = self.current_token.literal
            params.append(param)
        
        if not self.__expect_peek(TokenType.RPAREN):
            return None

        return params


    def __parse_return_statement(self) -> Optional[ReturnStatement]:
        self.__next_token()
        stmt_return_value = self.__parse_expression(PrecedenceType.P_LOWEST)
        if stmt_return_value is None:
            return None
        if not self.__expect_peek(TokenType.SEMICOLON):
            return None
        return ReturnStatement(stmt_return_value)
    
    def __parse_assignment_statement(self) -> Optional[AssignStatement]:
        ident = IdentifierLiteral(self.current_token.literal)
        self.__next_token() # skip ident

        operator = self.current_token.literal
        self.__next_token()

        rh = self.__parse_expression(PrecedenceType.P_LOWEST)
        if rh is None:
            return None
        self.__next_token()
        return AssignStatement(ident, rh, operator)
    
    def __parse_while_statement(self) -> Optional[WhileStatement]:
        self.__next_token()
        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        if not self.__expect_peek(TokenType.LBRACE):
            return None
        body = self.__parse_block_expression()

        if condition is None or body is None:
            return None
        else:
            return WhileStatement(condition, body)
    
    def __parse_for_statement(self) -> Optional[ForStatement]:
        if self.__peek_token_is(TokenType.LPAREN):
            self.__next_token()

        if not self.__expect_peek(TokenType.LET):
            return None
        
        var_declaration = self.__parse_let_statement()
        if var_declaration is None:
            return None
        
        self.__next_token()

        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        if condition is None:
            return None
        
        if not self.__expect_peek(TokenType.SEMICOLON):
            return None
        self.__next_token() # skip ;

        action = self.__parse_assignment_statement()
        if action is None:
            return None
        
        if self.__peek_token_is(TokenType.RPAREN):
            self.__next_token()

        if not self.__expect_peek(TokenType.LBRACE):
            return None
        
        body = self.__parse_block_expression()
        if body is None:
            return None

        return ForStatement(var_declaration, condition, action, body)
    
    def __parse_break_statement(self) -> BreakStatement:
        self.__next_token()
        return BreakStatement()
    
    def __parse_continue_statement(self) -> ContinueStatement:
        self.__next_token()
        return ContinueStatement()
    
    def __parse_import_statement(self) -> Optional[ImportStatement]:
        if not self.__expect_peek(TokenType.STRING):
            return None
        stmt = ImportStatement(self.current_token.literal)
        if not self.__expect_peek(TokenType.SEMICOLON):
            return None
        return stmt
    
    def __parse_struct_statement(self) -> Optional[StructStatement]:
        # current_token is 'struct' when called
        if not self.__expect_peek(TokenType.IDENT):
            return None
        ident = IdentifierLiteral(self.current_token.literal)

        if not self.__expect_peek(TokenType.LBRACE):
            return None

        fields: list[tuple[str, str]] = []

        # move into the first token after '{'
        self.__next_token()

        # parse `name: Type;` entries until `}` or EOF
        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            # field name must be an identifier
            if not self.__current_token_is(TokenType.IDENT):
                self.__peek_error(TokenType.IDENT)   # yields a helpful error message
                return None
            field_name = self.current_token.literal

            # next must be ':'
            if not self.__expect_peek(TokenType.COLON):
                return None
            # advance to the type token
            self.__next_token()

            # field type can be either a builtin TYPE token or an IDENT (user-defined type)
            if not (self.__current_token_is(TokenType.TYPE) or self.__current_token_is(TokenType.IDENT)):
                self.__peek_error(TokenType.TYPE)
                return None
            field_type = self.current_token.literal

            fields.append((field_name, field_type))

            # expect semicolon after a field
            if not self.__expect_peek(TokenType.SEMICOLON):
                return None

            # advance into the next token after the semicolon
            self.__next_token()

        # ensure we are at the closing brace (should be, but be strict)
        if not self.__current_token_is(TokenType.RBRACE):
            if not self.__expect_peek(TokenType.RBRACE):
                return None

        return StructStatement(ident, fields)


    # endregion

    # region Expression Methods
    def __parse_expression(self, precedence: PrecedenceType) -> Optional[Expression]:
        if self.__current_token_is(TokenType.LBRACE):
            return self.__parse_block_expression()

        prefix_fn = self.prefix_parse_fns.get(self.current_token.type)
        if prefix_fn is None:
            self.__no_prefix_parse_fn_error(self.current_token.type)
            return None
        
        left_expr = prefix_fn()
        while not self.__peek_token_is(TokenType.SEMICOLON) and not self.__peek_token_is(TokenType.RBRACE) and precedence.value < self.__peek_precedence().value:
            infix_fn = self.infix_parse_fns.get(self.peek_token.type)
            if infix_fn is None:
                return left_expr
            
            self.__next_token()

            if left_expr is None:
                return None
            left_expr = infix_fn(left_expr)
        
        return left_expr

    def __parse_infix_expression(self, left_node: Expression) -> Optional[Expression]:
        operator = self.current_token.literal
        precedence = self.__current_precedence()
        self.__next_token()
        right_node = self.__parse_expression(precedence)
        if right_node is None:
            return None
        return InfixExpression(left_node, operator, right_node)
    
    def __parse_grouped_expression(self) -> Optional[Expression]:
        self.__next_token()
        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        if not self.__expect_peek(TokenType.RPAREN):
            return None
        return expr
    
    def __parse_block_expression(self) -> Optional[BlockExpression]:
        expr = BlockExpression()
        # Enter the block: current_token should move to the first token after '{'
        self.__next_token()

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            # Skip stray empty statements / semicolons
            if self.__current_token_is(TokenType.SEMICOLON):
                self.__next_token()
                continue

            stmt = self.__parse_statement()
            if stmt is None:
                # Error recovery: skip until ;, }, or EOF
                while (not self.__current_token_is(TokenType.SEMICOLON)
                    and not self.__current_token_is(TokenType.RBRACE)
                    and not self.__current_token_is(TokenType.EOF)):
                    self.__next_token()
                # If we stopped at a semicolon, consume it so we don't re-parse it.
                if self.__current_token_is(TokenType.SEMICOLON):
                    self.__next_token()
                continue

            # If the statement is an ExpressionStatement, we must determine
            # whether it was terminated by a semicolon (i.e. current_token == ';')
            # or not (current_token is last token of expression and peek_token == RBRACE).
            if isinstance(stmt, ExpressionStatement):
                # If current token is a semicolon -> it was an expression statement with ';'
                if self.__current_token_is(TokenType.SEMICOLON):
                    # It's a statement (not a block return), keep it in statements (optional)
                    expr.statements.append(stmt)
                    # consume the semicolon and continue
                    self.__next_token()
                    continue
                else:
                    # No trailing semicolon -> this is the block's return expression
                    expr.return_expression = stmt.expr
                    # Do not consume the closing '}' here; leave loop so caller will consume it below
                    break

            # Normal statement (Let, Function, etc.)
            expr.statements.append(stmt)

            # If a statement left us on a semicolon, consume it so the loop continues correctly.
            if self.__current_token_is(TokenType.SEMICOLON):
                self.__next_token()
            else:
                # Otherwise advance one token to continue parsing inside the block
                self.__next_token()

        # Ensure we're positioned at the closing brace '}' (or advance to it).
        if not self.__current_token_is(TokenType.RBRACE):
            # If the RBRACE is the peek token, advance to it; otherwise it's an error.
            if not self.__expect_peek(TokenType.RBRACE):
                return None

        return expr

    def __parse_if_expression(self) -> Optional[IfExpression]:
        self.__next_token()

        condition = self.__parse_expression(PrecedenceType.P_LOWEST)
        if condition is None:
            return None

        if not self.__expect_peek(TokenType.LBRACE):
            return None
        
        consequence = self.__parse_block_expression()
        if not consequence:
            return None
        
        if self.__peek_token_is(TokenType.ELSE):
            self.__next_token()

            if not self.__expect_peek(TokenType.LBRACE):
                return None
            
            alternative = self.__parse_block_expression()
        else:
            alternative = None
        
        return IfExpression(condition, consequence, alternative)

    def __parse_call_expression(self, function: IdentifierLiteral) -> Optional[CallExpression]:
        expr = CallExpression(function)
        args = self.__parse_expression_list(TokenType.RPAREN)
        if args is None:
            return None
        expr.args = args
        return expr
    
    def __parse_pipe_call_expression(self, lhs: Expression) -> Optional[Expression]:
        # next token should be an identifier (function name)
        if not self.__peek_token_is(TokenType.IDENT):
            self.__peek_error(TokenType.IDENT)
            return None

        # advance to IDENT (now current_token is IDENT)
        self.__next_token()
        func_ident = IdentifierLiteral(self.current_token.literal)

        # if a call follows (LPAREN), advance into LPAREN and reuse call parser
        if self.__peek_token_is(TokenType.LPAREN):
            # advance to LPAREN so __parse_call_expression sees same shape as normal calls
            self.__next_token()  # now current_token == LPAREN
            call_expr = self.__parse_call_expression(func_ident)
            if call_expr is None:
                return None
            # insert the left-hand expression as the first argument
            call_expr.args.insert(0, lhs)
            return call_expr
        else:
            # bare identifier -> make a call with a single arg
            call_expr = CallExpression(func_ident)
            call_expr.args = [lhs]
            return call_expr

    
    def __parse_expression_list(self, end: TokenType) -> Optional[list[Expression]]:
        exprs: list[Expression] = []

        if self.__peek_token_is(end):
            self.__next_token() # R paren
            return exprs
        self.__next_token() # L paren

        expr = self.__parse_expression(PrecedenceType.P_LOWEST)
        if expr is None:
            return None
        exprs.append(expr)

        while self.__peek_token_is(TokenType.COMMA):
            self.__next_token()
            self.__next_token()
            expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            if expr is None:
                return None
            exprs.append(expr)

        if not self.__expect_peek(end):
            return None

        return exprs
    
    def __parse_prefix_expression(self) -> Optional[PrefixExpression]:
        operator = self.current_token.literal  
        self.__next_token()
        right_node = self.__parse_expression(PrecedenceType.P_PREFIX)
        if right_node is None:
            return None
        return PrefixExpression(operator, right_node)

    def __parse_new_struct_expression(self) -> Optional[NewStructExpression]:
        # current_token == NEW
        # next token must be the type name (TYPE or IDENT)
        if not (self.__peek_token_is(TokenType.TYPE) or self.__peek_token_is(TokenType.IDENT)):
            self.__peek_error(TokenType.TYPE)
            return None

        self.__next_token()  # now current_token is the type name
        struct_ident = IdentifierLiteral(self.current_token.literal)

        # Expect opening brace
        if not self.__expect_peek(TokenType.LBRACE):
            return None

        # Move into first token inside braces
        self.__next_token()

        # collect field initializers
        fields: list[tuple[IdentifierLiteral, Expression]] = []

        while not self.__current_token_is(TokenType.RBRACE) and not self.__current_token_is(TokenType.EOF):
            # skip stray semicolons
            if self.__current_token_is(TokenType.SEMICOLON):
                self.__next_token()
                continue

            # field name must be an identifier
            if not self.__current_token_is(TokenType.IDENT):
                self.__peek_error(TokenType.IDENT)
                return None
            field_name = IdentifierLiteral(self.current_token.literal)

            # expect =
            if not self.__expect_peek(TokenType.EQ):
                return None

            # advance into the initializer expression token
            self.__next_token()

            init_expr = self.__parse_expression(PrecedenceType.P_LOWEST)
            if init_expr is None:
                return None

            fields.append((field_name, init_expr))

            # optional semicolon termination
            if self.__peek_token_is(TokenType.SEMICOLON):
                self.__next_token()  # consume ';'
                self.__next_token()  # advance to next token
                continue
            else:
                self.__next_token()

        # ensure we are at RBRACE
        if not self.__current_token_is(TokenType.RBRACE):
            if not self.__expect_peek(TokenType.RBRACE):
                return None

        return NewStructExpression(struct_ident=struct_ident, fields=fields)

    def __parse_field_access_expression(self, lhs: Expression) -> Optional[FieldAccessExpression]:
        if not self.__peek_token_is(TokenType.IDENT):
            self.__peek_error(TokenType.IDENT)
            return None

        self.__next_token()  
        field_ident = IdentifierLiteral(self.current_token.literal)
        return FieldAccessExpression(lhs, field_ident)

    # endregion

    # region Prefix Methods
    def __parse_i32_literal(self) -> Optional[Expression]:
        try:
            return I32Literal(int(self.current_token.literal))
        except ValueError:
            self.errors.append(f"Could not parse `{self.current_token.literal}` as an integer")
            return None
        
    def __parse_f32_literal(self) -> Optional[Expression]:
        try:
            return F32Literal(float(self.current_token.literal))
        except ValueError:
            self.errors.append(f"Could not parse `{self.current_token.literal}` as a float")
            return None
        
    def __parse_ident(self) -> Optional[IdentifierLiteral]:
        return IdentifierLiteral(self.current_token.literal)
    
    def __parse_boolean(self) -> BooleanLiteral:
        return BooleanLiteral(value=self.__current_token_is(TokenType.TRUE))
    
    def __parse_string_literal(self) -> StringLiteral:
        return StringLiteral(self.current_token.literal)
    # endregion