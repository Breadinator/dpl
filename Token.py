from enum import Enum
from typing import Any

class TokenType(Enum):
    # Special tokens
    EOF = "EOF"
    ILLEGAL = "ILLEGAL"

    # Data Types
    IDENT = "IDENT"
    I32 = "I32"
    F32 = "F32"
    STRING = "STRING"

    # Arithmetic Tokens
    PLUS = "PLUS"
    MINUS = "MINUS"
    ASTERISK = "ASTERISK"
    SLASH = "SLASH"
    POW = "POW"
    MODULUS = "MODULUS"

    # Assignment
    EQ = "EQ"
    PLUS_EQ = "PLUS_EQ"
    MINUS_EQ = "MINUS_EQ"
    MUL_EQ = "MUL_EQ"
    DIV_EQ = "DIV_EQ"

    # Comp symbols
    LT = '<'
    GT = '>'
    EQ_EQ = '=='
    NOT_EQ = '!='
    LT_EQ = '<='
    GT_EQ = '>='

    # Symbols
    COLON = "COLON"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    ARROW = "ARROW"
    FATARROW = "FATARROW"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    PIPE = "PIPE"
    DOT = "DOT"

    # Prefix symbols
    BANG = "BANG"

    # Keywords
    LET = "LET"
    FN = "FN"
    RETURN = "RETURN"
    IF = "IF"
    ELSE = "ELSE"
    TRUE = "TRUE"
    FALSE = "FALSE"
    WHILE = "WHILE"
    FOR = "FOR"
    CONTINUE = "CONTINUE"
    BREAK = "BREAK"
    IMPORT = "IMPORT"
    STRUCT = "STRUCT"
    NEW = "NEW"

    # Typing
    TYPE = "TYPE"

class Token:
    def __init__(self, type: TokenType, literal: Any, line_no: int, position: int) -> None:
        self.type = type
        self.literal = literal
        self.line_no = line_no
        self.position = position

    def __str__(self) -> str:
        return f"Token[{self.type} : {self.literal} : Line {self.line_no} : Position {self.position}]"
    
    def __repr__(self) -> str:
        return str(self)

KEYWORDS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "fn": TokenType.FN,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "continue": TokenType.CONTINUE,
    "break": TokenType.BREAK,
    "import": TokenType.IMPORT,
    "struct": TokenType.STRUCT,
    "new": TokenType.NEW,
}

TYPE_KEYWORDS: list[str] = ["i32", "f32", "str", "void"]

def lookup_ident(ident: str) -> TokenType:
    tt = KEYWORDS.get(ident)
    if tt is not None:
        return tt
    
    if ident in TYPE_KEYWORDS:
        return TokenType.TYPE
    
    return TokenType.IDENT