from enum import Enum

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
    DOUBLE_COLON = "DOUBLE_COLON"
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
    AMPERSAND = "AMPERSAND"

    # Keywords
    LET = "LET"
    CONST = "CONST"
    FN = "FN"
    RETURN = "RETURN"
    IF = "IF"
    ELSE = "ELSE"
    TRUE = "TRUE"
    FALSE = "FALSE"
    NULL = "NULL"
    WHILE = "WHILE"
    FOR = "FOR"
    CONTINUE = "CONTINUE"
    BREAK = "BREAK"
    IMPORT = "IMPORT"
    STRUCT = "STRUCT"
    ENUM = "ENUM"
    UNION = "UNION"
    NEW = "NEW"
    MATCH = "MATCH"
    AS = "AS"

    # Typing
    TYPE = "TYPE"

class Token:
    def __init__(self, type: TokenType, literal: str, line_no: int, position: int) -> None:
        self.type = type
        self.literal = literal
        self.line_no = line_no
        self.position = position

    def __str__(self) -> str:
        return f"Token[{self.type} : {self.literal} : Line {self.line_no} : Position {self.position}]"
    
    def __repr__(self) -> str:
        return str(self)
    
    def add_exception_info(self, e: Exception) -> Exception:
        e.add_note(f"line {self.line_no} at pos {self.position}")
        return e

KEYWORDS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "const": TokenType.CONST,
    "fn": TokenType.FN,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "null": TokenType.NULL,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "continue": TokenType.CONTINUE,
    "break": TokenType.BREAK,
    "import": TokenType.IMPORT,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "union": TokenType.UNION,
    "new": TokenType.NEW,
    "match": TokenType.MATCH,
    "as": TokenType.AS,
}

TYPE_KEYWORDS: list[str] = ["i32", "f32", "str", "void"]

def lookup_ident(ident: str) -> TokenType:
    tt = KEYWORDS.get(ident)
    if tt is not None:
        return tt
    
    if ident in TYPE_KEYWORDS:
        return TokenType.TYPE
    
    return TokenType.IDENT