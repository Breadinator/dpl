import json
import os
import subprocess
import llvmlite.binding as llvm # pyright: ignore[reportMissingTypeStubs]

from Lexer import Lexer
from Parser import Parser
from Compiler import Compiler

LEXER_DEBUG = False

if __name__ == '__main__':
    with open("tests/test.dpl", "r") as f:
        code = f.read()
    
    if LEXER_DEBUG:
        debug_lex = Lexer(code)
        while debug_lex.current_char is not None:
            print(debug_lex.next_token())
        exit(0)

    l = Lexer(code)
    p = Parser(l)

    program = p.parse_program()
    if len(p.errors) > 0:
        for err in p.errors:
            print(err)
        exit(1)

    os.makedirs("./build", exist_ok=True)
    with open("build/ast.json", "w") as f:
        json.dump(program.json(), f, indent=4)

    c = Compiler()
    c.compile(program)

    # Output steps
    module = c.module
    module.triple = llvm.get_default_triple()

    with open("build/ir.ll", "w") as f:
        f.write(str(module))

    # run clang
    # clang build/ir.ll -o build/exe.exe
    subprocess.run(["clang", "build/ir.ll", "-o", "build/exe.exe"])