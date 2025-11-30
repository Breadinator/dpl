import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import llvmlite.binding as llvm

from Lexer import Lexer
from Parser import Parser
from Compiler import Compiler
from TypeChecker import TypeChecker

def main(path: str, lexer_debug: bool, check: bool):
    abs_path = Path(path).absolute()
    dir = abs_path.parent

    with open(abs_path, "r") as f:
        code = f.read()
    
    if lexer_debug:
        debug_lex = Lexer(code)
        while debug_lex.current_char is not None:
            print(debug_lex.next_token())
        exit(0)

    l = Lexer(code)
    p = Parser(l)

    try:
        program = p.parse_program()
    except Exception as e:
        print(repr(e))
        exit(1)
    finally:
        if len(p.errors) > 0:
            for err in p.errors:
                print(err)
            exit(1)
    logging.info("program parsed")
    
    if check:
        TypeChecker(dir).type_check(program)
        logging.info("no type errors detected")
        exit(0)

    os.makedirs("./build", exist_ok=True)
    with open("build/ast.json", "w") as f:
        json.dump(program.json(), f, indent=4)

    c = Compiler(dir)
    c.compile(program)
    logging.info("program compiled")

    # Output steps
    module = c.module
    module.triple = llvm.get_default_triple()

    with open("build/ir.ll", "w") as f:
        f.write(str(module))

    # run clang
    # clang build/ir.ll -o build/exe.exe
    subprocess.run(["clang", "build/ir.ll", "-o", "build/exe.exe"])

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s"
    )

def parse_args(args: list[str]) -> tuple[str, bool, bool]:
    path = None
    lexer_debug = False
    check = False

    for arg in args:
        if arg == "--lexer_debug":
            lexer_debug = True
        if arg == "--check":
            check = True
        else:
            path = arg

    if path is None:
        logging.error("Requires path")
        exit(1)
    return path, lexer_debug, check

if __name__ == '__main__':
    setup_logger()
    path, lexer_debug, check = parse_args(sys.argv[1:])
    main(path, lexer_debug, check)