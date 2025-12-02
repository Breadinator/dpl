from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
from typing import Optional, Any
import llvmlite.binding as llvm
from llvmlite.ir.module import Module

from Compiler import Compiler
from Lexer import Lexer
from Parser import Parser
from TypeChecker import TypeChecker

class InputReader(ABC):
    @abstractmethod
    def get_input(self) -> str: ...

class OutputWriter(ABC):
    @abstractmethod
    def write_ast(self, ast: dict[str, Any]) -> None: ...

    @abstractmethod
    def write_ir(self, module: Module) -> Optional[Path]: ...

class LLVMCompiler(ABC):
    @abstractmethod
    def compile_llvm(self, ir: Path): ...

@dataclass
class Builder:
    input_file: Path # I would like to get rid of this

    input_reader: InputReader
    output_writer: OutputWriter
    llvm_compiler: LLVMCompiler
    type_checker: Optional[TypeChecker] = None

    type_check: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./build").absolute())
    write_ast: bool = False

    def build(self):
        code = self.input_reader.get_input()
        l = Lexer(code)
        p = Parser(l)
        program = p.parse_program()

        if self.type_check and self.type_checker is not None:
            self.type_checker.type_check(program)

        if self.write_ast:
            self.output_writer.write_ast(program.json())
        
        c = Compiler(self.input_file.parent)
        c.compile(program)

        module = c.module
        module.triple = llvm.get_default_triple()

        path = self.output_writer.write_ir(module)
        if path is not None:
            self.llvm_compiler.compile_llvm(path)

class FileReader(InputReader):
    def __init__(self, path: Path) -> None:
        self.path = path

    def get_input(self) -> str:
        with open(self.path, "r") as f:
            return f.read()
        
class FileOutputWriter(OutputWriter):
    def __init__(self, name: str, output_dir: Path) -> None:
        self.name = name
        self.output_dir = output_dir
    
    def write_ast(self, ast: dict[str, Any]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_dir / f"{self.name}.ast.json", "w") as f:
            json.dump(ast, f, indent=4)

    def write_ir(self, module: Module) -> Optional[Path]:
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.output_dir / f"{self.name}.ll"
        with open(path, "w") as f:
            f.write(str(module))
        return path
    
class ClangLLVMCompiler(LLVMCompiler):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def compile_llvm(self, ir: Path):
        output_exe = self.output_dir / f"{ir.stem}.exe"
        subprocess.run(["clang", ir, "-o", output_exe])
