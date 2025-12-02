import logging
from pathlib import Path
import sys
import argparse

from Builder import *

def main(path: str, check: bool):
    abs_path = Path(path).absolute()
    output_dir = Path("./build").absolute()

    builder = Builder(
        input_file=abs_path,
        input_reader=FileReader(abs_path),
        output_writer=FileOutputWriter(abs_path.stem, output_dir),
        llvm_compiler=ClangLLVMCompiler(output_dir),
        type_check=check,
        output_dir=output_dir,
        write_ast=True,
    )
    builder.build()

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s"
    )

def parse_args(args: list[str]) -> tuple[str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Input file to be compiled")
    parser.add_argument("--check", action='store_true', help="Indicates that the compiler should run static type analysis")
    args = parser.parse_args() # pyright: ignore[reportAssignmentType]

    path: Path = args.path # type: ignore
    check: bool = args.check # type: ignore

    return path, check # type: ignore

if __name__ == '__main__':
    setup_logger()
    main(*parse_args(sys.argv[1:]))