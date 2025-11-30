from llvmlite import ir
from llvmlite.ir import instructions
from typing import Optional
from pathlib import Path

from AST import Node, NodeType, Expression, Program
from AST import ExpressionStatement, LetStatement, FunctionStatement, ReturnStatement, AssignStatement, ImportStatement
from AST import StructStatement, EnumStatement
from AST import WhileStatement, ForStatement, BreakStatement, ContinueStatement
from AST import InfixExpression, BlockExpression, IfExpression, CallExpression, PrefixExpression, NewStructExpression, FieldAccessExpression, EnumVariantAccessExpression
from AST import I32Literal, F32Literal, IdentifierLiteral, BooleanLiteral, StringLiteral

from Environment import Environment
from Lexer import Lexer
from Parser import Parser
from Exceptions import *

class Compiler:
    def __init__(self, dir: Path) -> None:
        self.type_map: dict[str, ir.Type] = {
            "i32": ir.IntType(32),
            "f32": ir.FloatType(),
            "bool": ir.IntType(1),
            "str": ir.PointerType(ir.IntType(8)),
            "void": ir.VoidType()
        }

        self.module = ir.Module('main')
        self.builder = ir.IRBuilder()

        self.counter = 0

        self.env = Environment()

        self.errors: list[str] = []

        self.__initialize_builtins()

        self.breakpoints: list[ir.Block] = []
        self.continues: list[ir.Block] = []

        self.dir = dir
        self.global_parsed_modules: dict[str, Program] = {}

    def __initialize_builtins(self) -> None:
        def define_bools():
            bool_type = self.type_map['bool']
            
            true_var = ir.GlobalVariable(self.module, bool_type, 'True')
            true_var.initializer = ir.Constant(bool_type, 1)
            true_var.global_constant = True

            false_var = ir.GlobalVariable(self.module, bool_type, 'False')
            false_var.initializer = ir.Constant(bool_type, 0)
            false_var.global_constant = True

            self.env.define('True', true_var, true_var.type)
            self.env.define('False', false_var, false_var.type)
        define_bools()

        def define_printf():
            fnty = ir.FunctionType(
                self.type_map['i32'],
                [ir.IntType(8).as_pointer()],
                var_arg=True,
            )
            fn = ir.Function(self.module, fnty, 'printf')
            self.env.define('printf', fn, ir.IntType(32))
        define_printf()

    def __increment_counter(self) -> int:
        self.counter += 1
        return self.counter

    def compile(self, node: Node) -> None:
        match node.type():
            case NodeType.Program:
                self.__visit_program(node) # pyright: ignore[reportArgumentType]
            
            # Statements
            case NodeType.ExpressionStatement:
                self.__visit_expression_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.LetStatement:
                self.__visit_let_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.FunctionStatement:
                self.__visit_function_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.ReturnStatement:
                self.__visit_return_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.AssignStatement:
                self.__visit_assign_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.WhileStatement:
                self.__visit_while_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.ForStatement:
                self.__visit_for_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.BreakStatement:
                self.__visit_break_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.ContinueStatement:
                self.__visit_continue_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.ImportStatement:
                self.__visit_import_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.StructStatement:
                self.__visit_struct_statement(node) # pyright: ignore[reportArgumentType]
            case NodeType.EnumStatement:
                self.__visit_enum_statement(node) # pyright: ignore[reportArgumentType]

            # Expressions
            case NodeType.InfixExpression:
                self.__visit_infix_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.BlockExpression:
                self.__visit_block_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.IfExpression:
                self.__visit_if_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.CallExpression:
                self.__visit_call_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.FieldAccessExpression:
                self.__visit_field_access_expression(node) # pyright: ignore[reportArgumentType]

            case _:
                raise NotImplementedError
    
    # region Visit methods
    def __visit_program(self, node: Program) -> None:
        for stmt in node.statements:
            self.compile(stmt)

    # region Statements
    def __visit_expression_statement(self, node: ExpressionStatement) -> None:
        self.compile(node.expr)

    def __visit_let_statement(self, node: LetStatement) -> None:
        name = node.name.value
        value = node.value
        _value_type = node.value_type  # TODO: implement

        value, typ = self.__resolve_value(value)
        if value is None or typ is None:
            return

        # If the variable does not exist yet
        existing_ptr, _ = self.env.lookup(name)
        if existing_ptr is None:
            # If `typ` is already a pointer (like from `new Struct`), allocate pointer-to-pointer
            if isinstance(typ, ir.PointerType):
                ptr = self.builder.alloca(typ)
                self.builder.store(value, ptr)
                self.env.define(name, ptr, typ)
            else:
                # Standard value type
                ptr = self.builder.alloca(typ)
                self.builder.store(value, ptr)
                self.env.define(name, ptr, typ)
        else:
            # Variable already exists, just store new value
            ptr = existing_ptr
            # If value is a pointer but the stored type is a struct, store the dereferenced value
            if isinstance(value.type, ir.PointerType) and isinstance(ptr.type.pointee, ir.LiteralStructType):
                value = self.builder.load(value)
            self.builder.store(value, ptr)


    def __visit_return_statement(self, node: ReturnStatement) -> None:
        value = node.return_value
        value, _typ = self.__resolve_value(value)
        if value is None:
            return None
        self.builder.ret(value)

    def __visit_function_statement(self, node: FunctionStatement) -> None:
        name: str = node.name.value
        self.__extract_ret(node.body)
        body = node.body
        params = node.params
        param_names: list[str] = [p.name for p in params]
        param_types: list[ir.Type] = [self.type_map[p.value_type] for p in params]
        return_type = self.type_map[node.return_type]

        fnty = ir.FunctionType(return_type, param_types)
        func = ir.Function(self.module, fnty, name)
        block = func.append_basic_block(f'{name}_entry')

        previous_builder = self.builder
        self.builder = ir.IRBuilder(block)

        # Store pointers to each param
        params_ptr: list[ir.AllocaInstr] = []
        for i, typ in enumerate(param_types):
            ptr = self.builder.alloca(typ)
            self.builder.store(func.args[i], ptr)
            params_ptr.append(ptr)

        # Add params to env
        previous_env = self.env
        self.env = Environment(parent=previous_env)
        for i, x in enumerate(zip(param_types, param_names)):
            ptr = params_ptr[i]
            self.env.define(x[1], ptr, x[0])

        self.env.define(name, func, return_type)

        self.compile(body)

        self.env = previous_env
        self.env.define(name, func, return_type)

        self.builder = previous_builder

    def __visit_assign_statement(self, node: AssignStatement) -> None:
        name = node.ident.value
        value = node.rh
        operator = node.operator

        var_ptr, _ = self.env.lookup(name)
        if var_ptr is None:
            self.errors.append(f"identifier `{name}` reassigned before declaration")
            return
 
        right_value, right_type = self.__resolve_value(value)
        if right_value is None:
            return       
        
        orig_value = self.builder.load(var_ptr)
        if isinstance(orig_value.type, ir.IntType) and isinstance(right_type, ir.FloatType):
            orig_value = self.builder.sitofp(orig_value, ir.FloatType())
        
        if isinstance(orig_value.type, ir.FloatType) and isinstance(right_type, ir.IntType):
            orig_value = self.builder.sitofp(right_value, ir.FloatType())
        
        value = None
        match operator:
            case '=':
                value = right_value
            case '+=':
                if isinstance(orig_value.type, ir.IntType) and isinstance(right_type, ir.IntType):
                    value = self.builder.add(orig_value, right_value)
                else:
                    value = self.builder.fadd(orig_value, right_value)
            case '-=':
                if isinstance(orig_value.type, ir.IntType) and isinstance(right_type, ir.IntType):
                    value = self.builder.sub(orig_value, right_value)
                else:
                    value = self.builder.fsub(orig_value, right_value)
            case '*=':
                if isinstance(orig_value.type, ir.IntType) and isinstance(right_type, ir.IntType):
                    value = self.builder.mul(orig_value, right_value)
                else:
                    value = self.builder.fmul(orig_value, right_value)
            case '/=':
                if isinstance(orig_value.type, ir.IntType) and isinstance(right_type, ir.IntType):
                    value = self.builder.sdiv(orig_value, right_value)
                else:
                    value = self.builder.fdiv(orig_value, right_value)
            case _:
                raise ValueError("Unsupported assignment operator")
        
        ptr, _ = self.env.lookup(name)
        if ptr is None:
            return None
        self.builder.store(value, ptr)

    def __visit_while_statement(self, node: WhileStatement) -> None:
        test, _ = self.__resolve_value(node.condition)
        if test is None:
            return

        while_loop_entry = self.builder.append_basic_block(f"while_loop_entry_{self.__increment_counter()}")
        while_loop_otherwise = self.builder.append_basic_block(f"while_loop_otherwise_{self.counter}")

        self.builder.cbranch(test, while_loop_entry, while_loop_otherwise)
        self.builder.position_at_start(while_loop_entry)
        self.compile(node.body)
        test, _ = self.__resolve_value(node.condition)
        if test is None:
            return None
        self.builder.cbranch(test, while_loop_entry, while_loop_otherwise)
        self.builder.position_at_start(while_loop_otherwise)

    def __visit_for_statement(self, node: ForStatement) -> None:
        prev_env = self.env
        self.env = Environment(parent=prev_env)

        # Compile var declaration
        self.compile(node.var_declaration)

        func = self.builder.function
        cond_bb = func.append_basic_block(f"for_cond_{self.__increment_counter()}")
        body_bb = func.append_basic_block(f"for_body_{self.counter}")
        inc_bb = func.append_basic_block(f"for_inc_{self.counter}")
        end_bb = func.append_basic_block(f"for_end_{self.counter}")

        # Setup break/continue
        self.breakpoints.append(end_bb)
        self.continues.append(inc_bb)

        self.builder.branch(cond_bb)

        # --- Condition ---
        self.builder.position_at_start(cond_bb)
        test, _ = self.__resolve_value(node.condition)
        if test is None:
            return None
        self.builder.cbranch(test, body_bb, end_bb)

        # --- Body ---
        self.builder.position_at_start(body_bb)
        self.compile(node.body)
        # If body did not branch, jump to increment
        if getattr(self.builder.block, "terminator", None) is None:
            self.builder.branch(inc_bb)

        # --- Increment ---
        self.builder.position_at_start(inc_bb)
        self.compile(node.action)
        self.builder.branch(cond_bb)

        # --- End ---
        self.builder.position_at_start(end_bb)

        self.breakpoints.pop()
        self.continues.pop()
        self.env = prev_env

    
    def __visit_break_statement(self, node: BreakStatement) -> None:
        self.builder.branch(self.breakpoints[-1])
    
    def __visit_continue_statement(self, node: ContinueStatement) -> None:
        self.builder.branch(self.continues[-1])

    def __visit_import_statement(self, node: ImportStatement) -> None:
        if self.global_parsed_modules.get(node.path) is not None:
            return None
        with open(self.dir / node.path, "r") as f:
            module_code = f.read()
        
        l = Lexer(module_code)
        p = Parser(l)
        program = p.parse_program()
        if len(p.errors) > 0:
            print(f"Error in imported module: {node.path}")
            for err in p.errors:
                print(err)
            exit(1)
        self.compile(program)
        self.global_parsed_modules[node.path] = program
    
    def __visit_struct_statement(self, node: StructStatement) -> None:
        name = node.ident.value
        existing_struct, _, _ = self.env.lookup_struct(name)
        if existing_struct is not None:
            self.errors.append(f"struct `{name}` already defined")
            return
        
        field_names = [f[0] for f in node.fields]
        field_type_names = [f[1] for f in node.fields]

        _llvm_struct = self.__define_struct(name, field_names, field_type_names)
    
    def __visit_enum_statement(self, node: EnumStatement) -> None:
        name = node.name.value
        variants = [variant.value for variant in node.variants]
        self.env.define_enum(name, variants)
    # endregion

    # region Expressions
    def __visit_infix_expression(self, node: InfixExpression):
        if node.right_node is None:
            return None

        operator = node.operator
        left_value, left_type = self.__resolve_value(node.left_node)
        right_value, right_type = self.__resolve_value(node.right_node)
        if left_value is None or left_type is None or right_value is None or right_type is None:
            return None

        value: Optional[ir.Instruction] = None
        typ: Optional[ir.Type] = None
        if isinstance(right_type, ir.IntType) and isinstance(left_type, ir.IntType):
            typ = self.type_map['i32']
            match operator:
                case '+':
                    value = self.builder.add(left_value, right_value)
                case '-':
                    value = self.builder.sub(left_value, right_value)
                case '*':
                    value = self.builder.mul(left_value, right_value)
                case '/':
                    value = self.builder.sdiv(left_value, right_value)
                case '%':
                    value = self.builder.srem(left_value, right_value)
                case '^':
                    raise NotImplementedError                
                case '<':
                    value = self.builder.icmp_signed('<', left_value, right_value)
                    typ = ir.IntType(1)
                case '<=':
                    value = self.builder.icmp_signed('<=', left_value, right_value)
                    typ = ir.IntType(1)
                case '>':
                    value = self.builder.icmp_signed('>', left_value, right_value)
                    typ = ir.IntType(1)
                case '>=':
                    value = self.builder.icmp_signed('>=', left_value, right_value)
                    typ = ir.IntType(1)
                case '==':
                    value = self.builder.icmp_signed('==', left_value, right_value)
                    typ = ir.IntType(1)
                case '!=':
                    value = self.builder.icmp_signed('!=', left_value, right_value)
                    typ = ir.IntType(1)

                case _:
                    raise NotImplementedError
        elif isinstance(right_type, ir.FloatType) and isinstance(left_type, ir.FloatType):
            typ = self.type_map['f32']
            match operator:
                case '+':
                    value = self.builder.fadd(left_value, right_value)
                case '-':
                    value = self.builder.fsub(left_value, right_value)
                case '*':
                    value = self.builder.fmul(left_value, right_value)
                case '/':
                    value = self.builder.fdiv(left_value, right_value)
                case '%':
                    value = self.builder.frem(left_value, right_value)
                case '^':
                    raise NotImplementedError
                case '<':
                    value = self.builder.fcmp_ordered('<', left_value, right_value)
                    typ = ir.IntType(1)
                case '<=':
                    value = self.builder.fcmp_ordered('<=', left_value, right_value)
                    typ = ir.IntType(1)
                case '>':
                    value = self.builder.fcmp_ordered('>', left_value, right_value)
                    typ = ir.IntType(1)
                case '>=':
                    value = self.builder.fcmp_ordered('>=', left_value, right_value)
                    typ = ir.IntType(1)
                case '==':
                    value = self.builder.fcmp_ordered('==', left_value, right_value)
                    typ = ir.IntType(1)
                case '!=':
                    value = self.builder.fcmp_ordered('!=', left_value, right_value)
                    typ = ir.IntType(1)

                case _:
                    raise NotImplementedError
        else:
            raise NotImplementedError("Infix operation {operator} not defined between {left_type} and {right_type}")
                
        return value, typ
    
    def __visit_block_expression(self, node: BlockExpression) -> tuple[Optional[ir.Value], Optional[ir.Type]]:
        for stmt in node.statements:
            self.compile(stmt)

        if node.return_expression is not None:
            return self.__resolve_value(node.return_expression)
        else:
            return ir.Constant(self.type_map["i32"], 0), self.type_map["i32"]
    
    def __visit_if_expression(self, node: IfExpression) -> tuple[Optional[ir.Value], Optional[ir.Type]]:
        cond_val, cond_type = self.__resolve_value(node.condition)
        if cond_val is None or cond_type is None:
            return None, None

        # normalize cond -> i1
        if isinstance(cond_type, ir.IntType) and cond_type.width != 1:
            zero = ir.Constant(cond_type, 0)
            test = self.builder.icmp_signed('!=', cond_val, zero)
        elif isinstance(cond_type, ir.FloatType):
            zero = ir.Constant(cond_type, 0.0)
            test = self.builder.fcmp_ordered('!=', cond_val, zero)
        else:
            test = cond_val

        func = self.builder.function
        then_bb = func.append_basic_block("if.then")
        else_bb = func.append_basic_block("if.else")
        merge_bb = func.append_basic_block("if.end")

        # conditional branch into then/else
        self.builder.cbranch(test, then_bb, else_bb)

        # --- then branch ---
        self.builder.position_at_end(then_bb)
        then_val, then_type = self.__resolve_value(node.consequence)
        # If we didn't already terminate the block, branch to merge
        then_branches_to_merge = False
        if getattr(self.builder.block, "terminator", None) is None:
            self.builder.branch(merge_bb)
            then_branches_to_merge = True
        then_block_for_phi = self.builder.block if then_branches_to_merge else None

        # --- else branch ---
        self.builder.position_at_end(else_bb)
        if node.alternative is not None:
            else_val, else_type = self.__resolve_value(node.alternative)
        else:
            else_val, else_type = None, None

        else_branches_to_merge = False
        if getattr(self.builder.block, "terminator", None) is None:
            self.builder.branch(merge_bb)
            else_branches_to_merge = True
        else_block_for_phi = self.builder.block if else_branches_to_merge else None

        # --- merge block ---
        self.builder.position_at_end(merge_bb)

        # If both branches produced no value, the if expression yields no value:
        if (then_type is None) and (else_type is None):
            return None, None

        # Normalize types and defaults (your existing logic)
        if then_type is None and else_type is not None:
            then_type = else_type
        if else_type is None and then_type is not None:
            else_type = then_type

        if type(then_type) is not type(else_type):
            raise TypeError("Mismatched types in if branches")

        # If no branch actually branched to merge, nothing to phi — return default/None
        incoming_blocks: list[ir.Block] = []
        incoming_values: list[ir.Value] = []

        if then_block_for_phi is not None:
            if then_val is None:
                then_val = ir.Constant(then_type, 0 if isinstance(then_type, ir.IntType) else 0.0)
            incoming_blocks.append(then_block_for_phi)
            incoming_values.append(then_val)

        if else_block_for_phi is not None:
            if else_val is None:
                else_val = ir.Constant(else_type, 0 if isinstance(else_type, ir.IntType) else 0.0)
            incoming_blocks.append(else_block_for_phi)
            incoming_values.append(else_val)

        # If only one branch actually flows to merge, the result is the incoming value from that branch
        if len(incoming_blocks) == 1:
            return incoming_values[0], then_type

        # Otherwise create a phi with exactly the same number of incoming entries as actual predecessors
        phi = self.builder.phi(then_type) # pyright: ignore[reportArgumentType]
        for val, blk in zip(incoming_values, incoming_blocks):
            phi.add_incoming(val, blk)

        return phi, then_type


    def __visit_call_expression(self, node: CallExpression) -> tuple[Optional[ir.Instruction], Optional[ir.Type]]:
        name = node.function.value
        params = node.args
        
        args: list[ir.Value] = []
        types: list[ir.Type] = []
        for param in params:
            p_val, p_typ = self.__resolve_value(param)
            if p_val is None or p_typ is None:
                return None, None
            args.append(p_val)
            types.append(p_typ)

        ret = None
        match name:
            case 'printf':
                ret = self.builtin_printf(args, types[0]) # pyright: ignore[reportArgumentType]
                ret_type = self.type_map['i32']
            case _:
                func, ret_type = self.env.lookup(name)
                if func is None:
                    return None, None
                ret = self.builder.call(func, args)
        
        return ret, ret_type
    
    def __visit_prefix_expression(self, node: PrefixExpression) -> tuple[Optional[ir.Value], Optional[ir.Type]]:
        right_value, right_type = self.__resolve_value(node.right_node)
        if right_value is None or right_type is None:
            return None, None

        typ = None
        value = None

        if isinstance(right_type, ir.FloatType):
            typ = ir.FloatType()
            match node.operator:
                case '-':
                    value = self.builder.fmul(right_value, ir.Constant(ir.FloatType(), -1.0))
                case '!':
                    value = ir.Constant(ir.IntType(1), 0)
                case _:
                    raise ValueError(f"invalid operator {node.operator}")
        elif isinstance(right_type, ir.IntType):
            typ = ir.IntType(32)
            match node.operator:
                case '-':
                    value = self.builder.mul(right_value, ir.Constant(ir.IntType(32), -1))
                case '!':
                    value = self.builder.not_(right_value)
                case _:
                    raise ValueError(f"invalid operator {node.operator}")
        
        return value, typ

    def __visit_new_struct_expression(self, node: NewStructExpression) -> tuple[ir.Value, ir.Type]:
        name = node.struct_ident.value
        field_names = [field[0].value for field in node.fields]
        field_exprs = [field[1] for field in node.fields]

        # Lookup struct definition
        struct_type, expected_field_names, expected_field_types = self.env.lookup_struct(name)
        if struct_type is None or expected_field_names is None or expected_field_types is None:
            raise LookupError(f"struct `{name}` not defined")

        # Allocate memory for the struct
        ptr = self.builder.alloca(struct_type)

        # Check field names match
        if set(field_names) != set(expected_field_names):
            raise FieldMismatchError(f"struct `{name}` missing or extra fields")

        # Resolve all field expressions
        field_values = [self.__resolve_value(expr) for expr in field_exprs]

        # Store each field
        for i, (value, typ) in enumerate(field_values):
            if value is None or typ is None:
                raise ValueResolverError(f"couldn't resolve field {field_names[i]} on struct `{name}`")
            
            expected_type = expected_field_types[i]
            
            # If the expected type is a struct and value is a pointer (from alloca), load it
            if isinstance(expected_type, (ir.IdentifiedStructType, ir.LiteralStructType)):
                if isinstance(typ, ir.PointerType) and typ.pointee == expected_type:
                    value = self.builder.load(value)
                    typ = expected_type

            # Type check
            if typ != expected_type:
                raise TypeMismatchError(f"field {field_names[i]} on struct `{name}` resolved as wrong type")

            # Get pointer to the field
            field_ptr = self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), i)])
            # Store the value into the struct
            self.builder.store(value, field_ptr)

        return ptr, ptr.type

    def __visit_field_access_expression(self, node: FieldAccessExpression) -> tuple[ir.Value, ir.Type]:
        field_name = node.field.value

        # Resolve base
        base_value, base_type = self.__resolve_value(node.base)
        if base_value is None or base_type is None:
            raise ValueResolverError("cannot resolve base of field access")

        # Unwrap pointer type if needed
        if isinstance(base_type, ir.PointerType):
            check_type = base_type.pointee
        else:
            check_type = base_type

        if not isinstance(check_type, (ir.IdentifiedStructType, ir.LiteralStructType)):
            raise TypeMismatchError(f"field access base must be a struct, got {check_type}")

        # Lookup struct info
        struct_type, field_names, field_types = self.env.lookup_struct(check_type.name)
        if struct_type is None:
            raise LookupError(f"struct `{check_type.name}` not found")
        if field_name not in field_names:
            raise FieldMismatchError(f"struct `{check_type.name}` has no field `{field_name}`")
        field_index = field_names.index(field_name)
        field_type = field_types[field_index]

        # Pointer to the field inside the struct
        field_ptr = self.builder.gep(
            base_value,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_index)]
        )

        # Nested struct: return pointer, primitives: load
        if isinstance(field_type, (ir.IdentifiedStructType, ir.LiteralStructType)):
            return field_ptr, field_type
        else:
            return self.builder.load(field_ptr), field_type

    def __visit_enum_variant_access_expression(self, node: EnumVariantAccessExpression) -> tuple[ir.Value, ir.Type]:
        enum_metadata = self.env.lookup_enum(node.name.value)
        if enum_metadata is None:
            raise LookupError(f"couldn't find enum with name `{node.name.value}`")
        if node.variant.value not in enum_metadata.variants:
            raise FieldMismatchError(f"enum `{node.name.value}` doesn't have variant `{node.variant.value}`")
        idx = enum_metadata.variants.index(node.variant.value)
        return ir.Constant(ir.IntType(32), idx), ir.IntType(32)
    # endregion

    # endregion

    # region Helper Methods
    def __resolve_value(self, node: Expression, value_type: Optional[str] = None) -> tuple[Optional[ir.Value], Optional[ir.Type]]: # pyright: ignore[reportRedeclaration]
        match node.type():
            case NodeType.I32Literal:
                node: I32Literal = node # pyright: ignore[reportRedeclaration, reportAssignmentType]
                value, typ = node.value, self.type_map['i32' if value_type is None else value_type]
                return ir.Constant(typ, value), typ
            case NodeType.F32Literal:
                node: F32Literal = node # pyright: ignore[reportRedeclaration, reportAssignmentType]
                value, typ = node.value, self.type_map['f32' if value_type is None else value_type]
                return ir.Constant(typ, value), typ
            case NodeType.IdentifierLiteral:
                node: IdentifierLiteral = node # pyright: ignore[reportRedeclaration, reportAssignmentType]
                ptr, typ = self.env.lookup(node.value)
                if ptr is None or typ is None:
                    raise ValueError(f"`Either is None:\n\t{ptr}`\n\tor `{typ}`")
                return self.builder.load(ptr), typ
            case NodeType.BooleanLiteral:
                node: BooleanLiteral = node # pyright: ignore[reportRedeclaration, reportAssignmentType]
                return ir.Constant(ir.IntType(1), 1 if node.value else 0), ir.IntType(1)
            case NodeType.StringLiteral:
                node: StringLiteral = node # pyright: ignore[reportAssignmentType]
                return self.__convert_string(node.value)
            
            # Expression Values
            case NodeType.InfixExpression:
                return self.__visit_infix_expression(node) # type: ignore
            case NodeType.BlockExpression:
                return self.__visit_block_expression(node) # type: ignore
            case NodeType.IfExpression:
                return self.__visit_if_expression(node) # type: ignore
            case NodeType.CallExpression:
                return self.__visit_call_expression(node) # type: ignore
            case NodeType.PrefixExpression:
                return self.__visit_prefix_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.NewStructExpression:
                return self.__visit_new_struct_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.FieldAccessExpression:
                return self.__visit_field_access_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.EnumVariantAccessExpression:
                return self.__visit_enum_variant_access_expression(node)
            
            case default:
                raise NotImplementedError(f"Not implemented: {default}")
    
    def __convert_string(self, string: str) -> tuple[ir.GlobalVariable, ir.Type]:
        string = string.replace("\\n", "\n\0")
        fmt = f"{string}\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)), bytearray(fmt.encode("utf8")))
        global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name=f'__str_{self.__increment_counter()}')
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt
        return global_fmt, global_fmt.type

    def builtin_printf(self, params: list[ir.Value], return_type: ir.Type) -> Optional[instructions.CallInstr]:
        """ Basic C builtin printf: expects first param to be a pointer/GlobalVariable to the format. """
        func, _ = self.env.lookup('printf')
        if func is None:
            return None

        if len(params) == 0:
            return None

        fmt_val = params[0]
        rest_params = params[1:]

        # If the argument is a LoadInstr from a global, extract the global pointer.
        # Example: load i8*, i8** @someglobal  -> we want @someglobal
        if isinstance(fmt_val, instructions.LoadInstr):
            # load has operands; the first operand is the pointer loaded from
            operand0 = fmt_val.operands[0]
            # bitcast the loaded value (which may already be i8*) to i8*
            fmt_arg = self.builder.bitcast(operand0, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # If the argument is a GlobalVariable (created by __convert_string), bitcast it to i8*.
        # NOTE: GlobalVariable is a subclass of Value in llvmlite; check by attribute
        if hasattr(fmt_val, "initializer") and isinstance(fmt_val, ir.GlobalVariable):
            fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # If it's already a pointer (like an i8*), use as-is
        if hasattr(fmt_val, "type") and isinstance(fmt_val.type, ir.PointerType):
            fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # Last-resort: if it's some constant global index or something else, try to coerce.
        try:
            fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])
        except Exception:
            # couldn't transform the value into a format pointer; bail gracefully
            return None

    def __extract_ret(self, node: BlockExpression):
        if node.return_expression is not None:
            node.statements.append(ReturnStatement(node.return_expression))
            node.return_expression = None
    
    def __resolve_type(self, name: str) -> Optional[ir.Type]:
        if name in self.type_map:
            return self.type_map[name]
        if name in self.env.structs:
            struct, _, _ = self.env.lookup_struct(name)
            return struct
        else:
            return None

    def __define_struct(
        self, name: str, field_names: list[str], field_type_names: list[str]
    ) -> ir.IdentifiedStructType:
        # Create an opaque struct first (needed for recursive types)
        identified = ir.global_context.get_identified_type(name)
        
        # Register struct early in environment
        self.env.define_struct(name, identified, field_names, None)

        resolved_types: list[ir.Type] = []
        for tname in field_type_names:
            typ = self.__resolve_type(tname)
            if typ is None:
                raise TypeError(f"unknown field type '{tname}' for struct '{name}'")
            
            # Do NOT wrap struct types in pointers anymore
            # Everything is embedded by value
            resolved_types.append(typ)

        # Set the actual body of the struct
        identified.set_body(*resolved_types)
        
        # Update environment with resolved types
        self.env.define_struct(name, identified, field_names, resolved_types)
        return identified

    def __is_struct_type(self, t: ir.Type) -> bool:
        return isinstance(t, (ir.IdentifiedStructType, ir.LiteralStructType))
    
    def __is_pointer_to_struct(self, t: ir.Type) -> bool:
        return isinstance(t, ir.PointerType) and self.__is_struct_type(t.pointee)
    # endregion
