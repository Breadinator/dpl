from llvmlite import ir
from llvmlite.ir import instructions
from typing import Optional
from pathlib import Path

from AST import *
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

            self.env.define_record('True', true_var, true_var.type)
            self.env.define_record('False', false_var, false_var.type)
        define_bools()

        def define_printf():
            fnty = ir.FunctionType(
                self.type_map['i32'],
                [ir.IntType(8).as_pointer()],
                var_arg=True,
            )
            fn = ir.Function(self.module, fnty, 'printf')
            self.env.define_record('printf', fn, ir.IntType(32))
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
            case NodeType.UnionStatement:
                self.__visit_union_statement(node) # pyright: ignore[reportArgumentType]

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
            case NodeType.MatchExpression:
                self.__visit_match_expression(node) # pyright: ignore[reportArgumentType]

            case _:
                raise NotImplementedError(node.type().name)
    
    # region Visit methods
    def __visit_program(self, node: Program) -> None:
        for stmt in node.statements:
            self.compile(stmt)

    # region Statements
    def __visit_expression_statement(self, node: ExpressionStatement) -> None:
        self.compile(node.expr)

    def __visit_let_statement(self, node: LetStatement):
        name = node.name.value
        store_type = self.__resolve_type(node.value_type)
        
        rhs_val, rhs_type = self.__resolve_value(node.value)
        
        if isinstance(store_type, ir.PointerType):
            var_ptr = self.builder.alloca(store_type)
            if not isinstance(rhs_val.type, ir.PointerType):
                raise TypeMismatchError(f"Cannot store non-pointer {rhs_val.type} into reference type {store_type}")
            self.builder.store(rhs_val, var_ptr)
            self.env.define_record(name, var_ptr, store_type)
            return
        
        if isinstance(rhs_type, ir.PointerType) and (not isinstance(store_type, ir.PointerType)):
            rhs_val = self.builder.load(rhs_val)
            rhs_type = rhs_type.pointee

        if store_type and store_type != rhs_type:
            raise TypeMismatchError(
                f"cannot store value of type `{rhs_type}` into variable `{name}` of type `{store_type}`"
            )

        alloc_type = store_type
        var_ptr = self.builder.alloca(alloc_type)

        self.builder.store(rhs_val, var_ptr)
        self.env.define_record(name, var_ptr, alloc_type, node.const, False)

    def __visit_return_statement(self, node: ReturnStatement) -> None:
        value = node.return_value
        value, _typ = self.__resolve_value(value)
        self.builder.ret(value)

    def __visit_function_statement(self, node: FunctionStatement) -> None:
        name: str = node.name.value
        self.__extract_ret(node.body)
        body = node.body
        params = node.params
        _param_names: list[str] = [p.name for p in params]
        param_types: list[ir.Type] = [self.type_map[p.value_type.replace('&', '')] for p in params]
        return_type = self.type_map[node.return_type]

        fnty = ir.FunctionType(return_type, param_types)
        func = ir.Function(self.module, fnty, name)
        block = func.append_basic_block(f'{name}_entry')

        previous_builder = self.builder
        self.builder = ir.IRBuilder(block)

        params_ptr: list[ir.AllocaInstr] = []
        for i, param in enumerate(params):
            typ = self.type_map[param.value_type.replace('&', '')]  # underlying type
            ptr = self.builder.alloca(typ)
            self.builder.store(func.args[i], ptr)
            params_ptr.append(ptr)

        previous_env = self.env
        self.env = Environment(parent=previous_env)

        for i, param in enumerate(params):
            ptr = params_ptr[i]
            base_type = self.type_map[param.value_type.replace('&', '')]

            if param.value_type.startswith('&'):
                self.env.define_record(param.name, ptr, base_type.as_pointer())
            else:
                self.env.define_record(param.name, ptr, base_type)

        self.env.define_record(name, func, return_type)

        self.compile(body)

        self.env = previous_env
        self.env.define_record(name, func, return_type, True, True)
        self.builder = previous_builder

    def __visit_assign_statement(self, node: AssignStatement) -> None:
        name = node.ident.value
        value = node.rh
        operator = node.operator

        var = self.env.lookup_record(name)
        if var is None:
            self.errors.append(f"identifier `{name}` reassigned before declaration")
            return
        if var.is_const:
            raise ReassignConstError(f"tried to reassign const `{name}`")
        var_ptr = var.value
 
        right_value, right_type = self.__resolve_value(value)
        
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
        
        ptr_record = self.env.lookup_record(name)
        if ptr_record is None:
            return None
        ptr = ptr_record.value
        self.builder.store(value, ptr)

    def __visit_while_statement(self, node: WhileStatement) -> None:
        test, _ = self.__resolve_value(node.condition)

        while_loop_entry = self.builder.append_basic_block(f"while_loop_entry_{self.__increment_counter()}")
        while_loop_otherwise = self.builder.append_basic_block(f"while_loop_otherwise_{self.counter}")

        self.builder.cbranch(test, while_loop_entry, while_loop_otherwise)
        self.builder.position_at_start(while_loop_entry)
        self.compile(node.body)
        test, _ = self.__resolve_value(node.condition)
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

        self.__define_struct(name, field_names, field_type_names)
    
    def __visit_enum_statement(self, node: EnumStatement) -> None:
        name = node.name.value
        variants = [variant.value for variant in node.variants]
        self.env.define_enum(name, variants)
    
    def __visit_union_statement(self, node: UnionStatement) -> None:
        name = node.name.value
        variant_names = [variant[0].value for variant in node.variants]
        variant_type_names = [variant[1] for variant in node.variants]
        
        resolved_types: list[Optional[ir.Type]] = []
        for tname in variant_type_names:
            if tname is None:
                resolved_types.append(None)
                continue
            typ = self.__resolve_type(tname)
            resolved_types.append(typ)
        
        identified = ir.global_context.get_identified_type(name) # type: ignore
        if not isinstance(identified, ir.IdentifiedStructType):
            raise Exception("this is annoying")
        
        # tag, pointer
        identified.set_body(ir.IntType(32), ir.IntType(8).as_pointer())
        self.env.define_union(name, variant_names, resolved_types, identified)
    # endregion

    # region Expressions
    def __visit_infix_expression(self, node: InfixExpression) -> tuple[ir.Value, ir.Type]:
        if node.right_node is None:
            raise CompilerException("right node is none")

        operator = node.operator
        left_value, left_type = self.__resolve_value(node.left_node)
        right_value, right_type = self.__resolve_value(node.right_node)

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
            raise NotImplementedError(f"infix operation {operator} not defined between {left_type} and {right_type}")
                
        return value, typ
    
    def __visit_block_expression(self, node: BlockExpression) -> tuple[ir.Value, ir.Type]:
        for stmt in node.statements:
            self.compile(stmt)

        if node.return_expression is not None:
            return self.__resolve_value(node.return_expression)
        else:
            return ir.Constant(self.type_map["i32"], 0), self.type_map["i32"]
    
    def __visit_if_expression(self, node: IfExpression) -> tuple[ir.Value, ir.Type]:
        cond_val, cond_type = self.__resolve_value(node.condition)

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

        if type(then_type) is not type(else_type):
            raise TypeError("Mismatched types in if branches")

        # If no branch actually branched to merge, nothing to phi — return default/None
        incoming_blocks: list[ir.Block] = []
        incoming_values: list[ir.Value] = []

        if then_block_for_phi is not None:
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


    def __visit_call_expression(self, node: CallExpression) -> tuple[ir.Instruction, ir.Type]:
        name = node.function.value
        params = node.args
        
        args: list[ir.Value] = []
        types: list[ir.Type] = []
        for param in params:
            p_val, p_typ = self.__resolve_value(param)
            args.append(p_val)
            types.append(p_typ)

        ret = None
        match name:
            case 'printf':
                ret = self.builtin_printf(args, types[0]) # pyright: ignore[reportArgumentType]
                ret_type = self.type_map['i32']
            case _:
                record = self.env.lookup_record(name)
                if record is None:
                    raise LookupError(f"couldn't lookup `{name}`")
                func = record.value
                ret_type = record.typ
                ret = self.builder.call(func, args)
        
        return ret, ret_type
    
    def __visit_prefix_expression(self, node: PrefixExpression) -> tuple[ir.Value, ir.Type]:
        op = node.operator
        if op == '&':
            if isinstance(node.right_node, IdentifierLiteral):
                record = self.env.lookup_record(node.right_node.value)
                if record is None:
                    raise ValueResolverError(f"cannot take address of undefined variable `{node.right_node.value}`")
                ptr = record.value
                typ = record.typ
                return ptr, ptr.type
            if isinstance(node.right_node, FieldAccessExpression):
                val, typ = self.__visit_field_access_expression(node.right_node, return_pointer=True)
                return val, typ
            raise ValueResolverError("address-of supports only identifiers or field access")

        if op == '*':
            ptr_to_ptr, ptr_type = self.__resolve_value(node.right_node, return_pointer=True)
            if not isinstance(ptr_type, ir.PointerType):
                raise TypeMismatchError("dereferencing a non-pointer")
            ptr = self.builder.load(ptr_to_ptr)
            value = self.builder.load(ptr)
            return value, ptr_type.pointee

        val, typ = self.__resolve_value(node.right_node)
        if isinstance(typ, ir.FloatType):
            if op == '-':
                return self.builder.fmul(val, ir.Constant(ir.FloatType(), -1.0)), typ
            if op == '!':
                return self.builder.fcmp_ordered('==', val, ir.Constant(ir.FloatType(), 0.0)), ir.IntType(1)
        if isinstance(typ, ir.IntType):
            if op == '-':
                return self.builder.mul(val, ir.Constant(ir.IntType(32), -1)), typ
            if op == '!':
                return self.builder.not_(val), ir.IntType(1)
        raise TypeError(f"unsupported prefix operator `{op}` for type {typ}")

    def __visit_new_struct_expression(self, node: NewStructExpression) -> tuple[ir.Value, ir.Type]:
        name = node.struct_ident.value
        struct_type, field_names, field_types = self.env.lookup_struct(name)
        if struct_type is None or field_names is None or field_types is None:
            raise LookupError(f"struct `{name}` not defined")

        ptr = self.builder.alloca(struct_type)

        for i, (field_expr, expected_type) in enumerate(zip((f[1] for f in node.fields), field_types)):
            val, val_type = self.__resolve_value(field_expr)

            if isinstance(val_type, ir.PointerType):
                val = self.builder.load(val)
                val_type = val.type

            if val_type != expected_type:
                raise TypeMismatchError(f"field `{field_names[i]}` expects `{expected_type}`, got `{val_type}`")

            field_ptr = self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0),
                                            ir.Constant(ir.IntType(32), i)])
            self.builder.store(val, field_ptr)

        return ptr, ir.PointerType(struct_type)

    def __visit_field_access_expression(self, node: FieldAccessExpression, return_pointer: bool = False) -> tuple[ir.Value, ir.Type]:
        base_val, base_type = self.__resolve_value(node.base, return_pointer=True)
        
        if isinstance(base_type, ir.PointerType) and isinstance(base_type.pointee, ir.IdentifiedStructType):
            struct_ptr = base_val
            struct_type = base_type.pointee
        elif isinstance(base_type, ir.IdentifiedStructType):
            struct_ptr = base_val
            struct_type = base_type
        else:
            raise TypeError(f"cannot access field on non-struct type {base_type}")

        struct_def, field_names, field_types = self.env.lookup_struct(struct_type.name)
        if struct_def is None or field_names is None or field_types is None:
            raise LookupError(f"struct `{struct_type.name}` not found")

        idx = field_names.index(node.field.value)
        field_ptr = self.builder.gep(struct_ptr, [ir.Constant(ir.IntType(32), 0),
                                                ir.Constant(ir.IntType(32), idx)])

        if return_pointer:
            return field_ptr, field_types[idx] if not isinstance(field_types[idx], ir.PointerType) else field_types[idx]
        else:
            return self.builder.load(field_ptr), field_types[idx]

    def __visit_enum_variant_access_expression(self, node: EnumVariantAccessExpression) -> tuple[ir.Value, ir.Type]:
        enum_metadata = self.env.lookup_enum(node.name.value)
        if enum_metadata is None:
            return self.__visit_union_access_expression(node)
        if node.variant.value not in enum_metadata.variants:
            raise FieldMismatchError(f"enum `{node.name.value}` doesn't have variant `{node.variant.value}`")
        idx = enum_metadata.variants.index(node.variant.value)
        return ir.Constant(ir.IntType(32), idx), ir.IntType(32)
    
    def __visit_union_access_expression(self, node: EnumVariantAccessExpression) -> tuple[ir.Value, ir.Type]:
        union_metadata = self.env.lookup_union(node.name.value)
        if union_metadata is None:
            raise LookupError(f"no enum or union with name `{node.name.value}` found")

        if node.variant.value not in union_metadata.variant_names:
            raise FieldMismatchError(f"enum `{node.name.value}` doesn't have variant `{node.variant.value}`")
        variant_index = union_metadata.variant_names.index(node.variant.value)

        union_llvm_type = union_metadata.llvm_struct
        ptr = self.builder.alloca(union_llvm_type)

        tag_ptr = self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        self.builder.store(ir.Constant(ir.IntType(32), variant_index), tag_ptr)

        payload_field_ptr = self.builder.gep(ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)])
        i8_ptr_type = ir.IntType(8).as_pointer()

        expected_type = union_metadata.variant_types[variant_index]  # Optional[ir.Type]

        if expected_type is None:
            if node.value is not None:
                raise TypeMismatchError(f"variant `{node.variant.value}` of `{node.name.value}` does not accept a payload")
            null_ptr = ir.Constant(i8_ptr_type, None)
            self.builder.store(null_ptr, payload_field_ptr)
            return ptr, ptr.type

        if node.value is None:
            raise ValueResolverError(f"variant `{node.variant.value}` of `{node.name.value}` expects a payload")

        payload_val, payload_typ = self.__resolve_value(node.value)
        
        if isinstance(payload_typ, ir.PointerType) and payload_typ.pointee == expected_type:
            payload_ptr_val = payload_val
        elif payload_typ == expected_type:
            tmp = self.builder.alloca(expected_type)
            self.builder.store(payload_val, tmp)
            payload_ptr_val = tmp
        elif isinstance(payload_typ, ir.PointerType) and expected_type == payload_typ.pointee:
            payload_ptr_val = payload_val
        else:
            raise TypeMismatchError(f"union `{node.name.value}` variant `{node.variant.value}` expects payload of type `{expected_type}`, got `{payload_typ}`")

        payload_i8 = self.builder.bitcast(payload_ptr_val, i8_ptr_type)
        self.builder.store(payload_i8, payload_field_ptr)

        return ptr, ptr.type

    def __visit_match_expression(self, node: MatchExpression) -> tuple[ir.Value, ir.Type]:
        value, typ = self.__resolve_value(node.match)
        if typ != ir.IntType(32):
            # union or illegal
            return self.__visit_match_expression_union(node)
        
        counter = self.__increment_counter()
        fn = self.builder.function

        blocks: list[tuple[int, ir.Block, ir.Value, ir.Type]] = []

        prev_block = self.builder.block

        for evae, block_expression in node.cases:
            enum = self.env.lookup_enum(evae.name.value)
            if enum is None:
                raise LookupError(f"enum named `{evae.name.value}` not found")
            if evae.variant.value not in enum.variants:
                raise FieldMismatchError(f"variant {evae.variant.value} doesn't exist in enum `{evae.name.value}`")
            variant_value = enum.variants.index(evae.variant.value)

            block = fn.append_basic_block(f"switch_{counter}_case_{variant_value}")
            self.builder.position_at_start(block)

            result, result_ty = self.__visit_block_expression(block_expression)
            blocks.append((variant_value, block, result, result_ty))

        end = fn.append_basic_block(f"switch_{counter}_end")

        self.builder.position_at_end(prev_block)
        switch = self.builder.switch(value, end)
        for variant_value, block, _, _ in blocks:
            switch.add_case(ir.Constant(ir.IntType(32), variant_value), block)

        for _, block, _, _ in blocks:
            self.builder.position_at_end(block)
            if not self.builder.block.is_terminated:
                self.builder.branch(end)

        self.builder.position_at_start(end)
        ty = blocks[0][3]
        phi = self.builder.phi(ty)
        for _, block, result, _ in blocks:
            phi.add_incoming(result, block)
        phi.add_incoming(ir.Constant(ty, ir.Undefined), prev_block)
        return phi, ty

    def __visit_match_expression_union(self, node: MatchExpression) -> tuple[ir.Value, ir.Type]:
        # Resolve the union value
        union_val, union_type = self.__resolve_value(node.match)
        
        # If passed a pointer, load the struct
        is_ptr = isinstance(union_type, ir.PointerType)
        if is_ptr:
            union_ptr = union_val
            union_val = self.builder.load(union_ptr)
            union_type = union_type.pointee
        else:
            union_ptr = self.builder.alloca(union_type)
            self.builder.store(union_val, union_ptr)

        if not isinstance(union_type, ir.IdentifiedStructType):
            raise TypeMismatchError(f"expected union struct type in match expression, got {union_type}")

        union_struct = union_type
        # Lookup the union metadata by name
        union_metadata = self.env.lookup_union(union_struct.name)
        if union_metadata is None:
            raise LookupError(f"union for struct `{union_struct}` not found")

        fn = self.builder.function
        counter = self.__increment_counter()

        # Load the tag and value pointer
        tag_ptr = self.builder.gep(union_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        tag_val = self.builder.load(tag_ptr)
        val_ptr_ptr = self.builder.gep(union_ptr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1)])
        val_ptr = self.builder.load(val_ptr_ptr)

        # Prepare blocks
        entry_block = self.builder.block
        end_block = fn.append_basic_block(f"match_{counter}_end")

        case_blocks: list[tuple[int, ir.Block, ir.Value, ir.Type]] = []

        # Create blocks for each variant
        for evae, block_expr in node.cases:
            if evae.variant.value not in union_metadata.variant_names:
                raise FieldMismatchError(f"union `{union_metadata.name}` doesn't have variant `{evae.variant.value}`")
            variant_idx = union_metadata.variant_names.index(evae.variant.value)

            case_block = fn.append_basic_block(f"match_{counter}_case_{variant_idx}")
            self.builder.position_at_start(case_block)

            # If the variant has a payload, bind it to the local variable
            payload_ty = union_metadata.variant_types[variant_idx]
            if payload_ty is not None and evae.value is not None:
                # evae.value is an IdentifierLiteral for the binding
                payload_ptr = self.builder.bitcast(val_ptr, ir.PointerType(payload_ty))
                if not isinstance(evae.value, IdentifierLiteral):
                    raise Exception("should be literal")
                self.env.define_record(evae.value.value, payload_ptr, payload_ty)

            # Visit the block
            result_val, result_ty = self.__visit_block_expression(block_expr)

            # Branch to end block if not already terminated
            if not self.builder.block.is_terminated:
                self.builder.branch(end_block)

            case_blocks.append((variant_idx, case_block, result_val, result_ty))

        # Insert switch in the entry block
        self.builder.position_at_end(entry_block)
        switch_inst = self.builder.switch(tag_val, end_block)
        for variant_idx, case_block, _, _ in case_blocks:
            switch_inst.add_case(ir.Constant(ir.IntType(32), variant_idx), case_block)

        # Build PHI node at the end
        self.builder.position_at_start(end_block)
        if not case_blocks:
            return ir.Constant(ir.IntType(32), 0), ir.IntType(32)
        
        phi_ty = case_blocks[0][3]
        phi = self.builder.phi(phi_ty)
        for _, case_block, result_val, _ in case_blocks:
            phi.add_incoming(result_val, case_block)
        # Default incoming from entry block (should never execute)
        phi.add_incoming(ir.Constant(phi_ty, ir.Undefined), entry_block)

        return phi, phi_ty
    # endregion

    # endregion

    # region Helper Methods
    def __resolve_value(
        self, node: Expression, value_type: Optional[str] = None, return_pointer: bool = False
    ) -> tuple[ir.Value, ir.Type]:
        match node.type():
            case NodeType.I32Literal:
                typ = self.type_map['i32' if value_type is None else value_type]
                return ir.Constant(typ, node.value), typ # type: ignore
            case NodeType.F32Literal:
                typ = self.type_map['f32' if value_type is None else value_type]
                return ir.Constant(typ, node.value), typ # type: ignore
            case NodeType.BooleanLiteral:
                return ir.Constant(ir.IntType(1), 1 if node.value else 0), ir.IntType(1) # type: ignore
            case NodeType.StringLiteral:
                return self.__convert_string(node.value) # type: ignore
            case NodeType.IdentifierLiteral:
                record = self.env.lookup_record(node.value) # type: ignore
                if record is None:
                    raise ValueResolverError(f"identifier `{node.value}` not found") # type: ignore
                ptr = record.value
                typ = record.typ

                if return_pointer:
                    return ptr, typ

                loaded = self.builder.load(ptr)
                
                if isinstance(typ, ir.PointerType):
                    if not return_pointer and isinstance(loaded.type, ir.PointerType) and isinstance(loaded.type.pointee, (ir.IntType, ir.FloatType, ir.IntType)):
                        return self.builder.load(loaded), loaded.type.pointee
                    return loaded, loaded.type
                else:
                    return loaded, typ
            
            case NodeType.InfixExpression:
                return self.__visit_infix_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.BlockExpression:
                return self.__visit_block_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.IfExpression:
                return self.__visit_if_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.CallExpression:
                return self.__visit_call_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.PrefixExpression:
                return self.__visit_prefix_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.NewStructExpression:
                # Always return pointer to struct, do not load
                return self.__visit_new_struct_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.FieldAccessExpression:
                # Load field if it's a primitive, otherwise return pointer
                return self.__visit_field_access_expression(node, return_pointer=return_pointer) # pyright: ignore[reportArgumentType]
            case NodeType.EnumVariantAccessExpression:
                return self.__visit_enum_variant_access_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.MatchExpression:
                return self.__visit_match_expression(node) # pyright: ignore[reportArgumentType]

            case _:
                raise NotImplementedError(f"not implemented: {node.type().name}")


    
    def __convert_string(self, string: str) -> tuple[ir.Value, ir.Type]:
        string = string.replace("\\n", "\n\0")
        fmt = f"{string}\0"
        c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)), bytearray(fmt.encode("utf8")))
        global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name=f'__str_{self.__increment_counter()}')
        global_fmt.linkage = 'internal'
        global_fmt.global_constant = True
        global_fmt.initializer = c_fmt

        gep = self.builder.gep(global_fmt, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        ptr = self.builder.bitcast(gep, ir.IntType(8).as_pointer())
        return ptr, ir.IntType(8).as_pointer()


    def builtin_printf(self, params: list[ir.Value], return_type: ir.Type) -> instructions.CallInstr:
        record = self.env.lookup_record('printf')
        if record is None or len(params) == 0:
           raise LookupError("couldn't lookup `printf` or params was none") 
        func = record.value

        fmt_val = params[0]
        rest_params = params[1:]

        # If global string, bitcast [N x i8]* to i8*
        if isinstance(fmt_val, ir.GlobalVariable):
            fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # If already pointer type, bitcast to i8* just in case
        if isinstance(fmt_val.type, ir.PointerType):
            fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # If it's a load instruction from a global
        if isinstance(fmt_val, instructions.LoadInstr):
            operand0 = fmt_val.operands[0]
            fmt_arg = self.builder.bitcast(operand0, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])

        # Fallback: try to bitcast
        # Can raise
        fmt_arg = self.builder.bitcast(fmt_val, ir.IntType(8).as_pointer())
        return self.builder.call(func, [fmt_arg, *rest_params])

    def __extract_ret(self, node: BlockExpression):
        if node.return_expression is not None:
            node.statements.append(ReturnStatement(node.return_expression))
            node.return_expression = None
    
    def __resolve_type(self, name: str) -> ir.Type:
        if name.startswith('&'):
            # Pointer
            inner_name = name[1:]
            inner = self.__resolve_type(inner_name)
            return ir.PointerType(inner)
        elif name in self.type_map:
            # Primitive
            return self.type_map[name]
        else:
            # Struct
            struct, _, _ = self.env.lookup_struct(name)
            if struct is not None:
                return struct
            
            # Enum
            enum = self.env.lookup_enum(name)
            if enum is not None:
                return ir.IntType(32)
            
            # Union
            union = self.env.lookup_union(name)
            if union is not None:
                return union.llvm_struct
            
            # None found
            raise TypeNotFoundError(f"couldn't resolve type of `{name}`")

    def __define_struct(
        self, name: str, field_names: list[str], field_type_names: list[str]
    ) -> ir.IdentifiedStructType:
        identified = ir.global_context.get_identified_type(name) # type: ignore
        if not isinstance(identified, ir.IdentifiedStructType):
            raise Exception("this is annoying")

        resolved_types: list[ir.Type] = []
        for tname in field_type_names:
            typ = self.__resolve_type(tname)
            resolved_types.append(typ)

        identified.set_body(*resolved_types)
        
        self.env.define_struct(name, identified, field_names, resolved_types)
        return identified
    # endregion
