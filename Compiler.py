from llvmlite import ir
from typing import Optional

from llvmlite.ir import instructions

from AST import Node, NodeType, Statement, Expression, Program
from AST import FunctionParameter
from AST import ExpressionStatement, LetStatement, FunctionStatement, ReturnStatement, AssignStatement, WhileStatement
from AST import InfixExpression, BlockExpression, IfExpression, CallExpression
from AST import I32Literal, F32Literal, IdentifierLiteral, BooleanLiteral, StringLiteral

from Environment import Environment

class Compiler:
    def __init__(self) -> None:
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

            # Expressions
            case NodeType.InfixExpression:
                self.__visit_infix_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.BlockExpression:
                self.__visit_block_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.IfExpression:
                self.__visit_if_expression(node) # pyright: ignore[reportArgumentType]
            case NodeType.CallExpression:
                self.__visit_call_expression(node) # pyright: ignore[reportArgumentType]

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
        _value_type = node.value_type # TODO: implement

        value, typ = self.__resolve_value(value)
        if value is None or typ is None:
            return

        if self.env.lookup(name)[0] is None:
            ptr = self.builder.alloca(typ)
            self.builder.store(value, ptr)
            self.env.define(name, ptr, typ)
        else:
            ptr, _ = self.env.lookup(name)
            if ptr is None:
                return
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

        value, _typ = self.__resolve_value(value)
        if value is None:
            return

        ptr, _ = self.env.lookup(name)
        if ptr is None:
            self.errors.append(f"identifier `{name}` reassigned before declaration")
        else:
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
        # node.condition, node.consequence, node.alternative
        cond_node = node.condition
        cons_node = node.consequence
        alt_node = node.alternative

        # Evaluate condition -> get LLVM value and its type
        cond_val, cond_type = self.__resolve_value(cond_node)
        if cond_val is None or cond_type is None:
            return None, None

        # Convert condition to an i1 (boolean) if necessary
        _i1 = ir.IntType(1)
        if isinstance(cond_type, ir.IntType):
            zero = ir.Constant(cond_type, 0)
            test = self.builder.icmp_signed('!=', cond_val, zero)
        elif isinstance(cond_type, ir.FloatType):
            zero = ir.Constant(cond_type, 0.0)
            test = self.builder.fcmp_ordered('!=', cond_val, zero)
        elif isinstance(cond_type, ir.IntType) and cond_type.width == 1:
            test = cond_val  # already i1
        else:
            raise TypeError("Unsupported condition type for if expression")

        # Create blocks
        func = self.builder.function
        then_bb = func.append_basic_block("if.then")
        else_bb = func.append_basic_block("if.else") if alt_node is not None else func.append_basic_block("if.else_tmp")
        merge_bb = func.append_basic_block("if.end")

        # Emit conditional branch
        self.builder.cbranch(test, then_bb, else_bb)

        # --- then branch ---
        self.builder.position_at_end(then_bb)
        then_val = None
        then_type = None
        # compile consequence and try to obtain value/type
        # If consequence is a block/expression that returns a value, use __resolve_value
        # otherwise compile it for side-effects (and treat as producing no value)
        then_val, then_type = self.__resolve_value(cons_node)

        # If the then block is not terminated (no return/ret emitted), branch to merge.
        # Best-effort check for terminator: some llvmlite versions provide builder.block.terminator
        term = getattr(self.builder.block, "terminator", None)
        if term is None:
            # no terminator -> add branch to merge
            self.builder.branch(merge_bb)

        # capture the block object for phi incoming
        then_block = self.builder.block

        # --- else branch ---
        self.builder.position_at_end(else_bb)
        else_val = None
        else_type = None
        if alt_node is not None:
            else_val, else_type = self.__resolve_value(alt_node)
        else:
            # no alternative: produce a default value matching the consequence type (if any)
            if then_type is None:
                # Nothing to return in either branch
                else_val, else_type = None, None
            else:
                # default zero value for the type
                if isinstance(then_type, ir.IntType):
                    else_val = ir.Constant(then_type, 0)
                    else_type = then_type
                elif isinstance(then_type, ir.FloatType):
                    else_val = ir.Constant(then_type, 0.0)
                    else_type = then_type
                else:
                    raise TypeError("Unsupported branch result type for default else")

        # If else block is not terminated, branch to merge
        term = getattr(self.builder.block, "terminator", None)
        if term is None:
            self.builder.branch(merge_bb)

        else_block = self.builder.block

        # --- merge block ---
        self.builder.position_at_end(merge_bb)

        # If both branches produced no value, the if-expression yields no value.
        if then_type is None and else_type is None:
            return None, None

        # If one branch had None but the other had a type, try to use the other's type as result:
        # (we already created default else for missing alt, so mismatch should be rare)
        if then_type is None and else_type is not None:
            then_type = else_type
        if else_type is None and then_type is not None:
            else_type = then_type

        # ensure the branch result types match
        if type(then_type) is not type(else_type):
            raise TypeError("Mismatched types in if branches: then %s vs else %s" % (then_type, else_type))

        # create phi node to select the branch result
        phi = self.builder.phi(then_type) # pyright: ignore[reportArgumentType]
        # add incoming values. Use the block objects captured earlier
        if then_val is None:
            # if then had no value, use zero default
            if isinstance(then_type, ir.IntType):
                then_val = ir.Constant(then_type, 0)
            elif isinstance(then_type, ir.FloatType):
                then_val = ir.Constant(then_type, 0.0)
        if else_val is None:
            if isinstance(else_type, ir.IntType):
                else_val = ir.Constant(else_type, 0)
            elif isinstance(else_type, ir.FloatType):
                else_val = ir.Constant(else_type, 0.0)

        phi.add_incoming(then_val, then_block) # pyright: ignore[reportArgumentType]
        phi.add_incoming(else_val, else_block) # pyright: ignore[reportArgumentType]

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

    def builtin_printf(self, params: list[ir.Instruction], return_type: ir.Type) -> Optional[instructions.CallInstr]:
        """ Basic C builtin printf """
        func, _ = self.env.lookup('printf')
        if func is None:
            return 

        c_str = self.builder.alloca(return_type)
        self.builder.store(params[0], c_str)

        rest_params = params[1:]

        if isinstance(params[0], ir.LoadInstr):
            """ Printing from a variable load instruction """
            # let a: str = "yeet";
            # print(a)
            c_fmt: ir.LoadInstr = params[0]
            g_var_ptr = c_fmt.operands[0]
            string_val = self.builder.load(g_var_ptr)
            fmt_arg = self.builder.bitcast(string_val, ir.IntType(8).as_pointer())
            return self.builder.call(func, [fmt_arg, *rest_params])
        else:
            """ Printing from a normal string declared within printf """
            # print("yeet %i", 23)
            # TODO: HANDLE PRINTING FLOATS
            fmt_arg = self.builder.bitcast(self.module.get_global(f"__str_{self.counter}"), ir.IntType(8).as_pointer())

            return self.builder.call(func, [fmt_arg, *rest_params])

    def __extract_ret(self, node: BlockExpression):
        if node.return_expression is not None:
            node.statements.append(ReturnStatement(node.return_expression))
            node.return_expression = None
    # endregion
