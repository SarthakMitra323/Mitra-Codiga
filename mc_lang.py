# Mitra Codiga - A small, expressive interpreter implemented in pure Python
# Features: numbers, strings, variables, arithmetic, comparisons, boolean logic,
# if/elif/else, while loops, function definitions and calls, return statements,
# blocks with { ... }, semicolons or newlines as statement separators, and a few built-ins.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Tuple

##############################################################
# LEXER
##############################################################

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_STRING = 'STRING'
TT_IDENTIFIER = 'IDENT'
TT_KEYWORD = 'KW'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_EQ = 'EQ'              # '='
TT_EE = 'EE'              # '=='
TT_NE = 'NE'              # '!='
TT_LT = 'LT'              # '<'
TT_LTE = 'LTE'            # '<='
TT_GT = 'GT'              # '>'
TT_GTE = 'GTE'            # '>='
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_LBRACE = 'LBRACE'
TT_RBRACE = 'RBRACE'
TT_COMMA = 'COMMA'
TT_SEMI = 'SEMI'
TT_NEWLINE = 'NEWLINE'
TT_EOF = 'EOF'

KEYWORDS = {
    'let', 'if', 'elif', 'else', 'while', 'fun', 'return',
    'true', 'false', 'null', 'and', 'or', 'not'
}

@dataclass
class Pos:
    idx: int
    ln: int
    col: int
    fn: str
    ftxt: str

    def advance(self, current_char: Optional[str] = None):
        self.idx += 1
        self.col += 1
        if current_char == '\n':
            self.ln += 1
            self.col = 0
        return self

    def copy(self) -> 'Pos':
        return Pos(self.idx, self.ln, self.col, self.fn, self.ftxt)

@dataclass
class Token:
    type: str
    value: Any = None
    pos_start: Optional[Pos] = None
    pos_end: Optional[Pos] = None

    def matches(self, type_: str, value: Any) -> bool:
        return self.type == type_ and self.value == value

    def __repr__(self) -> str:
        return f"{self.type}:{self.value}" if self.value is not None else self.type

class LexerError(Exception):
    def __init__(self, pos_start: Pos, pos_end: Pos, details: str):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.details = details
        super().__init__(self.__str__())

    def __str__(self):
        snippet = get_snippet(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return f"Illegal Character: {self.details}\nFile {self.pos_start.fn}, line {self.pos_start.ln+1}\n{snippet}"

class Lexer:
    def __init__(self, fn: str, text: str):
        self.fn = fn
        self.text = text
        self.pos = Pos(-1, 0, -1, fn, text)
        self.current_char: Optional[str] = None
        self.advance()

    def advance(self):
        if self.current_char is not None:
            self.pos.advance(self.current_char)
        else:
            self.pos.advance()
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def peek(self) -> Optional[str]:
        nxt_idx = self.pos.idx + 1
        return self.text[nxt_idx] if nxt_idx < len(self.text) else None

    def make_tokens(self) -> List[Token]:
        tokens: List[Token] = []
        while self.current_char is not None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char == '#':
                self.skip_comment()
            elif self.current_char == '\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char.isdigit():
                tokens.append(self.make_number())
            elif self.current_char.isalpha() or self.current_char == '_':
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '*':
                if self.peek() == '*':
                    pos_start = self.pos.copy(); self.advance(); self.advance()
                    tokens.append(Token(TT_POW, pos_start=pos_start, pos_end=self.pos.copy()))
                else:
                    tokens.append(Token(TT_MUL, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                    self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '{':
                tokens.append(Token(TT_LBRACE, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '}':
                tokens.append(Token(TT_RBRACE, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == ';':
                tokens.append(Token(TT_SEMI, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
                self.advance()
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '!':
                tokens.append(self.make_not_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less())
            elif self.current_char == '>':
                tokens.append(self.make_greater())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                raise LexerError(pos_start, self.pos.copy(), f"'{char}'")
        tokens.append(Token(TT_EOF, pos_start=self.pos.copy(), pos_end=self.pos.copy()))
        return tokens

    def skip_comment(self):
        while self.current_char is not None and self.current_char != '\n':
            self.advance()

    def make_number(self) -> Token:
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()
        if num_str.startswith('.') or num_str.endswith('.'):
            num_str = num_str.strip('.')
            dot_count = 0 if num_str.isdigit() else 1
        tok_type = TT_FLOAT if dot_count == 1 else TT_INT
        value = float(num_str) if tok_type == TT_FLOAT else int(num_str)
        return Token(tok_type, value, pos_start, self.pos.copy())

    def make_identifier(self) -> Token:
        id_str = ''
        pos_start = self.pos.copy()
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            id_str += self.current_char
            self.advance()
        if id_str in KEYWORDS:
            return Token(TT_KEYWORD, id_str, pos_start, self.pos.copy())
        return Token(TT_IDENTIFIER, id_str, pos_start, self.pos.copy())

    def make_string(self) -> Token:
        pos_start = self.pos.copy()
        self.advance()  # skip opening quote
        s = ''
        escape = False
        escapes = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}
        while self.current_char is not None and (escape or self.current_char != '"'):
            ch = self.current_char
            if escape:
                s += escapes.get(ch, ch)
                escape = False
            else:
                if ch == '\\':
                    escape = True
                else:
                    s += ch
            self.advance()
        if self.current_char != '"':
            raise LexerError(pos_start, self.pos.copy(), 'Unterminated string')
        self.advance()  # skip closing quote
        return Token(TT_STRING, s, pos_start, self.pos.copy())

    def make_equals(self) -> Token:
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_EE, pos_start=pos_start, pos_end=self.pos.copy())
        return Token(TT_EQ, pos_start=pos_start, pos_end=self.pos.copy())

    def make_not_equals(self) -> Token:
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos.copy())
        raise LexerError(pos_start, self.pos.copy(), "Expected '=' after '!' for '!='")

    def make_less(self) -> Token:
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_LTE, pos_start=pos_start, pos_end=self.pos.copy())
        return Token(TT_LT, pos_start=pos_start, pos_end=self.pos.copy())

    def make_greater(self) -> Token:
        pos_start = self.pos.copy()
        self.advance()
        if self.current_char == '=':
            self.advance()
            return Token(TT_GTE, pos_start=pos_start, pos_end=self.pos.copy())
        return Token(TT_GT, pos_start=pos_start, pos_end=self.pos.copy())

##############################################################
# PARSER
##############################################################

class ParseError(Exception):
    def __init__(self, token: Token, details: str):
        self.token = token
        self.details = details
        super().__init__(self.__str__())

    def __str__(self):
        ps = self.token.pos_start or Pos(0,0,0,'<unknown>','')
        pe = self.token.pos_end or ps
        snippet = get_snippet(ps.ftxt, ps, pe)
        return f"Parse Error: {self.details}\nFile {ps.fn}, line {ps.ln+1}\n{snippet}"

# AST Nodes
@dataclass
class NumberNode:
    tok: Token

@dataclass
class StringNode:
    tok: Token

@dataclass
class VarAccessNode:
    name_tok: Token

@dataclass
class VarAssignNode:
    name_tok: Token
    value_node: Any

@dataclass
class BinOpNode:
    left: Any
    op_tok: Token
    right: Any

@dataclass
class UnaryOpNode:
    op_tok: Token
    node: Any

@dataclass
class IfCase:
    cond: Any
    body: Any

@dataclass
class IfNode:
    cases: List[IfCase]
    else_case: Optional[Any]

@dataclass
class WhileNode:
    cond: Any
    body: Any

@dataclass
class FuncDefNode:
    name_tok: Optional[Token]
    arg_name_toks: List[Token]
    body_node: Any

@dataclass
class CallNode:
    node_to_call: Any
    arg_nodes: List[Any]

@dataclass
class ReturnNode:
    node_to_return: Optional[Any]

@dataclass
class BlockNode:
    statements: List[Any]

@dataclass
class NoOpNode:
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.tok_idx = -1
        self.current_tok: Token = tokens[0]
        self.advance()

    def advance(self) -> Token:
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self) -> BlockNode:
        statements: List[Any] = []
        while self.current_tok.type == TT_NEWLINE:
            self.advance()
        stmt = self.statement()
        statements.append(stmt)
        while self.current_tok.type in (TT_SEMI, TT_NEWLINE):
            self.advance()
            while self.current_tok.type == TT_NEWLINE:
                self.advance()
            if self.current_tok.type in (TT_EOF, TT_RBRACE):
                break
            statements.append(self.statement())
        return BlockNode(statements)

    def statement(self):
        tok = self.current_tok
        if tok.type == TT_KEYWORD and tok.value == 'return':
            self.advance()
            if self.current_tok.type in (TT_SEMI, TT_NEWLINE, TT_RBRACE, TT_EOF):
                return ReturnNode(None)
            expr = self.expr()
            return ReturnNode(expr)
        if tok.type == TT_KEYWORD and tok.value == 'if':
            return self.if_expr()
        if tok.type == TT_KEYWORD and tok.value == 'while':
            return self.while_expr()
        if tok.type == TT_KEYWORD and tok.value == 'fun':
            return self.func_def()
        if tok.type == TT_LBRACE:
            return self.block()
        if tok.type == TT_KEYWORD and tok.value == 'let':
            self.advance()
            if self.current_tok.type != TT_IDENTIFIER:
                raise ParseError(self.current_tok, "Expected identifier after 'let'")
            name_tok = self.current_tok
            self.advance()
            if self.current_tok.type != TT_EQ:
                raise ParseError(self.current_tok, "Expected '=' after identifier in declaration")
            self.advance()
            value = self.expr()
            return VarAssignNode(name_tok, value)
        # assignment or expression
        if tok.type == TT_IDENTIFIER and self.peek_type() == TT_EQ:
            name_tok = tok
            self.advance()  # name
            self.advance()  # '='
            value = self.expr()
            return VarAssignNode(name_tok, value)
        return self.expr()

    def block(self) -> BlockNode:
        # assumes current token is '{'
        self.advance()
        statements: List[Any] = []
        while self.current_tok.type in (TT_NEWLINE, TT_SEMI):
            self.advance()
        while self.current_tok.type not in (TT_RBRACE, TT_EOF):
            statements.append(self.statement())
            if self.current_tok.type in (TT_SEMI, TT_NEWLINE):
                while self.current_tok.type in (TT_SEMI, TT_NEWLINE):
                    self.advance()
            else:
                # optional separators inside block
                pass
        if self.current_tok.type != TT_RBRACE:
            raise ParseError(self.current_tok, "Expected '}' to close block")
        self.advance()
        return BlockNode(statements)

    def if_expr(self) -> IfNode:
        # 'if' expr block_or_stmt ('elif' expr block_or_stmt)* ('else' block_or_stmt)?
        self.advance()  # skip 'if'
        cond = self.expr()
        body = self.block_or_single_stmt()
        cases = [IfCase(cond, body)]
        while self.current_tok.type == TT_KEYWORD and self.current_tok.value == 'elif':
            self.advance()
            cond = self.expr()
            body = self.block_or_single_stmt()
            cases.append(IfCase(cond, body))
        else_case = None
        if self.current_tok.type == TT_KEYWORD and self.current_tok.value == 'else':
            self.advance()
            else_case = self.block_or_single_stmt()
        return IfNode(cases, else_case)

    def while_expr(self) -> WhileNode:
        self.advance()  # skip 'while'
        cond = self.expr()
        body = self.block_or_single_stmt()
        return WhileNode(cond, body)

    def func_def(self) -> FuncDefNode:
        self.advance()  # skip 'fun'
        name_tok: Optional[Token] = None
        if self.current_tok.type == TT_IDENTIFIER:
            name_tok = self.current_tok
            self.advance()
        if self.current_tok.type != TT_LPAREN:
            raise ParseError(self.current_tok, "Expected '('")
        self.advance()
        arg_name_toks: List[Token] = []
        if self.current_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.current_tok)
            self.advance()
            while self.current_tok.type == TT_COMMA:
                self.advance()
                if self.current_tok.type != TT_IDENTIFIER:
                    raise ParseError(self.current_tok, 'Expected parameter name')
                arg_name_toks.append(self.current_tok)
                self.advance()
        if self.current_tok.type != TT_RPAREN:
            raise ParseError(self.current_tok, "Expected ')'")
        self.advance()
        body = self.block_or_single_stmt()
        return FuncDefNode(name_tok, arg_name_toks, body)

    def block_or_single_stmt(self) -> BlockNode:
        if self.current_tok.type == TT_LBRACE:
            return self.block()
        # single statement converted into a block
        stmt = self.statement()
        return BlockNode([stmt])

    def expr(self):
        return self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or')))

    def comp_expr(self):
        if self.current_tok.type == TT_KEYWORD and self.current_tok.value == 'not':
            op_tok = self.current_tok
            self.advance()
            node = self.comp_expr()
            return UnaryOpNode(op_tok, node)
        node = self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_LTE, TT_GT, TT_GTE))
        return node

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def term(self):
        return self.bin_op(self.power, (TT_MUL, TT_DIV))

    def power(self):
        return self.bin_op(self.factor, (TT_POW,), right_assoc=True)

    def factor(self):
        tok = self.current_tok
        if tok.type in (TT_PLUS, TT_MINUS):
            self.advance()
            node = self.factor()
            return UnaryOpNode(tok, node)
        return self.call()

    def call(self):
        atom = self.atom()
        while self.current_tok.type == TT_LPAREN:
            self.advance()
            arg_nodes: List[Any] = []
            if self.current_tok.type != TT_RPAREN:
                arg_nodes.append(self.expr())
                while self.current_tok.type == TT_COMMA:
                    self.advance()
                    arg_nodes.append(self.expr())
            if self.current_tok.type != TT_RPAREN:
                raise ParseError(self.current_tok, "Expected ')' after arguments")
            self.advance()
            atom = CallNode(atom, arg_nodes)
        return atom

    def atom(self):
        tok = self.current_tok
        if tok.type in (TT_INT, TT_FLOAT):
            self.advance()
            return NumberNode(tok)
        if tok.type == TT_STRING:
            self.advance()
            return StringNode(tok)
        if tok.type == TT_IDENTIFIER:
            self.advance()
            return VarAccessNode(tok)
        if tok.type == TT_KEYWORD:
            if tok.value in ('true', 'false', 'null'):
                self.advance()
                return VarAccessNode(tok)  # handled as keywords in runtime
            raise ParseError(tok, f"Unexpected keyword '{tok.value}'")
        if tok.type == TT_LPAREN:
            self.advance()
            expr = self.expr()
            if self.current_tok.type != TT_RPAREN:
                raise ParseError(self.current_tok, "Expected ')'")
            self.advance()
            return expr
        if tok.type == TT_LBRACE:
            return self.block()
        raise ParseError(tok, 'Expected expression')

    def bin_op(self, func, ops, right_assoc: bool=False):
        left = func()
        if right_assoc:
            while True:
                tok = self.current_tok
                if (isinstance(ops, tuple) or isinstance(ops, list)) and any(
                    (isinstance(op, tuple) and tok.type == op[0] and tok.value == op[1]) or (isinstance(op, str) and tok.type == op)
                    for op in ops
                ):
                    self.advance()
                    right = func()
                    left = BinOpNode(left, tok, right)
                else:
                    break
            return left
        while True:
            tok = self.current_tok
            if (isinstance(ops, tuple) or isinstance(ops, list)) and any(
                (isinstance(op, tuple) and tok.type == op[0] and tok.value == op[1]) or (isinstance(op, str) and tok.type == op)
                for op in ops
            ):
                self.advance()
                right = func()
                left = BinOpNode(left, tok, right)
            else:
                break
        return left

    def peek_type(self) -> Optional[str]:
        nxt_idx = self.tok_idx + 1
        if 0 <= nxt_idx < len(self.tokens):
            return self.tokens[nxt_idx].type
        return None

##############################################################
# RUNTIME
##############################################################

class RTError(Exception):
    def __init__(self, details: str):
        super().__init__(details)

class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class Function:
    def __init__(self, name: Optional[str], arg_names: List[str], body: BlockNode, parent: 'SymbolTable'):
        self.name = name
        self.arg_names = arg_names
        self.body = body
        self.parent = parent

    def __call__(self, interpreter: 'Interpreter', args: List[Any]):
        if len(args) != len(self.arg_names):
            raise RTError(f"Function expected {len(self.arg_names)} args, got {len(args)}")
        symbols = SymbolTable(self.parent)
        for n, v in zip(self.arg_names, args):
            symbols.set(n, v)
        return interpreter.exec_block(self.body, symbols)

    def __repr__(self):
        return f"<fun {self.name or '<anon>'}({', '.join(self.arg_names)})>"

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable']=None):
        self.symbols: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.get(name)
        raise RTError(f"Undefined name '{name}'")

    def set(self, name: str, value: Any):
        self.symbols[name] = value

class Interpreter:
    def __init__(self):
        self.globals = SymbolTable()
        self.inject_builtins(self.globals)

    def inject_builtins(self, syms: SymbolTable):
        # Python functions wrapped directly
        syms.set('print', lambda *vals: print(*vals))
        syms.set('len', lambda x: len(x))
        syms.set('input', lambda prompt='': input(prompt))
        syms.set('int', lambda x: int(x))
        syms.set('float', lambda x: float(x))
        syms.set('str', lambda x: str(x))
        syms.set('true', True)
        syms.set('false', False)
        syms.set('null', None)

    def run(self, node: BlockNode, symbols: Optional[SymbolTable]=None) -> Any:
        if symbols is None:
            symbols = self.globals
        return self.exec_block(node, symbols)

    def exec_block(self, block: BlockNode, symbols: SymbolTable) -> Any:
        last = None
        try:
            for stmt in block.statements:
                last = self.visit(stmt, symbols)
            return last
        except ReturnSignal as rs:
            return rs.value

    def visit(self, node: Any, symbols: SymbolTable) -> Any:
        t = type(node)
        if t is NumberNode:
            return node.tok.value
        if t is StringNode:
            return node.tok.value
        if t is VarAccessNode:
            name = node.name_tok.value if node.name_tok.type == TT_IDENTIFIER else node.name_tok.value
            return symbols.get(name)
        if t is VarAssignNode:
            name = node.name_tok.value
            val = self.visit(node.value_node, symbols)
            symbols.set(name, val)
            return val
        if t is BinOpNode:
            left = self.visit(node.left, symbols)
            right = self.visit(node.right, symbols)
            return self.apply_binop(left, node.op_tok, right)
        if t is UnaryOpNode:
            val = self.visit(node.node, symbols)
            if node.op_tok.type == TT_MINUS:
                return -val
            if node.op_tok.type == TT_PLUS:
                return +val
            if node.op_tok.matches(TT_KEYWORD, 'not'):
                return not truthy(val)
            raise RTError('Unknown unary op')
        if t is IfNode:
            for case in node.cases:
                if truthy(self.visit(case.cond, symbols)):
                    return self.exec_block(case.body, SymbolTable(symbols))
            if node.else_case:
                return self.exec_block(node.else_case, SymbolTable(symbols))
            return None
        if t is WhileNode:
            last = None
            while truthy(self.visit(node.cond, symbols)):
                # Execute while-body in the same scope so assignments persist
                last = self.exec_block(node.body, symbols)
            return last
        if t is FuncDefNode:
            name = node.name_tok.value if node.name_tok else None
            args = [t.value for t in node.arg_name_toks]
            fn = Function(name, args, node.body_node, symbols)
            if name:
                symbols.set(name, fn)
            return fn
        if t is CallNode:
            callee = self.visit(node.node_to_call, symbols)
            args = [self.visit(a, symbols) for a in node.arg_nodes]
            if isinstance(callee, Function):
                return callee(self, args)
            if callable(callee):
                return callee(*args)
            raise RTError('Attempted to call a non-function value')
        if t is ReturnNode:
            val = self.visit(node.node_to_return, symbols) if node.node_to_return is not None else None
            raise ReturnSignal(val)
        if t is BlockNode:
            return self.exec_block(node, SymbolTable(symbols))
        if t is NoOpNode:
            return None
        raise RTError(f'No visit method for node type {t.__name__}')

    def apply_binop(self, left: Any, op_tok: Token, right: Any) -> Any:
        t = op_tok.type
        if t == TT_PLUS:
            return left + right
        if t == TT_MINUS:
            return left - right
        if t == TT_MUL:
            return left * right
        if t == TT_DIV:
            return left / right
        if t == TT_POW:
            return left ** right
        if t == TT_EE:
            return left == right
        if t == TT_NE:
            return left != right
        if t == TT_LT:
            return left < right
        if t == TT_LTE:
            return left <= right
        if t == TT_GT:
            return left > right
        if t == TT_GTE:
            return left >= right
        if op_tok.matches(TT_KEYWORD, 'and'):
            return truthy(left) and truthy(right)
        if op_tok.matches(TT_KEYWORD, 'or'):
            return truthy(left) or truthy(right)
        raise RTError(f'Unknown operator {op_tok}')

##############################################################
# ENTRY POINT
##############################################################

def truthy(val: Any) -> bool:
    return bool(val)

def get_snippet(text: str, pos_start: Pos, pos_end: Pos) -> str:
    try:
        lines = text.splitlines()
        line_idx = max(0, min(pos_start.ln, len(lines) - 1))
        line = lines[line_idx] if lines else ''
        start_col = max(0, pos_start.col)
        end_col = max(start_col + 1, (pos_end.col if pos_end and pos_end.ln == pos_start.ln else start_col + 1))
        caret_line = ' ' * start_col + ('^' if end_col <= start_col + 1 else '^' * (end_col - start_col))
        # Include 1-based line number
        return f"{line_idx+1} | {line}\n    {caret_line}"
    except Exception:
        return "(no snippet)"

def run(code: str, fn: str = '<stdin>') -> Tuple[Any, Optional[str]]:
    # Lex
    try:
        tokens = Lexer(fn, code).make_tokens()
    except LexerError as le:
        return None, str(le)
    # Parse
    try:
        parser = Parser(tokens)
        tree = parser.parse()
    except ParseError as pe:
        return None, str(pe)
    # Interpret
    try:
        result = Interpreter().run(tree)
        return result, None
    except RTError as rte:
        return None, f"Runtime Error: {rte}"
    except Exception as ex:
        return None, f"Runtime Error: {type(ex).__name__}: {ex}"

# Tiny CLI when executed directly
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        res, err = run(src, path)
        if err:
            print(err)
            sys.exit(1)
        if res is not None:
            print(res)
    else:
        print("Mitra Codiga REPL. End statements with newline or ';'. Ctrl+C to exit.")
        while True:
            try:
                line = input('mitra > ')
            except KeyboardInterrupt:
                print() ; break
            if not line.strip():
                continue
            res, err = run(line)
            if err:
                print(err)
            elif res is not None:
                print(res)
