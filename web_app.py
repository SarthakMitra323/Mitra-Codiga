from flask import Flask, render_template, request, jsonify
from mc_lang import run, LexerError, ParseError, RTError
import traceback

class InputNeeded(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/execute', methods=['POST'])
def execute_code():
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'})
        
        # Create a custom interpreter with output capture
        from mc_lang import Lexer, Parser, Interpreter, SymbolTable
        
        class WebInterpreter(Interpreter):
            def __init__(self):
                super().__init__()
                self.captured_output = []
                self.input_prompts = []
                self.input_values = data.get('inputs', [])
                self.input_index = 0
            
            def inject_builtins(self, syms: SymbolTable):
                # Capture print output instead of printing to console
                def web_print(*vals):
                    output_text = ' '.join(str(val) for val in vals)
                    self.captured_output.append(output_text)
                
                # Handle input requests
                def web_input(prompt=''):
                    if self.input_index < len(self.input_values):
                        # Use provided input value
                        value = self.input_values[self.input_index]
                        self.input_index += 1
                        self.captured_output.append(f"{prompt}{value}")
                        return str(value)
                    else:
                        # Request more input
                        self.input_prompts.append(prompt)
                        raise InputNeeded(f"Input needed: {prompt}")
                
                # Add all built-ins with custom print and input
                syms.set('print', web_print)
                syms.set('input', web_input)
                syms.set('len', lambda x: len(x))
                syms.set('int', lambda x: int(x))
                syms.set('float', lambda x: float(x))
                syms.set('str', lambda x: str(x))
                syms.set('type', lambda x: type(x).__name__)
                syms.set('range', lambda *args: list(range(*args)))
                syms.set('map', lambda f, lst: list(map(f, lst)))
                syms.set('filter', lambda f, lst: list(filter(f, lst)))
                syms.set('sum', lambda lst: sum(lst))
                syms.set('min', lambda *args: min(*args) if len(args) > 1 else min(args[0]) if len(args) == 1 else (_ for _ in ()).throw(ValueError("min() arg is an empty sequence")))
                syms.set('max', lambda *args: max(*args) if len(args) > 1 else max(args[0]) if len(args) == 1 and len(args[0]) > 0 else (_ for _ in ()).throw(ValueError("max() arg is an empty sequence")))
                syms.set('abs', lambda x: abs(x))
                syms.set('round', lambda x, n=0: round(x, n))
                syms.set('sorted', lambda lst, reverse=False: sorted(lst, reverse=reverse))
                syms.set('reversed', lambda lst: list(reversed(lst)))
                syms.set('enumerate', lambda lst: list(enumerate(lst)))
                syms.set('zip', lambda *lsts: list(zip(*lsts)))
                syms.set('all', lambda lst: all(lst))
                syms.set('any', lambda lst: any(lst))
                syms.set('true', True)
                syms.set('false', False)
                syms.set('null', None)
                # Stdlib (restricted for web: file IO disabled for safety)
                import json, time
                syms.set('json_parse', lambda s: json.loads(s))
                syms.set('json_stringify', lambda v: json.dumps(v))
                syms.set('now', lambda: time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()))
                syms.set('timestamp', lambda: int(time.time()))
                syms.set('sleep', lambda sec: time.sleep(sec))
                # File IO placeholders (return error message strings)
                syms.set('read_file', lambda path: f"Error: read_file disabled in web sandbox: {path}")
                syms.set('write_file', lambda path, text: f"Error: write_file disabled in web sandbox: {path}")
                syms.set('append_file', lambda path, text: f"Error: append_file disabled in web sandbox: {path}")
        
        # Parse and execute with custom interpreter
        try:
            # Tokenize
            lexer = Lexer('<web>', code)
            tokens = lexer.make_tokens()
            
            # Parse
            parser = Parser(tokens)
            tree = parser.parse()
            
            # Execute with web interpreter
            interpreter = WebInterpreter()
            result = interpreter.run(tree)
            
            return jsonify({
                'output': '\n'.join(interpreter.captured_output),
                'result': str(result) if result is not None else None
            })
            
        except InputNeeded as e:
            # Return input request to frontend
            return jsonify({
                'input_needed': True,
                'prompt': e.message,
                'output': '\n'.join(interpreter.captured_output) if 'interpreter' in locals() else ''
            })
            
        except (LexerError, ParseError) as e:
            # Structured syntax/parse error with line/col/snippet
            try:
                # Extract positions
                if isinstance(e, LexerError):
                    ps, pe = e.pos_start, e.pos_end
                    err_type = 'lexer'
                else:
                    tok = e.token
                    ps, pe = tok.pos_start, tok.pos_end
                    err_type = 'parse'
                line = ps.ln + 1 if ps else None
                col = ps.col if ps else None
                from mc_lang import get_snippet
                snippet = get_snippet(ps.ftxt if ps else code, ps, pe) if ps and pe else None
                return jsonify({'error': str(e), 'error_detail': {
                    'type': err_type,
                    'line': line,
                    'col': col,
                    'snippet': snippet
                }})
            except Exception:
                return jsonify({'error': str(e)})
        except RTError as e:
            return jsonify({'error': f"Runtime Error: {e}", 'error_detail': {'type': 'runtime'}})
        except Exception as e:
            return jsonify({'error': f"Server Error: {e}", 'error_detail': {'type': 'server'}})
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/api/examples')
def get_examples():
    examples = [
        {
            'name': 'Hello World',
            'code': 'print("Hello, World!")\nlet x = 42\nprint("The answer is", x)'
        },
        {
            'name': 'Lists & Arrays',
            'code': '# Create a list\nlet nums = [1, 2, 3, 4, 5]\nprint("List:", nums)\nprint("First:", nums[0])\nprint("Length:", len(nums))\n\n# List methods\nnums.append(6)\nprint("After append:", nums)\n\n# List operations\nprint("Sum:", sum(nums))\nprint("Max:", max(nums))\nprint("Sorted:", sorted([5, 2, 8, 1]))'
        },
        {
            'name': 'For Loops',
            'code': 'print("=== For Loop Examples ===")\n\n# Loop through list\nfor x in [1, 2, 3, 4, 5] {\n    print("Number:", x)\n}\n\nprint("\\nUsing range:")\nfor i in range(5) {\n    print(i)\n}\n\nprint("\\nWith break & continue:")\nfor i in range(10) {\n    if i == 3 { continue }\n    if i == 7 { break }\n    print(i)\n}'
        },
        {
            'name': 'Lambda Functions',
            'code': '# Lambda basics\nlet square = lambda x: x * x\nprint("Square of 5:", square(5))\n\nlet add = lambda a, b: a + b\nprint("Add 3 + 7:", add(3, 7))\n\n# Lambda with map\nlet nums = [1, 2, 3, 4, 5]\nlet doubled = map(lambda x: x * 2, nums)\nprint("Doubled:", doubled)\n\n# Lambda with filter\nlet evens = filter(lambda x: x % 2 == 0, nums)\nprint("Even numbers:", evens)'
        },
        {
            'name': 'Dictionaries',
            'code': '# Dictionary examples\nlet user = { name: "Alice", age: 30 }\nprint("User name:", user["name"])\nprint("Keys:", user.keys())\nuser["city"] = "Paris"\nprint("Updated:", user)\nlet cfg = dict(theme="dark", version=1)\nprint("Config:", cfg)'
        },
        {
            'name': 'Import Module',
            'code': '# Assuming math_utils.mc exists with fun add(a,b){ return a+b }\nlet m = import "math_utils.mc"\nprint("Module keys:", m.keys())\nprint("Add via module:", m["add"](2,3))'
        },
        {
            'name': 'JSON & Time',
            'code': '# JSON & time built-ins\nlet obj = json_parse("{\\"x\\":10,\\"y\\":20}")\nprint("Obj:", obj)\nprint("Stringify:", json_stringify(obj))\nprint("Now:", now())\nprint("Timestamp:", timestamp())'
        },
        {
            'name': 'String Methods',
            'code': 'let text = "Hello World"\nprint("Original:", text)\nprint("Upper:", text.upper())\nprint("Lower:", text.lower())\nprint("Split:", text.split(" "))\nprint("Replace:", text.replace("World", "Mitra"))\n\nlet data = "  hello  "\nprint("Strip:", data.strip())\nprint("Starts with \'He\':", text.startswith("He"))'
        },
        {
            'name': 'Try-Catch',
            'code': 'print("=== Error Handling ===")\n\ntry {\n    let result = 10 / 0\n    print("This won\'t run")\n} catch (err) {\n    print("Caught error:", err)\n}\n\nprint("Program continues...")\n\ntry {\n    let safe = 10 / 2\n    print("Safe division:", safe)\n} catch (err) {\n    print("No error here")\n}\n\nprint("Done!")'
        },
        {
            'name': 'FizzBuzz',
            'code': 'print("=== FizzBuzz ===\")\nfor i in range(1, 21) {\n    if i % 15 == 0 {\n        print("FizzBuzz")\n    } elif i % 3 == 0 {\n        print("Fizz")\n    } elif i % 5 == 0 {\n        print("Buzz")\n    } else {\n        print(i)\n    }\n}'
        },
        {
            'name': 'Calculator',
            'code': 'fun add(a, b) { return a + b }\nfun multiply(a, b) { return a * b }\nfun divide(a, b) {\n    if b == 0 {\n        return "Error: Division by zero"\n    }\n    return a / b\n}\n\nlet x = 5\nlet y = 3\nprint("Sum:", add(x, y))\nprint("Product:", multiply(x, y))\nprint("Division:", divide(x, y))\nprint("Division by 0:", divide(x, 0))'
        },
        {
            'name': 'Interactive Input',
            'code': 'let name = input("What\'s your name? ")\nprint("Hello,", name, "!")\n\nlet age = int(input("How old are you? "))\nif age >= 18 {\n    print("You are an adult!")\n} else {\n    print("You are", 18 - age, "years away from being an adult")\n}'
        },
        {
            'name': 'Advanced: Nested Lists',
            'code': '# Matrix example\nlet matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nprint("Matrix:", matrix)\nprint("Row 0:", matrix[0])\nprint("Element [1][2]:", matrix[1][2])\n\n# Modify nested list\nmatrix[0][0] = 100\nprint("Modified:", matrix)\n\n# List of names\nlet names = ["Alice", "Bob", "Charlie"]\nfor name in names {\n    print("Hello,", name)\n}'
        }
    ]
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
