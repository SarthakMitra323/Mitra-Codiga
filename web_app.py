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
                syms.set('true', True)
                syms.set('false', False)
                syms.set('null', None)
        
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
            
        except Exception as e:
            # Handle any parsing or runtime errors
            error_msg = str(e)
            return jsonify({'error': error_msg})
            
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
            'name': 'Interactive Input',
            'code': 'let name = input("What\'s your name? ")\nprint("Hello,", name, "!")\n\nlet age = int(input("How old are you? "))\nif age >= 18 {\n    print("You are an adult!")\n} else {\n    print("You are", 18 - age, "years away from being an adult")\n}'
        },
        {
            'name': 'Calculator',
            'code': 'fun add(a, b) { return a + b }\nfun multiply(a, b) { return a * b }\n\nlet x = 5\nlet y = 3\nprint("Sum:", add(x, y))\nprint("Product:", multiply(x, y))'
        },
        {
            'name': 'Fibonacci',
            'code': 'fun fib(n) {\n    if n <= 1 { return n }\n    let a = 0\n    let b = 1\n    let i = 2\n    while i <= n {\n        let temp = a + b\n        a = b\n        b = temp\n        i = i + 1\n    }\n    return b\n}\n\nprint("Fibonacci sequence:")\nlet i = 0\nwhile i < 10 {\n    print("fib(" + str(i) + ") =", fib(i))\n    i = i + 1\n}'
        },
        {
            'name': 'Control Flow',
            'code': 'let score = 85\n\nif score >= 90 {\n    print("Grade: A")\n} elif score >= 80 {\n    print("Grade: B")\n} elif score >= 70 {\n    print("Grade: C")\n} else {\n    print("Grade: F")\n}\n\nlet countdown = 5\nprint("\\nCountdown:")\nwhile countdown > 0 {\n    print(countdown)\n    countdown = countdown - 1\n}\nprint("Blast off!")'
        },
        {
            'name': 'Interactive Quiz',
            'code': 'print("=== Math Quiz ===")\nlet score = 0\n\nlet answer1 = int(input("What is 5 + 3? "))\nif answer1 == 8 {\n    print("Correct!")\n    score = score + 1\n} else {\n    print("Wrong! The answer is 8")\n}\n\nlet answer2 = int(input("What is 7 * 6? "))\nif answer2 == 42 {\n    print("Correct!")\n    score = score + 1\n} else {\n    print("Wrong! The answer is 42")\n}\n\nprint("Your score:", score, "out of 2")\nif score == 2 {\n    print("Perfect score! 🎉")\n} elif score == 1 {\n    print("Good job! 👍")\n} else {\n    print("Keep practicing! 📚")\n}'
        }
    ]
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)