# Mitra Codiga Language Interpreter

An advanced, educational programming language interpreter built in Python. Features variables, functions, control flow, and more!

## 🚀 Try it Online

**Live Demo:** [Deploy to your preferred platform using the instructions below]

## ✨ Features

- **Variables & Types**: `let x = 10`, strings, numbers, booleans
- **Functions**: `fun name(args) { return value }`  
- **Control Flow**: `if/elif/else`, `while` loops
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `**`), comparisons, boolean logic
- **Built-ins**: `print()`, `input()`, `len()`, `int()`, `float()`, `str()`
- **Interactive REPL**: Test code interactively
- **Web Interface**: Online code editor and executor

## 🏃‍♂️ Quick Start

### Local Installation

```bash
git clone <your-repo-url>
cd mitra-codiga
```

### Run Programs

```bash
# Run a .mc file
python run_mc.py demo.mc

# Start interactive REPL
python mc_lang.py
```

### Web Interface

```bash
pip install -r requirements.txt
python web_app.py
# Visit http://localhost:5000
```

## 📝 Example Code

```javascript
# Variables and arithmetic
let x = 10;
let y = 20;
print("Sum:", x + y);

# Functions  
fun fibonacci(n) {
    if n <= 1 { return n; }
    return fibonacci(n-1) + fibonacci(n-2);
}

print("Fib(10):", fibonacci(10));

# Control flow
let i = 0;
while i < 5 {
    print("Count:", i);
    i = i + 1;
}
```

## 🌐 Free Hosting Options

### 1. **Repl.it** (Easiest)
- Fork this repository
- Import to Repl.it
- Runs immediately with web interface

### 2. **Heroku** 
```bash
# Add these files (already included):
# - Procfile
# - requirements.txt
# - runtime.txt

heroku create your-app-name
git push heroku main
```

### 3. **Railway**
- Connect your GitHub repo
- Automatic deployment from `main` branch
- Zero configuration needed

### 4. **Render**
- Connect GitHub repository  
- Select "Web Service"
- Build: `pip install -r requirements.txt`
- Start: `python web_app.py`

### 5. **Vercel** (Static + Serverless)
- Install Vercel CLI: `npm i -g vercel`
- Run: `vercel --prod`
- Automatic deployments from GitHub

### 6. **Netlify**
- Drag & drop the folder to Netlify
- Or connect via GitHub for auto-deployment

## 📂 Project Structure

```
mitra-codiga/
├── mc_lang.py          # Core interpreter implementation
├── run_mc.py           # CLI runner for .mc files  
├── web_app.py          # Flask web interface
├── demo.mc             # Example program
├── templates/
│   └── index.html      # Web interface
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku deployment
├── runtime.txt        # Python version spec
├── vercel.json        # Vercel config
├── netlify.toml       # Netlify config
└── README.md          # This file
```

## 🔧 Development

### Architecture

1. **Lexer**: Converts source code to tokens
2. **Parser**: Builds Abstract Syntax Tree (AST)  
3. **Interpreter**: Executes the AST with symbol tables

### Adding Features

- **New operators**: Modify lexer tokens and interpreter logic
- **Built-in functions**: Add to `inject_builtins()` 
- **Language constructs**: Add AST nodes and parser rules

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - feel free to use this for educational purposes!

## 🙏 Acknowledgments

Built as an educational project to demonstrate:
- Programming language design
- Interpreter implementation  
- Web deployment strategies
- Modern development practices

---

**Made with ❤️ and AI assistance**
