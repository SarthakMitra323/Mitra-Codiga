# Interactive Input Demo for Mitra Codiga

print("=== Interactive Input Demo ===")

# Basic input
let name = input("What's your name? ")
print("Hello,", name, "!")

# Input with type conversion
let age_str = input("How old are you? ")
let age = int(age_str)
print("You are", age, "years old")

# Conditional logic with input
if age >= 18 {
    print("You are an adult!")
} else {
    print("You are a minor")
}

# Mathematical input
let num1_str = input("Enter first number: ")
let num2_str = input("Enter second number: ")
let num1 = float(num1_str)
let num2 = float(num2_str)

print("You entered:", num1, "and", num2)
print("Sum:", num1 + num2)
print("Product:", num1 * num2)

# Interactive calculator loop
print("\n=== Simple Calculator ===")
print("Enter 'quit' to exit")

let continue_calc = true
while continue_calc {
    let operation = input("Enter operation (+, -, *, /, **) or 'quit': ")
    
    if operation == "quit" {
        continue_calc = false
        print("Goodbye!")
    } else {
        let a = float(input("Enter first number: "))
        let b = float(input("Enter second number: "))
        
        if operation == "+" {
            print("Result:", a + b)
        } elif operation == "-" {
            print("Result:", a - b)
        } elif operation == "*" {
            print("Result:", a * b)
        } elif operation == "/" {
            print("Result:", a / b)
        } elif operation == "**" {
            print("Result:", a ** b)
        } else {
            print("Invalid operation!")
        }
    }
}
