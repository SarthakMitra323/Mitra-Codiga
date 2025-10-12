# Simple Interactive Calculator Program
print("=== Mitra Codiga Calculator ===")

# Define calculator functions
fun add_nums(a, b) { return a + b }
fun multiply(a, b) { return a * b }
fun power_calc(base, exp) { return base ** exp }

# Test the functions
let x = 5
let y = 3

print("Numbers:", x, "and", y)
print("Addition:", add_nums(x, y))
print("Multiplication:", multiply(x, y)) 
print("Power:", power_calc(x, y))

# Interactive calculation with conditionals
let result = add_nums(x, y)
if result > 7 {
    print("Result is greater than 7!")
    result = result * 2
    print("Doubled result:", result)
}

# Loop example - countdown
print("Countdown from", result, ":")
while result > 0 {
    print(result)
    result = result - 1
}
print("Done!")