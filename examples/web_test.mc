# Quick test for the output capture fix
# This code should display output in the web UI, not in terminal

print("Testing output capture!")
let name = "Mitra Codiga"
print("Language name:", name)

# Test math
let x = 10
let y = 5
print("x + y =", x + y)
print("x * y =", x * y)

# Test function
fun greet(person) {
    return "Hello, " + person + "!"
}

print(greet(name))

# Test loop
let i = 1
while i <= 3 {
    print("Count:", i)
    i = i + 1
}

print("All tests complete!")
