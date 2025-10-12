# Simple Input Demo

print("=== Mitra Codiga Input Demo ===")

# Get user's name
let name = input("Enter your name: ")
print("Nice to meet you,", name)

# Get a number and do math
let num_str = input("Enter a number: ")
let number = int(num_str)
print("Your number squared is:", number * number)

# Simple yes/no question
let answer = input("Do you like programming? (yes/no): ")
if answer == "yes" {
    print("That's awesome!")
} else {
    print("Maybe you'll enjoy Mitra Codiga!")
}

print("Thanks for trying the demo!")
