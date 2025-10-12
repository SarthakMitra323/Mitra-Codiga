# Interactive Demo Session - Number Guessing Game
# This shows what I'll type in the REPL step by step

# Step 1: Setup the game
let secret = 7; print("I'm thinking of a number between 1 and 10")

# Step 2: Get user input  
let guess = int(input("Enter your guess: "))

# Step 3: Check the guess with conditionals
if guess == secret { print("Correct! You won!") } elif guess < secret { print("Too low!") } else { print("Too high!") }

# Step 4: Define a function for multiple rounds
fun check_guess(g, s) { if g == s { return "Correct!" } elif g < s { return "Too low!" } else { return "Too high!" } }

# Step 5: Test the function
print(check_guess(5, 7))
print(check_guess(8, 7)) 
print(check_guess(7, 7))

# Step 6: Create a scoring system
let score = 100; let attempts = 0

# Step 7: Simulate penalty for wrong guesses
attempts = attempts + 1; score = score - 10; print("Attempts:", attempts, "Score:", score)
