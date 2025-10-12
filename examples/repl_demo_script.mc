# Here's what I'll demonstrate in the REPL:
# 1. Create variables
# 2. Define a function
# 3. Use control flow
# 4. Interactive calculations

# Sample REPL session transcript:

mitra > let name = "Alice"
Alice
mitra > print("Hello", name)
Hello Alice
mitra > fun greet(person) { return "Welcome, " + person + "!" }
<fun greet(person)>
mitra > greet(name)
Welcome, Alice!
mitra > let age = 25
25
mitra > if age >= 18 { print(name, "is an adult") } else { print(name, "is a minor") }
Alice is an adult
mitra > let i = 3; while i > 0 { print("Countdown:", i); i = i - 1 }
Countdown: 3
Countdown: 2
Countdown: 1
3
mitra > let scores = 0; scores = scores + 10; scores = scores * 2; print("Final score:", scores)
Final score: 20
