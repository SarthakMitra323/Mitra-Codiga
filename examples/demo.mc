# Demo program for Mitra Codiga language

# Variables and arithmetic
let x = 10;
let y = 20;
let z = x * (y + 3) - 2;
print("z =", z);

# Conditionals
if z > 100 {
    print("z is big");
} elif z == 100 {
    print("z is exactly 100");
} else {
    print("z is small");
}

# While loop
let i = 0;
while i < 5 {
    print("i:", i);
    i = i + 1;
}

# Functions
fun add(a, b) { return a + b; }
fun fact(n) {
    let result = 1;
    while n > 1 {
        result = result * n;
        n = n - 1;
    }
    return result;
}

print("add(2, 3) =", add(2, 3));
print("fact(5) =", fact(5));

# Strings and builtins
print("len(\"hello\"):", len("hello"));
print("str(123):", str(123));
