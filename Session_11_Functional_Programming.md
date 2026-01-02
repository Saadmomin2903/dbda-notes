# Session 11 – Functional Programming

**Topics Covered:** Functional Programming Concepts, Functional Interfaces, Predicate, Consumer, Supplier, Function, Lambda Expressions, Method References, Effect on Collections

---

## 1. What is Functional Programming?

**Functional Programming** is a programming paradigm that treats computation as evaluation of mathematical functions and avoids changing state and mutable data.

### Key Concepts
- **First-class functions:** Functions can be assigned to variables, passed as arguments, returned from other functions
- **Pure functions:** Same input always produces same output, no side effects
- **Immutability:** Data cannot be modified after creation
- **Higher-order functions:** Functions that take other functions as parameters or return functions

---

## 2. Functional Interface

### Definition
An interface with **exactly one abstract method** (SAM - Single Abstract Method).

```java
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);  // Single abstract method
}
```

### Rules for Functional Interface

| Allowed | Not Allowed |
|---------|-------------|
| Exactly one abstract method | Multiple abstract methods |
| Multiple default methods | N/A |
| Multiple static methods | N/A |
| Methods from Object class | Counted as abstract |

```java
@FunctionalInterface
interface MyInterface {
    // 1 abstract method - OK
    void doSomething();
    
    // Default methods - OK
    default void method1() { }
    default void method2() { }
    
    // Static methods - OK
    static void staticMethod() { }
    
    // Object methods don't count
    String toString();  // OK
    boolean equals(Object obj);  // OK
}
```

⚠️ **Common Mistake:**
```java
@FunctionalInterface
interface Invalid {
    void method1();
    void method2();  // ERROR: More than one abstract method
}
```

⭐ **Exam Fact:** @FunctionalInterface annotation is **optional** but recommended for compile-time checking.

---

## 3. Built-in Functional Interfaces (java.util.function)

### 3.1 Predicate\<T>

**Signature:** `boolean test(T t)`  
**Purpose:** Test a condition, return boolean

```java
import java.util.function.Predicate;

Predicate<Integer> isEven = n -> n % 2 == 0;
System.out.println(isEven.test(4));  // true
System.out.println(isEven.test(5));  // false

// Real-world example
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6);
List<Integer> evenNumbers = numbers.stream()
    .filter(n -> n % 2 == 0)  // Predicate
    .collect(Collectors.toList());
System.out.println(evenNumbers);  // [2, 4, 6]
```

### Predicate Methods

```java
Predicate<Integer> isEven = n -> n % 2 == 0;
Predicate<Integer> isPositive = n -> n > 0;

// and() - Both must be true
Predicate<Integer> isEvenAndPositive = isEven.and(isPositive);
System.out.println(isEvenAndPositive.test(4));   // true
System.out.println(isEvenAndPositive.test(-4));  // false

// or() - At least one must be true
Predicate<Integer> isEvenOrNegative = isEven.or(n -> n < 0);
System.out.println(isEvenOrNegative.test(3));   // false
System.out.println(isEvenOrNegative.test(-3));  // true

// negate() - Logical NOT
Predicate<Integer> isOdd = isEven.negate();
System.out.println(isOdd.test(3));  // true
```

---

### 3.2 Consumer\<T>

**Signature:** `void accept(T t)`  
**Purpose:** Accept input, perform action, return nothing

```java
import java.util.function.Consumer;

Consumer<String> print = s -> System.out.println(s);
print.accept("Hello World");  // Prints: Hello World

// Real-world example
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
names.forEach(name -> System.out.println("Hello, " + name));
// OR
names.forEach(System.out::println);
```

### Consumer Methods

```java
Consumer<String> print = s -> System.out.println(s);
Consumer<String> printUpperCase = s -> System.out.println(s.toUpperCase());

// andThen() - Chain consumers
Consumer<String> combined = print.andThen(printUpperCase);
combined.accept("java");
// Output:
// java
// JAVA
```

---

### 3.3 Supplier\<T>

**Signature:** `T get()`  
**Purpose:** Supply/provide a value, take nothing

```java
import java.util.function.Supplier;

Supplier<Double> randomSupplier = () -> Math.random();
System.out.println(randomSupplier.get());  // Random number

Supplier<String> greeting = () -> "Hello, World!";
System.out.println(greeting.get());  // Hello, World!

// Real-world example - Lazy initialization
Supplier<ExpensiveObject> lazyInit = () -> new ExpensiveObject();
// Object not created yet
ExpensiveObject obj = lazyInit.get();  // Created only when needed
```

---

### 3.4 Function\<T, R>

**Signature:** `R apply(T t)`  
**Purpose:** Transform input type T to output type R

```java
import java.util.function.Function;

Function<Integer, String> converter = n -> "Number: " + n;
System.out.println(converter.apply(10));  // "Number: 10"

Function<String, Integer> length = s -> s.length();
System.out.println(length.apply("Hello"));  // 5

// Real-world example
List<String> words = Arrays.asList("Java", "Python", "C++");
List<Integer> lengths = words.stream()
    .map(s -> s.length())  // Function<String, Integer>
    .collect(Collectors.toList());
System.out.println(lengths);  // [4, 6, 3]
```

### Function Methods

```java
Function<Integer, Integer> square = n -> n * n;
Function<Integer, Integer> addTen = n -> n + 10;
Function<Integer, String> toString = n -> n.toString();

// andThen() - Execute this, then that
Function<Integer, Integer> squareThenAdd = square.andThen(addTen);
System.out.println(squareThenAdd.apply(5));  // (5*5) + 10 = 35

// compose() - Execute that, then this
Function<Integer, Integer> addThenSquare = square.compose(addTen);
System.out.println(addThenSquare.apply(5));  // (5+10)*(5+10) = 225

// Chaining multiple
Function<Integer, String> pipeline = square.andThen(addTen).andThen(toString);
System.out.println(pipeline.apply(5));  // "35"
```

---

### 3.5 UnaryOperator\<T>

**Signature:** `T apply(T t)`  
**Purpose:** Special case of Function where input and output types are same

```java
import java.util.function.UnaryOperator;

UnaryOperator<Integer> square = n -> n * n;
System.out.println(square.apply(5));  // 25

UnaryOperator<String> toUpperCase = s -> s.toUpperCase();
System.out.println(toUpperCase.apply("java"));  // JAVA
```

---

### 3.6 BinaryOperator\<T>

**Signature:** `T apply(T t1, T t2)`  
**Purpose:** Takes two arguments of same type, returns same type

```java
import java.util.function.BinaryOperator;

BinaryOperator<Integer> add = (a, b) -> a + b;
System.out.println(add.apply(5, 3));  // 8

BinaryOperator<Integer> max = (a, b) -> a > b ? a : b;
System.out.println(max.apply(10, 20));  // 20

BinaryOperator<String> concat = (s1, s2) -> s1 + s2;
System.out.println(concat.apply("Hello, ", "World!"));  // Hello, World!

// Real-world example
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int sum = numbers.stream()
    .reduce(0, (a, b) -> a + b);  // BinaryOperator
System.out.println(sum);  // 15
```

---

### Summary Table: Built-in Functional Interfaces

| Interface | Method | Parameters | Return | Use Case |
|-----------|--------|------------|--------|----------|
| `Predicate<T>` | `test(T)` | 1 | boolean | Filtering, conditions |
| `Consumer<T>` | `accept(T)` | 1 | void | Side effects, printing |
| `Supplier<T>` | `get()` | 0 | T | Factory, lazy init |
| `Function<T,R>` | `apply(T)` | 1 | R | Transformation |
| `UnaryOperator<T>` | `apply(T)` | 1 | T | Same type transform |
| `BinaryOperator<T>` | `apply(T,T)` | 2 | T | Accumulation, reduction |

---

## 4. Lambda Expressions

### What is Lambda?
**Anonymous function** - function without name, implemented inline.

### Syntax

```java
// Full syntax
(parameter1, parameter2) -> { statements; return value; }

// Variations
() -> expression                    // No parameters
(x) -> expression                   // One parameter (parentheses optional)
x -> expression                     // One parameter (simplified)
(x, y) -> expression                // Multiple parameters
(x, y) -> { statements; }           // Block body
(int x, int y) -> x + y             // With type (usually inferred)
```

### Examples

```java
// No parameters
Runnable r = () -> System.out.println("Running");

// One parameter (multiple forms)
Consumer<String> print1 = (s) -> System.out.println(s);
Consumer<String> print2 = s -> System.out.println(s);  // Parentheses optional

// Multiple parameters
BinaryOperator<Integer> add = (a, b) -> a + b;

// Block body
Consumer<Integer> printSquare = n -> {
    int square = n * n;
    System.out.println("Square: " + square);
};

// With type annotations
BinaryOperator<Integer> max = (Integer a, Integer b) -> a > b ? a : b;
```

### Before vs After Lambda

```java
// Before Java 8 - Anonymous Inner Class
Runnable r1 = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running");
    }
};

// After Java 8 - Lambda
Runnable r2 = () -> System.out.println("Running");

// Comparator example
// Before
Collections.sort(list, new Comparator<String>() {
    public int compare(String s1, String s2) {
        return s1.compareTo(s2);
    }
});

// After
Collections.sort(list, (s1, s2) -> s1.compareTo(s2));
```

---

## 5. Variable Capture in Lambda

### Effectively Final

Lambda can access local variables from enclosing scope, but they must be **effectively final** (not modified after initialization).

```java
int x = 10;
Consumer<Integer> consumer = n -> System.out.println(n + x);
// x = 20;  // ERROR: x is not effectively final

consumer.accept(5);  // Prints: 15
```

### Why Effectively Final?

```java
void method() {
    int counter = 0;
    
    Runnable r = () -> {
        // counter++;  // ERROR: Cannot modify effectively final variable
        System.out.println(counter);  // OK: Can read
    };
}
```

⭐ **Exam Fact:** Lambda can **read** effectively final variables, but cannot **modify** them.

### Instance Variables (Not Restricted)

```java
class MyClass {
    private int counter = 0;
    
    public void method() {
        Runnable r = () -> {
            counter++;  // OK: Instance variable can be modified
            System.out.println(counter);
        };
        r.run();
    }
}
```

---

## 6. Method References

**Method Reference** is shorthand for lambda that calls a single method.

### Syntax: `::`

### Types of Method References

#### 1. Static Method Reference

```java
// Lambda
Function<String, Integer> parse1 = s -> Integer.parseInt(s);

// Method reference
Function<String, Integer> parse2 = Integer::parseInt;

// Usage
System.out.println(parse2.apply("123"));  // 123
```

#### 2. Instance Method Reference (Specific Instance)

```java
String str = "Hello";

// Lambda
Supplier<Integer> length1 = () -> str.length();

// Method reference
Supplier<Integer> length2 = str::length;

// Usage
System.out.println(length2.get());  // 5
```

#### 3. Instance Method Reference (Arbitrary Instance)

```java
// Lambda
Function<String, Integer> length1 = s -> s.length();

// Method reference
Function<String, Integer> length2 = String::length;

// Usage
System.out.println(length2.apply("World"));  // 5

// Real example
List<String> words = Arrays.asList("Java", "Python", "C++");
words.stream()
    .map(String::toUpperCase)  // Instance method reference
    .forEach(System.out::println);
```

#### 4. Constructor Reference

```java
// Lambda
Supplier<ArrayList<String>> supplier1 = () -> new ArrayList<>();

// Constructor reference
Supplier<ArrayList<String>> supplier2 = ArrayList::new;

// Usage
ArrayList<String> list = supplier2.get();

// With parameters
Function<Integer, ArrayList<String>> createList = ArrayList::new;
ArrayList<String> list2 = createList.apply(10);  // Initial capacity 10
```

### Method Reference Examples

```java
// Static method
Arrays.asList("1", "2", "3").stream()
    .map(Integer::parseInt)
    .forEach(System.out::println);

// Instance method (specific)
String prefix = "Number: ";
Consumer<Integer> print = prefix::concat;  // Doesn't work directly
// Better:
BiConsumer<String, Integer> concat = String::concat;

// Instance method (arbitrary)
List<String> names = Arrays.asList("alice", "bob", "charlie");
names.stream()
    .map(String::toUpperCase)
    .forEach(System.out::println);

// Constructor
Stream.of("Java", "Python")
    .map(String::new)  // Creates copy
    .forEach(System.out::println);
```

---

## 7. Effect on Collections

### forEach()

```java
List<String> list = Arrays.asList("A", "B", "C");

// Traditional for loop
for (String s : list) {
    System.out.println(s);
}

// Using forEach with lambda
list.forEach(s -> System.out.println(s));

// Using method reference
list.forEach(System.out::println);
```

### removeIf()

```java
List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 6));

// Remove even numbers
numbers.removeIf(n -> n % 2 == 0);
System.out.println(numbers);  // [1, 3, 5]
```

### replaceAll()

```java
List<String> words = new ArrayList<>(Arrays.asList("java", "python", "c++"));

// Convert to uppercase
words.replaceAll(s -> s.toUpperCase());
// OR
words.replaceAll(String::toUpperCase);

System.out.println(words);  // [JAVA, PYTHON, C++]
```

### sort()

```java
List<String> names = Arrays.asList("Charlie", "Alice", "Bob");

// Sort with lambda
names.sort((s1, s2) -> s1.compareTo(s2));

// Sort with method reference
names.sort(String::compareTo);

System.out.println(names);  // [Alice, Bob, Charlie]
```

---

## 🔥 Top MCQs for Session 11

### MCQ 1: Functional Interface
**Q:** How many abstract methods in functional interface?
```java
@FunctionalInterface
interface Test {
    void method1();
    default void method2() { }
    static void method3() { }
}
```
1. 0
2. 1
3. 2
4. 3

**Answer:** 2. 1  
**Explanation:** Only abstract methods count. Default and static don't count.

---

### MCQ 2: Predicate
**Q:** Which functional interface returns boolean?
1. Consumer
2. Supplier
3. Predicate
4. Function

**Answer:** 3. Predicate  
**Explanation:** Predicate.test() returns boolean.

---

### MCQ 3: Supplier
**Q:** Supplier<T> method signature?
1. T accept(T t)
2. T get()
3. void accept(T t)
4. boolean test(T t)

**Answer:** 2. T get()  
**Explanation:** Supplier takes nothing, returns T.

---

### MCQ 4: Consumer
**Q:** Consumer<T> returns?
1. T
2. boolean
3. void
4. Object

**Answer:** 3. void  
**Explanation:** Consumer.accept() returns void.

---

### MCQ 5: Effectively Final
**Q:** What happens?
```java
int x = 10;
Consumer<Integer> c = n -> System.out.println(n + x);
x = 20;
```
1. Compiles fine
2. Compile error
3. Runtime error
4. Prints 30

**Answer:** 2. Compile error  
**Explanation:** x is not effectively final (modified after lambda).

---

### MCQ 6: Method Reference
**Q:** Which is valid static method reference?
1. String::length
2. Integer::parseInt
3. System.out::println
4. new ArrayList<>()

**Answer:** 2. Integer::parseInt  
**Explanation:** parseInt is static method of Integer class.

---

### MCQ 7: Lambda Syntax
**Q:** Which lambda is INVALID?
1. () -> 5
2. x -> x * 2
3. (x, y) -> x + y
4. x, y -> x + y

**Answer:** 4. x, y -> x + y  
**Explanation:** Multiple parameters need parentheses: (x, y) -> x + y

---

### MCQ 8: BinaryOperator
**Q:** BinaryOperator<T> takes:
1. No parameters
2. One parameter
3. Two parameters of same type
4. Two parameters of different types

**Answer:** 3. Two parameters of same type  
**Explanation:** BinaryOperator<T> signature: T apply(T t1, T t2)

---

### MCQ 9: Function chaining
**Q:** What is output?
```java
Function<Integer, Integer> f1 = x -> x * 2;
Function<Integer, Integer> f2 = x -> x + 10;
Function<Integer, Integer> f3 = f1.andThen(f2);
System.out.println(f3.apply(5));
```
1. 20
2. 30
3. 15
4. 25

**Answer:** 2. 20  
**Explanation:** f1(5) = 10, then f2(10) = 20. andThen executes f1 then f2.

---

### MCQ 10: Constructor Reference
**Q:** Which is constructor reference?
1. ArrayList::new
2. ArrayList::size
3. new ArrayList()
4. ArrayList.new

**Answer:** 1. ArrayList::new  
**Explanation:** ClassName::new is constructor reference syntax.

---

## ⚠️ Common Mistakes

1. **Multiple abstract methods** in @FunctionalInterface
2. **Modifying effectively final** variables in lambda
3. **Confusing andThen vs compose** in Function
4. **Wrong method reference syntax** (using . instead of ::)
5. **Forgetting parentheses** for multiple lambda parameters
6. **Mixing return types** in lambda
7. **Trying to access non-final** local variables

---

## ⭐ One-liner Exam Facts

1. Functional interface has **exactly one abstract method**
2. @FunctionalInterface is **optional** but recommended
3. **Default and static** methods don't count in functional interface
4. Predicate → **boolean**, Consumer → **void**, Supplier → **T**
5. Function<T, R> → **T in, R out**
6. UnaryOperator<T> → **T in, T out**
7. BinaryOperator<T> → **T, T in, T out**
8. Lambda can access **effectively final** variables only
9. **Instance variables** can be modified in lambda
10. Method reference syntax: **::** (double colon)
11. **Static method** reference: ClassName::methodName
12. **Instance method** reference: instance::methodName or ClassName::methodName
13. **Constructor** reference: ClassName::new
14. andThen() → **execute this then that**
15. compose() → **execute that then this**

---

**End of Session 11**
