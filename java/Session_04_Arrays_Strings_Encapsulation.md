# Session 4 – Arrays, Strings & Encapsulation

**Topics Covered:** Packages & Classpath, Arrays (1D, 2D), String vs StringBuilder vs StringBuffer, Immutability, Methods & Encapsulation, Access Modifiers, Method Overloading, Constructors, Passing Data

---

## 1. Packages & Classpath

### What is Classpath?
Classpath tells JVM where to find compiled .class files.

```bash
# Set classpath (Linux/Mac)
export CLASSPATH=/path/to/classes:/path/to/lib/mylib.jar

# Set classpath (Windows)
set CLASSPATH=C:\classes;C:\lib\mylib.jar

# Compile with classpath
javac -cp lib/commons.jar MyApp.java

# Run with classpath
java -cp .:lib/commons.jar MyApp
```

⭐ **Exam Fact:** `.` (dot) represents current directory in classpath.

---

## 2. Arrays

### 1D Arrays

```java
// Declaration
int[] arr1;          // Preferred
int arr2[];          // Valid but not preferred

// Initialization
int[] arr = new int[5];              // All elements = 0
int[] arr = {1, 2, 3, 4, 5};         // Array literal
int[] arr = new int[]{1, 2, 3};      // Anonymous array

// Access
arr[0] = 10;
int first = arr[0];

// Length
System.out.println(arr.length);  // 5 (property, not method!)
```

⚠️ **Common Mistakes:**
```java
int[] arr = new int[5];
System.out.println(arr.length());  // ERROR: length is property, not method
System.out.println(arr[5]);         // ArrayIndexOutOfBoundsException
```

### 2D Arrays

```java
// Declaration & Initialization
int[][] matrix = new int[3][4];      // 3 rows, 4 columns
int[][] matrix = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

// Jagged Array (different column sizes)
int[][] jagged = new int[3][];
jagged[0] = new int[2];
jagged[1] = new int[4];
jagged[2] = new int[3];

// Access
matrix[0][0] = 10;
int val = matrix[1][2];

// Length
System.out.println(matrix.length);      // 3 (rows)
System.out.println(matrix[0].length);   // 4 (columns in row 0)
```

### Array Default Values

| Type | Default Value |
|------|---------------|
| int[] | 0 |
| double[] | 0.0 |
| boolean[] | false |
| String[] | null |

---

## 3. String vs StringBuilder vs StringBuffer

### String (Immutable)

```java
String s1 = "Hello";
String s2 = "Hello";
String s3 = new String("Hello");

System.out.println(s1 == s2);        // true (string pool)
System.out.println(s1 == s3);        // false (different objects)
System.out.println(s1.equals(s3));   // true (value comparison)
```

### String Pool (Intern Pool)

```
Heap Memory
┌────────────────────────────────────┐
│  String Pool                       │
│  ┌──────────────┐                  │
│  │ "Hello"      │ ←───┐            │
│  └──────────────┘     │            │
│                       │            │
│  Regular Objects      │            │
│  ┌──────────────┐     │            │
│  │ String obj   │     │            │
│  │ value="Hello"│     │            │
│  └──────────────┘     │            │
└───────────────────────┼────────────┘
                        │
Stack                   │
┌──────┐                │
│ s1   │ ───────────────┘
│ s2   │ ───────────────┘
│ s3   │ ────────────→ Regular Object
└──────┘
```

### String Immutability

```java
String s = "Hello";
s = s + " World";  // Creates NEW string, old "Hello" eligible for GC

// Inefficient (creates many objects)
String result = "";
for (int i = 0; i < 1000; i++) {
    result += i;  // Creates 1000 String objects!
}
```

⭐ **Exam Fact:** Every string concatenation with `+` creates a **new String object**.

### StringBuilder (Mutable, Not Thread-Safe)

```java
StringBuilder sb = new StringBuilder("Hello");
sb.append(" World");        // Modifies same object
sb.insert(5, ",");          // Hello, World
sb.replace(0, 5, "Hi");     // Hi, World
sb.delete(2, 4);            // Hi World
sb.reverse();               // dlroW iH

String result = sb.toString();

// Performance
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.append(i);  // Efficient (1 object)
}
```

### StringBuffer (Mutable, Thread-Safe)

```java
StringBuffer sbf = new StringBuffer("Hello");
sbf.append(" World");  // Synchronized methods
```

### Comparison Table

| Feature | String | StringBuilder | StringBuffer |
|---------|--------|---------------|--------------|
| **Mutability** | Immutable | Mutable | Mutable |
| **Thread Safety** | Thread-safe | Not thread-safe | Thread-safe |
| **Performance** | Slow (creates objects) | Fast | Slower than StringBuilder |
| **Use Case** | Few modifications | Single-threaded, many modifications | Multi-threaded, many modifications |
| **Memory** | More (creates new objects) | Less | Less |

⭐ **Exam Facts:**
1. Use **String** for fixed text
2. Use **StringBuilder** for single-threaded concatenation
3. Use **StringBuffer** for multi-threaded concatenation
4. String pool exists only for **String**, not StringBuilder/StringBuffer

---

## 4. Important String Methods

```java
String s = "Hello World";

// Length
s.length();                    // 11

// Character access
s.charAt(0);                   // 'H'
s.indexOf('o');                // 4 (first occurrence)
s.lastIndexOf('o');            // 7

// Substring
s.substring(0, 5);             // "Hello"
s.substring(6);                // "World"

// Comparison
s.equals("Hello World");       // true
s.equalsIgnoreCase("hello world");  // true
s.compareTo("Hello");          // positive (lexicographically greater)

// Search
s.contains("World");           // true
s.startsWith("Hello");         // true
s.endsWith("World");           // true

// Modification (returns NEW string)
s.toUpperCase();               // "HELLO WORLD"
s.toLowerCase();               // "hello world"
s.trim();                      // Removes leading/trailing spaces
s.replace("World", "Java");    // "Hello Java"

// Split
String[] words = s.split(" "); // ["Hello", "World"]

// Empty check
s.isEmpty();                   // false
s.isBlank();                   // false (Java 11+, checks whitespace)
```

⚠️ **Common MCQ Trap:**
```java
String s = "Hello";
s.toUpperCase();
System.out.println(s);  // "Hello" (NOT "HELLO", String is immutable!)

// Correct way
s = s.toUpperCase();
System.out.println(s);  // "HELLO"
```

---

## 5. Methods & Encapsulation

### Method Syntax

```java
<access_modifier> <return_type> methodName(<parameters>) {
    // method body
    return value;  // if return type is not void
}
```

### Example

```java
public class Calculator {
    // Method with return value
    public int add(int a, int b) {
        return a + b;
    }
    
    // Method without return value
    public void display() {
        System.out.println("Calculator");
    }
    
    // Method with varargs
    public int sum(int... numbers) {
        int total = 0;
        for (int num : numbers) {
            total += num;
        }
        return total;
    }
}

// Usage
Calculator calc = new Calculator();
int result = calc.add(5, 3);
calc.display();
int s = calc.sum(1, 2, 3, 4, 5);  // Varargs
```

---

## 6. Access Modifiers

### Access Levels

| Modifier | Class | Package | Subclass | World |
|----------|-------|---------|----------|-------|
| **public** | ✅ | ✅ | ✅ | ✅ |
| **protected** | ✅ | ✅ | ✅ | ❌ |
| **default** (package-private) | ✅ | ✅ | ❌ | ❌ |
| **private** | ✅ | ❌ | ❌ | ❌ |

### Examples

```java
public class AccessDemo {
    public int a = 10;        // Accessible everywhere
    protected int b = 20;     // Accessible in package + subclasses
    int c = 30;               // Accessible in package only (default)
    private int d = 40;       // Accessible in class only
    
    private void privateMethod() { }
    void defaultMethod() { }
    protected void protectedMethod() { }
    public void publicMethod() { }
}
```

⭐ **Exam Facts:**
1. **Class** can be `public` or default (not private/protected)
2. **Top-level class** cannot be private or protected
3. **Inner class** can have any access modifier

---

## 7. Method Overloading

### Definition
Multiple methods with **same name** but **different parameters** in the same class.

```java
public class MathUtils {
    // Overloaded methods
    public int add(int a, int b) {
        return a + b;
    }
    
    public double add(double a, double b) {
        return a + b;
    }
    
    public int add(int a, int b, int c) {
        return a + b + c;
    }
    
    public String add(String a, String b) {
        return a + b;
    }
}
```

### Overloading Rules

✅ **Valid Overloading:**
- Different number of parameters
- Different types of parameters
- Different order of parameters

❌ **Invalid Overloading:**
- Only return type different
- Only access modifier different

```java
// VALID
public int add(int a, int b) { }
public int add(double a, double b) { }

// INVALID (compile error)
public int add(int a, int b) { }
public double add(int a, int b) { }  // Only return type different!
```

### Overloading with Varargs

```java
public void print(int a) { }
public void print(int... a) { }  // VALID but ambiguous

// Calling
print(5);  // Which method? Compile error!
```

⚠️ **Common MCQ Trap: Overloading Resolution**
```java
public void method(int a) { System.out.println("int"); }
public void method(Integer a) { System.out.println("Integer"); }

method(5);     // Prints "int" (primitive preferred)
method(null);  // Prints "Integer" (null can't be primitive)
```

---

## 8. Constructors

### Definition
Special method to initialize objects, same name as class, no return type.

```java
public class Employee {
    String name;
    int id;
    
    // Default constructor (provided by compiler if not defined)
    public Employee() {
        name = "Unknown";
        id = 0;
    }
    
    // Parameterized constructor
    public Employee(String name, int id) {
        this.name = name;
        this.id = id;
    }
    
    // Constructor overloading
    public Employee(String name) {
        this(name, 0);  // Calls parameterized constructor
    }
}
```

### Constructor Rules

1. **Name** must match class name
2. **No return type** (not even void)
3. **Can be overloaded**
4. **this()** must be first statement in constructor
5. If no constructor defined, compiler provides **default constructor**
6. If any constructor defined, compiler **does NOT** provide default

⚠️ **Common Mistake:**
```java
public class Test {
    public Test(int a) { }
}

Test t = new Test();  // ERROR: default constructor not available!
```

### Constructor Chaining

```java
public class Employee {
    String name;
    int id;
    
    public Employee() {
        this("Unknown", 0);
    }
    
    public Employee(String name) {
        this(name, 0);
    }
    
    public Employee(String name, int id) {
        this.name = name;
        this.id = id;
    }
}
```

---

## 9. Passing Data (Value vs Reference)

### Key Concept
Java is **always pass-by-value**, but for objects, the **value is the reference**.

### Passing Primitives

```java
public void modify(int x) {
    x = 20;  // Only changes local copy
}

int a = 10;
modify(a);
System.out.println(a);  // 10 (unchanged)
```

### Passing Objects

```java
class Box {
    int value;
}

public void modify(Box b) {
    b.value = 20;  // Modifies original object
}

Box box = new Box();
box.value = 10;
modify(box);
System.out.println(box.value);  //20 (changed!)
```

### Reassigning Reference

```java
public void modify(Box b) {
    b = new Box();  // Creates new object, doesn't affect original
    b.value = 30;
}

Box box = new Box();
box.value = 10;
modify(box);
System.out.println(box.value);  // 10 (unchanged!)
```

**Memory Diagram:**
```
Before modify():
┌─────┐        ┌──────────┐
│ box │───────→│ value=10 │
└─────┘        └──────────┘

Inside modify() before reassignment:
┌─────┐        ┌──────────┐
│ box │───────→│ value=10 │
└─────┘        └──────────┘
┌─────┐           ↑
│  b  │───────────┘
└─────┘

Inside modify() after b = new Box():
┌─────┐        ┌──────────┐
│ box │───────→│ value=10 │ (unchanged)
└─────┘        └──────────┘
┌─────┐        ┌──────────┐
│  b  │───────→│ value=30 │ (new object)
└─────┘        └──────────┘
```

⭐ **Exam Fact:** In Java, you **cannot change** where the original reference points, but you **can modify** the object it points to.

---

## 🔥 Top MCQs for Session 4

### MCQ 1: Array Length
**Q:** What is the output?
```java
int[] arr = {1, 2, 3};
System.out.println(arr.length());
```
1. 3
2. 2
3. Compile error
4. Runtime error

**Answer:** 3. Compile error  
**Explanation:** `length` is a property, not a method. Should be `arr.length`.

---

### MCQ 2: String Pool
**Q:** What is the output?
```java
String s1 = "Hello";
String s2 = "Hello";
String s3 = new String("Hello");
System.out.println((s1 == s2) + " " + (s1 == s3));
```
1. true true
2. true false
3. false true
4. false false

**Answer:** 2. true false  
**Explanation:** s1 and s2 refer to same pool object. s3 is a separate object.

---

### MCQ 3: String Immutability
**Q:** What is the output?
```java
String s = "Java";
s.concat(" Programming");
System.out.println(s);
```
1. Java Programming
2. Java
3. Programming
4. Compile error

**Answer:** 2. Java  
**Explanation:** String is immutable. concat() returns new string, doesn't modify original.

---

### MCQ 4: StringBuilder vs StringBuffer
**Q:** Which is thread-safe?
1. String
2. StringBuilder
3. StringBuffer
4. All of the above

**Answer:** 3. StringBuffer  
**Explanation:** StringBuffer methods are synchronized. String is immutable (different concept). StringBuilder is not thread-safe.

---

### MCQ 5: Access Modifiers
**Q:** Which members are accessible from a subclass in a different package?
1. private
2. default
3. protected
4. All of the above

**Answer:** 3. protected  
**Explanation:** Protected members are accessible in subclasses even in different package.

---

### MCQ 6: Method Overloading
**Q:** Is this valid overloading?
```java
public int add(int a, int b) { return a + b; }
public double add(int a, int b) { return a + b; }
```
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** Parameters are same. Only return type differs. **Invalid overloading.**

---

### MCQ 7: Constructor
**Q:** What is the output?
```java
public class Test {
    public Test(int a) {
        System.out.println(a);
    }
    
    public static void main(String[] args) {
        Test t = new Test();
    }
}
```
1. 0
2. null
3. Compile error
4. Runtime error

**Answer:** 3. Compile error  
**Explanation:** Parameterized constructor defined, so default constructor not provided.

---

### MCQ 8: Pass by Value
**Q:** What is the output?
```java
public void modify(int x) {
    x = 20;
}

public static void main(String[] args) {
    int a = 10;
    modify(a);
    System.out.println(a);
}
```
1. 10
2. 20
3. 0
4. Compile error

**Answer:** 1. 10  
**Explanation:** Primitives are passed by value. Modifying parameter doesn't affect original.

---

### MCQ 9: Object Reference
**Q:** What is the output?
```java
class Box {
    int value;
}

public void modify(Box b) {
    b.value = 20;
}

public static void main(String[] args) {
    Box box = new Box();
    box.value = 10;
    modify(box);
    System.out.println(box.value);
}
```
1. 10
2. 20
3. 0
4. Compile error

**Answer:** 2. 20  
**Explanation:** Reference is passed. Modifying object fields affects original.

---

### MCQ 10: equals vs ==
**Q:** What is the output?
```java
String s1 = new String("Test");
String s2 = new String("Test");
System.out.println(s1 == s2);
System.out.println(s1.equals(s2));
```
1. true true
2. true false
3. false true
4. false false

**Answer:** 3. false true  
**Explanation:** `==` compares references (different objects). `equals()` compares values.

---

## ⚠️ Common Mistakes

1. **array.length()** → ERROR (property, not method)
2. **String modification** without reassignment
3. **Using ==** instead of equals() for String comparison
4. **Method overloading** with only return type different
5. **Missing default constructor** when parameterized constructor defined
6. **Assuming pass-by-reference** for objects in Java

---

## ⭐ One-liner Exam Facts

1. Array **length** is a **property**, not a method
2. String literals go to **String pool**, `new String()` goes to heap
3. String is **immutable**, StringBuilder and StringBuffer are mutable
4. Use **equals()** for value comparison, **==** for reference comparison
5. **protected** members accessible in subclasses (even different package)
6. Method overloading requires **different parameters** (not just return type)
7. **this()** must be first statement in constructor
8. Java is **pass-by-value** (for objects, value is the reference)
9. StringBuilder is **not thread-safe**, StringBuffer is thread-safe
10. Default constructor provided **only if** no constructor defined

---

**End of Session 4**
