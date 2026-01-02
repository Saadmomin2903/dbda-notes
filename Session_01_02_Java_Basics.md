# Session 1 & 2 – Java Basics

**Topics Covered:** Overview & Features of Java, JVM Overview, Scope of Variables, Object Oriented Concepts, JDK Tools, Java Class Structure, Packages, Object References vs Primitives

---

## 1. Overview & Features of Java

### What is Java?
Java is a **high-level, object-oriented, platform-independent programming language** developed by **Sun Microsystems** (now Oracle) in 1995 by **James Gosling**.

### Key Features

| Feature | Description | Exam Importance |
|---------|-------------|-----------------|
| **Platform Independent** | "Write Once, Run Anywhere" (WORA) - bytecode runs on any JVM | ⭐ High - MCQs on .class vs .java |
| **Object-Oriented** | Everything is an object (except primitives) | ⭐ Medium - OOP principles |
| **Robust** | Strong type checking, exception handling, garbage collection | ⭐ High - GC questions |
| **Secure** | No pointers, bytecode verification, security manager | ⭐ Medium |
| **Multithreaded** | Built-in support for concurrent programming | ⭐ High - Thread lifecycle |
| **Architecture Neutral** | Bytecode is not tied to specific processor | ⭐ Medium |
| **Interpreted** | Bytecode is interpreted by JVM | ⭐ High - JIT questions |
| **High Performance** | JIT compiler improves performance | ⭐ Medium |

⚠️ **Common Mistake:** Students confuse "platform independent" with "portable". Java **source** is portable, Java **bytecode** is platform-independent.

⭐ **Exam Fact:** Java is **compiled to bytecode** (.class), then **interpreted/JIT-compiled** by JVM at runtime.

---

## 2. JVM Overview (Architecture-level Explanation)

### JVM Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JAVA APPLICATION                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   CLASS LOADER SUBSYSTEM                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Loading  │→ │ Linking  │→ │   Init   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   RUNTIME DATA AREAS                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Method Area (Class metadata, static vars, constants)│  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Heap (Objects, instance variables)                  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌───────┐ ┌───────┐ ┌───────┐                            │
│  │Stack 1│ │Stack 2│ │Stack N│ (per thread)               │
│  └───────┘ └───────┘ └───────┘                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ PC Registers (per thread)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Native Method Stacks (per thread)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION ENGINE                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Interpreter│  │JIT Compi│  │ Garbage  │                  │
│  │          │  │  ler     │  │Collector │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              NATIVE METHOD INTERFACE (JNI)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  NATIVE METHOD LIBRARIES                    │
└─────────────────────────────────────────────────────────────┘
```

### JVM Components Explained

#### 1. Class Loader Subsystem
- **Loading**: Reads .class file and loads bytecode into memory
  - Bootstrap ClassLoader (loads core Java classes)
  - Extension ClassLoader (loads from ext directory)
  - Application ClassLoader (loads from classpath)
- **Linking**: 
  - Verification (bytecode verification)
  - Preparation (allocate memory for static variables)
  - Resolution (replace symbolic references with direct references)
- **Initialization**: Execute static initializers and static blocks

⭐ **Exam Fact:** ClassLoaders follow **delegation hierarchy**: Application → Extension → Bootstrap

#### 2. Runtime Data Areas

| Area | Scope | Contents | Shared? |
|------|-------|----------|---------|
| **Method Area** | JVM-wide | Class metadata, static variables, constant pool | YES (all threads) |
| **Heap** | JVM-wide | Objects, instance variables | YES (all threads) |
| **Stack** | Thread-specific | Local variables, method calls, partial results | NO |
| **PC Register** | Thread-specific | Address of current instruction | NO |
| **Native Method Stack** | Thread-specific | Native method information | NO |

⚠️ **Common Mistake:** Students think Stack stores objects. **Stack stores references, Heap stores objects**.

#### 3. Execution Engine
- **Interpreter**: Executes bytecode line-by-line (slower)
- **JIT Compiler**: Compiles frequently used bytecode to native machine code (faster)
- **Garbage Collector**: Automatically reclaims memory from unreferenced objects

⭐ **Exam Fact:** JIT compiles **hot spots** (frequently executed code) to improve performance.

---

## 3. Stack vs Heap Memory

### Memory Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         STACK                               │
│  ┌────────────────────────────────────┐                     │
│  │ main() method frame                │                     │
│  │  ┌──────────────────────────────┐  │                     │
│  │  │ int a = 10;          [10]    │  │                     │
│  │  │ String s = ...;      [0x1A2] │──┼─────┐               │
│  │  │ Person p = ...;      [0x2B3] │──┼───┐ │               │
│  │  └──────────────────────────────┘  │   │ │               │
│  └────────────────────────────────────┘   │ │               │
│                                            │ │               │
└────────────────────────────────────────────┼─┼───────────────┘
                                             │ │
                                             │ │
┌────────────────────────────────────────────┼─┼───────────────┐
│                         HEAP                │ │               │
│                                             ↓ ↓               │
│  ┌──────────────────────────────────────────────────┐        │
│  │ Object @ 0x1A2                                   │        │
│  │  char[] value = ['H','e','l','l','o']           │        │
│  │  (String object)                                 │        │
│  └──────────────────────────────────────────────────┘        │
│                                                               │
│  ┌──────────────────────────────────────────────────┐        │
│  │ Object @ 0x2B3                                   │        │
│  │  String name = [0x1A2] ────────────────────┐     │        │
│  │  int age = 25                              │     │        │
│  │  (Person object)                           │     │        │
│  └────────────────────────────────────────────┼─────┘        │
│                                               │              │
│                (reference loop back)          │              │
└───────────────────────────────────────────────┼──────────────┘
                                                │
                                                └──────────────┐
                                                (same String)  │
```

### Key Differences

| Aspect | Stack | Heap |
|--------|-------|------|
| **Stores** | Primitives, references | Objects, arrays |
| **Scope** | Method-level | Application-level |
| **Lifetime** | Until method returns | Until GC collects |
| **Access Speed** | Faster (LIFO) | Slower |
| **Size** | Limited (StackOverflowError) | Larger (OutOfMemoryError) |
| **Thread Safety** | Thread-specific | Shared (needs sync) |

⚠️ **Common Mistake:** Thinking `String s = "Hello";` stores "Hello" on stack. **Wrong!** Stack stores reference `s`, heap stores String object.

---

## 4. Scope of Variables

### Types of Variables

```java
public class ScopeDemo {
    // 1. STATIC VARIABLE (Class variable)
    static int staticVar = 100;
    
    // 2. INSTANCE VARIABLE (Non-static field)
    int instanceVar = 200;
    
    public void method() {
        // 3. LOCAL VARIABLE (Method variable)
        int localVar = 300;
        
        // Block scope
        {
            int blockVar = 400;
            System.out.println(blockVar); // OK
        }
        // System.out.println(blockVar); // ERROR: out of scope
    }
}
```

### Variable Comparison Table

| Variable Type | Memory Location | Initialized By | Scope | Lifetime | Default Value |
|---------------|-----------------|----------------|-------|----------|---------------|
| **Static** | Method Area | ClassLoader | Class-wide | Until class unloaded | YES (0, null, false) |
| **Instance** | Heap | Constructor/initializer | Object-wide | Until object GCed | YES (0, null, false) |
| **Local** | Stack | Programmer | Method/block | Until method returns | NO (must initialize) |

⚠️ **Common MCQ Trap:**
```java
public void test() {
    int x;
    System.out.println(x); // COMPILE ERROR: variable x might not have been initialized
}
```

⭐ **Exam Fact:** Only **local variables** require explicit initialization. Instance and static variables have default values.

### Default Values Table

| Type | Default Value |
|------|---------------|
| byte, short, int, long | 0 |
| float, double | 0.0 |
| char | '\u0000' |
| boolean | false |
| Reference types | null |

---

## 5. Object-Oriented Concepts in Java

### Four Pillars of OOP

#### 1. Encapsulation
**Definition:** Bundling data (fields) and methods together, hiding internal implementation.

```java
public class BankAccount {
    private double balance; // Hidden
    
    public void deposit(double amount) { // Controlled access
        if (amount > 0) {
            balance += amount;
        }
    }
    
    public double getBalance() {
        return balance;
    }
}
```

⭐ **Exam Fact:** Private fields + public getters/setters = Encapsulation

#### 2. Inheritance
**Definition:** Acquiring properties and behaviors from parent class.

```java
class Animal {
    void eat() { }
}

class Dog extends Animal {
    void bark() { }
}
```

⭐ **Exam Fact:** Java supports **single inheritance** (class level), **multiple inheritance** via interfaces.

#### 3. Polymorphism
**Definition:** One interface, multiple implementations.

```java
Animal a = new Dog(); // Upcasting
a.eat(); // Runtime polymorphism
```

Types:
- **Compile-time (Overloading):** Same method name, different parameters
- **Runtime (Overriding):** Subclass redefines parent method

#### 4. Abstraction
**Definition:** Hiding complex implementation, showing only essential features.

```java
abstract class Shape {
    abstract void draw(); // No implementation
}

class Circle extends Shape {
    void draw() { 
        System.out.println("Drawing circle");
    }
}
```

⚠️ **Common Mistake:** Confusing **abstraction** with **encapsulation**. 
- Abstraction = hiding complexity
- Encapsulation = hiding data

---

## 6. JDK Tools

### Essential Tools

| Tool | Purpose | Command Example | Exam Importance |
|------|---------|-----------------|-----------------|
| **javac** | Java compiler (Java → bytecode) | `javac Hello.java` | ⭐ High |
| **java** | JVM launcher | `java Hello` | ⭐ High |
| **jdb** | Java debugger | `jdb Hello` | ⭐ Low |
| **javadoc** | Documentation generator | `javadoc *.java` | ⭐ Medium |
| **jar** | Archive tool | `jar cf app.jar *.class` | ⭐ Medium |
| **javap** | Class file disassembler | `javap -c Hello` | ⭐ Low |

### Detailed Examples

#### javac (Compiler)
```bash
# Compile single file
javac Hello.java

# Compile with classpath
javac -cp lib/commons.jar MyApp.java

# Compile with output directory
javac -d bin src/*.java
```

⭐ **Exam Fact:** `javac` produces **.class** files (bytecode), not machine code.

#### java (JVM Launcher)
```bash
# Run class (NO .class extension)
java Hello

# Run with classpath
java  -cp .:lib/commons.jar MyApp

# Run with system properties
java -Dfile.encoding=UTF-8 MyApp
```

⚠️ **Common Mistake:** Running `java Hello.class` → **ERROR!** Should be `java Hello`

#### javadoc (Documentation)
```java
/**
 * This class represents a simple calculator.
 * 
 * @author John Doe
 * @version 1.0
 */
public class Calculator {
    /**
     * Adds two integers.
     * 
     * @param a first number
     * @param b second number
     * @return sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
```

```bash
javadoc Calculator.java
# Generates HTML documentation
```

---

## 7. Java Class Structure

### Anatomy of a Java Class

```java
// 1. Package declaration (optional, must be first)
package com.example;

// 2. Import statements
import java.util.ArrayList;
import java.util.List;

// 3. Class declaration
public class Employee {
    
    // 4. Static variables
    static int employeeCount = 0;
    
    // 5. Instance variables
    private String name;
    private int id;
    
    // 6. Static block (executed once when class loads)
    static {
        System.out.println("Class loaded");
    }
    
    // 7. Instance block (executed before every constructor)
    {
        employeeCount++;
    }
    
    // 8. Constructor
    public Employee(String name, int id) {
        this.name = name;
        this.id = id;
    }
    
    // 9. Methods
    public void work() {
        System.out.println(name + " is working");
    }
    
    // 10. Static method
    public static int getEmployeeCount() {
        return employeeCount;
    }
    
    // 11. Inner class (optional)
    class Task {
        void execute() { }
    }
}
```

### Execution Order

```
1. Static block (once, when class loads)
2. Instance block (every object creation)
3. Constructor (every object creation)
```

**Example:**
```java
public class InitOrder {
    static { System.out.println("1. Static block"); }
    { System.out.println("2. Instance block"); }
    public InitOrder() { System.out.println("3. Constructor"); }
    
    public static void main(String[] args) {
        new InitOrder();
        new InitOrder();
    }
}
```

**Output:**
```
1. Static block
2. Instance block
3. Constructor
2. Instance block
3. Constructor
```

⚠️ **Common MCQ Trap:** Static block runs **only once**, instance block runs **every time** object is created.

---

## 8. Packages & Importing

### What are Packages?
Packages organize classes into namespaces, preventing name conflicts.

**Syntax:**
```java
package com.company.project;
```

### Package Naming Convention
- All lowercase
- Reverse domain name: `com.example.myapp`

### Types of Packages

#### 1. Built-in Packages
- `java.lang` (auto-imported)
- `java.util`
- `java.io`
- `java.net`
- etc.

#### 2. User-defined Packages

**File structure:**
```
src/
  com/
    example/
      util/
        Helper.java
      model/
        User.java
```

**Helper.java:**
```java
package com.example.util;

public class Helper {
    public static void print(String msg) {
        System.out.println(msg);
    }
}
```

### Import Statements

```java
// 1. Import specific class
import java.util.ArrayList;

// 2. Import all classes from package (NOT recommended)
import java.util.*;

// 3. Import static member
import static java.lang.Math.PI;
import static java.lang.Math.sqrt;

// Usage
double area = PI * r * r;
double side = sqrt(area);
```

⚠️ **Common Mistake:** 
```java
import java.util.*; // Does NOT import java.util.regex.*
```

⭐ **Exam Fact:** `java.lang` is **automatically imported** in every Java program.

### Fully Qualified Names
```java
// Without import
java.util.ArrayList<String> list = new java.util.ArrayList<>();

// With import
import java.util.ArrayList;
ArrayList<String> list = new ArrayList<>();
```

---

## 9. Object Reference vs Primitive Variables

### Primitives vs References

```java
// PRIMITIVE
int a = 10;        // Stores actual value
int b = a;         // Copies value
b = 20;
System.out.println(a); // 10 (unchanged)

// REFERENCE
Person p1 = new Person("Alice");  // Stores reference (address)
Person p2 = p1;                   // Copies reference
p2.name = "Bob";
System.out.println(p1.name);      // Bob (both point to same object!)
```

### Memory Representation

```
PRIMITIVES:
┌──────┐
│  a   │ = 10
└──────┘
┌──────┐
│  b   │ = 20
└──────┘

REFERENCES:
┌──────┐           ┌─────────────┐
│  p1  │ ─────────→│ Person obj  │
└──────┘           │ name="Bob"  │
┌──────┐           └─────────────┘
│  p2  │ ─────────→     ↑
└──────┘                 │
         (same object)───┘
```

### Primitive Types in Java

| Type | Size | Range | Default |
|------|------|-------|---------|
| byte | 8-bit | -128 to 127 | 0 |
| short | 16-bit | -32,768 to 32,767 | 0 |
| int | 32-bit | -2³¹ to 2³¹-1 | 0 |
| long | 64-bit | -2⁶³ to 2⁶³-1 | 0L |
| float | 32-bit | ~±3.4E38 | 0.0f |
| double | 64-bit | ~±1.7E308 | 0.0d |
| char | 16-bit | 0 to 65,535 (Unicode) | '\u0000' |
| boolean | 1-bit* | true/false | false |

*JVM-dependent, typically 8-bit

⚠️ **Common MCQ Trap:**
```java
char c = 'A';  // OK
char c = 65;   // OK (ASCII value)
char c = -1;   // ERROR: char is unsigned
```

---

## 10. Reading/Writing Object Fields

### Accessing Fields

```java
public class Car {
    String model;
    int year;
    private String vin;
    
    public String getVin() {
        return vin;
    }
    
    public void setVin(String vin) {
        this.vin = vin;  // 'this' refers to current object
    }
}

// Usage
Car car = new Car();
car.model = "Tesla";      // Direct access (public)
car.year = 2024;
// car.vin = "ABC123";    // ERROR: private
car.setVin("ABC123");     // OK via setter
```

### The 'this' Keyword

```java
public class Employee {
    String name;
    
    // Constructor
    public Employee(String name) {
        this.name = name;  // this.name = instance variable
                          // name = parameter
    }
    
    // Method returning current object
    public Employee setName(String name) {
        this.name = name;
        return this;  // Method chaining
    }
}

// Method chaining
Employee emp = new Employee("Alice")
                    .setName("Bob");
```

⭐ **Exam Fact:** `this` is a **reference to the current object**.

---

## 🔥 Top MCQs for Session 1-2

### MCQ 1: JVM Components
**Q:** Which JVM component is shared among all threads?
1. Stack
2. PC Register
3. Heap
4. Native Method Stack

**Answer:** 3. Heap  
**Explanation:** Heap and Method Area are shared. Stack, PC Register, and Native Method Stack are per-thread.

---

### MCQ 2: Variable Scope
**Q:** What is the output?
```java
public class Test {
    static int x = 10;
    int y = 20;
    
    public static void main(String[] args) {
        System.out.println(x);
        System.out.println(y); // Line 7
    }
}
```
1. 10 20
2. 10 0
3. Compile error at line 7
4. Runtime error

**Answer:** 3. Compile error at line 7  
**Explanation:** `y` is an instance variable, cannot be accessed from static context without object.

---

### MCQ 3: Memory Management
**Q:** Where is the String literal pool located in Java 8+?
1. Stack
2. Method Area
3. Heap
4. PC Register

**Answer:** 3. Heap  
**Explanation:** Since Java 7, String pool moved from PermGen (Method Area) to Heap.

---

### MCQ 4: Class Loader
**Q:** Which ClassLoader loads `java.lang.String`?
1. Application ClassLoader
2. Extension ClassLoader
3. Bootstrap ClassLoader
4. Custom ClassLoader

**Answer:** 3. Bootstrap ClassLoader  
**Explanation:** Bootstrap loads core Java classes (java.lang, java.util, etc.)

---

### MCQ 5: Variable Initialization
**Q:** What is the output?
```java
public class Test {
    int x;
    static int y;
    
    public static void main(String[] args) {
        Test t = new Test();
        System.out.println(t.x + " " + y);
    }
}
```
1. 0 0
2. null null
3. Compile error
4. Runtime error

**Answer:** 1. 0 0  
**Explanation:** Instance (x) and static (y) variables have default value 0 for int.

---

### MCQ 6: Reference vs Value
**Q:** What is the output?
```java
public class Test {
    public static void main(String[] args) {
        int a = 10;
        int b = a;
        b = 20;
        System.out.println(a);
    }
}
```
1. 10
2. 20
3. 0
4. Compile error

**Answer:** 1. 10  
**Explanation:** Primitives are passed by value. Changing `b` doesn't affect `a`.

---

### MCQ 7: Object References
**Q:** What is the output?
```java
class Box {
    int value;
}

public class Test {
    public static void main(String[] args) {
        Box b1 = new Box();
        b1.value = 10;
        Box b2 = b1;
        b2.value = 20;
        System.out.println(b1.value);
    }
}
```
1. 10
2. 20
3. 0
4. Compile error

**Answer:** 2. 20  
**Explanation:** b1 and b2 reference the **same object**. Changing via b2 affects b1.

---

### MCQ 8: Package & Import
**Q:** Which package is automatically imported?
1. java.util
2. java.io
3. java.lang
4. java.net

**Answer:** 3. java.lang  
**Explanation:** `java.lang` is implicitly imported in all Java programs.

---

### MCQ 9: JDK Tools
**Q:** Which tool compiles Java source code?
1. java
2. javac
3. javadoc
4. jar

**Answer:** 2. javac  
**Explanation:** `javac` = Java compiler (`.java` → `.class`)

---

### MCQ 10: Static vs Instance
**Q:** What is the output?
```java
public class Test {
    static {
        System.out.print("A");
    }
    
    {
        System.out.print("B");
    }
    
    public Test() {
        System.out.print("C");
    }
    
    public static void main(String[] args) {
        new Test();
        new Test();
    }
}
```
1. ABCABC
2. ABCBC
3. ABC
4. AABBCC

**Answer:** 2. ABCBC  
**Explanation:** Static block (A) runs once. Instance block (B) and constructor (C) run for each object.

---

### MCQ 11: Primitive Range
**Q:** Which statement is TRUE?
1. byte range is -127 to 128
2. char can store negative values
3. boolean size is 1 bit
4. int range is -2³¹ to 2³¹-1

**Answer:** 4. int range is -2³¹ to 2³¹-1  
**Explanation:**  
- byte: -128 to 127 (not -127 to 128)
- char: unsigned (0 to 65535)
- boolean: JVM-dependent (not guaranteed 1 bit)

---

### MCQ 12: Execution Order
**Q:** What is the output?
```java
public class Parent {
    static { System.out.print("1"); }
    { System.out.print("2"); }
    public Parent() { System.out.print("3"); }
}

public class Child extends Parent {
    static { System.out.print("4"); }
    { System.out.print("5"); }
    public Child() { System.out.print("6"); }
    
    public static void main(String[] args) {
        new Child();
    }
}
```
1. 123456
2. 142536
3. 412536
4. 145236

**Answer:** 2. 142536  
**Explanation:**  
1. Parent static block (1)
2. Child static block (4)
3. Parent instance block (2)
4. Parent constructor (3)
5. Child instance block (5)
6. Child constructor (6)

---

## ⚠️ Common Mistakes Summary

1. **Confusing .java and .class**: Compile `.java`, run class name (not `.class`)
2. **Local variable initialization**: Must initialize before use
3. **Static context**: Cannot access instance members from static methods directly
4. **Reference copying**: Copying reference doesn't copy object
5. **Import wildcard**: `import java.util.*` doesn't import subpackages
6. **Package naming**: File directory must match package declaration
7. **char range**: char is unsigned (0-65535), no negative values
8. **Stack vs Heap**: Stack stores primitives + references, Heap stores objects

---

## ⭐ One-liner Exam Facts

1. Java is **compiled** to bytecode, then **interpreted** by JVM
2. **Heap** and **Method Area** are shared among threads
3. **Stack** is thread-specific and stores local variables
4. Only **local variables** require explicit initialization
5. `java.lang` is **automatically imported**
6. Static block runs **once**, instance block runs **per object**
7. ClassLoader hierarchy: **Application → Extension → Bootstrap**
8. JIT compiles **hot spots** for better performance
9. **javac** compiles, **java** executes
10. `this` refers to **current object**
11. **String pool** is in Heap (Java 7+)
12. Primitives are **passed by value**, objects by **reference**

---

## 📚 References

### Official Documentation
- Oracle Java Documentation: https://docs.oracle.com/javase/
- JVM Specification: https://docs.oracle.com/javase/specs/jvms/se17/html/
- Java Language Specification: https://docs.oracle.com/javase/specs/jls/se17/html/

### Man Pages (Linux)
```bash
man javac
man java
man javadoc
```

### Books
- "Thinking in Java" by Bruce Eckel
- "Effective Java" by Joshua Bloch
- "Head First Java" by Kathy Sierra

---

**End of Session 1-2**
