# Session 3 – Object Lifecycle & Operators

**Topics Covered:** Object Creation & Reassignment, Garbage Collection, Wrapper Classes, Operators (Unary, Binary, Arithmetic, Logical, Relational, Ternary), Control Statements

---

## 1. Object Creation & Reassignment

### Creating Objects

```java
// Method 1: Using 'new' keyword (most common)
Person p1 = new Person("Alice", 25);

// Method 2: Using Class.forName() + newInstance()
Class<?> clazz = Class.forName("Person");
Person p2 = (Person) clazz.newInstance(); // Deprecated in Java 9+

// Method 3: Using clone()
Person p3 = (Person) p1.clone();

// Method 4: Using deserialization
ObjectInputStream ois = new ObjectInputStream(fis);
Person p4 = (Person) ois.readObject();

// Method 5: Using factory method
Person p5 = Person.create("Bob", 30);
```

### Object Creation Process

```
1. Memory allocation in Heap
2. Instance variables initialized to default values
3. Constructor invoked
4. Reference returned to stack
```

### Object Reassignment

```java
Person p1 = new Person("Alice");  // Object 1 created
Person p2 = new Person("Bob");    // Object 2 created
p1 = p2;                          // p1 now points to Object 2
                                  // Object 1 becomes eligible for GC
```

**Memory Diagram:**
```
Before reassignment:
┌────┐        ┌──────────────┐
│ p1 │───────→│ Object 1     │
└────┘        │ name="Alice" │
┌────┐        └──────────────┘
│ p2 │───────→┌──────────────┐
└────┘        │ Object 2     │
              │ name="Bob"   │
              └──────────────┘

After p1 = p2:
              ┌──────────────┐
              │ Object 1     │ ← ELIGIBLE FOR GC (no references)
              │ name="Alice" │
              └──────────────┘
┌────┐        
│ p1 │───────→┌──────────────┐
└────┘        │ Object 2     │
┌────┐        │ name="Bob"   │
│ p2 │───────→└──────────────┘
└────┘        (2 references to Object 2)
```

⭐ **Exam Fact:** An object becomes **eligible for GC** when it has **no references** pointing to it.

---

## 2. Garbage Collection (GC)

### What is Garbage Collection?
Automatic memory management that reclaims memory from unreachable objects.

### When is an Object Eligible for GC?

#### Case 1: Nullifying Reference
```java
Person p = new Person("Alice");
p = null;  // Object eligible for GC
```

#### Case 2: Reassigning Reference
```java
Person p = new Person("Alice");
p = new Person("Bob");  // "Alice" object eligible for GC
```

#### Case 3: Object Goes Out of Scope
```java
void method() {
    Person p = new Person("Alice");
}  // p goes out of scope, object eligible for GC
```

#### Case 4: Island of Isolation
```java
class Node {
    Node next;
}

Node n1 = new Node();
Node n2 = new Node();
n1.next = n2;
n2.next = n1;  // Circular reference

n1 = null;
n2 = null;  // Both objects eligible for GC (island of isolation)
```

### finalize() Method

```java
protected void finalize() throws Throwable {
    System.out.println("Object being garbage collected");
    // Cleanup code (close files, release resources)
}
```

⚠️ **Common GC Myths:**

| Myth | Reality |
|------|---------|
| Calling `System.gc()` **guarantees** GC | `System.gc()` is a **request**, JVM may ignore it |
| `finalize()` always runs before GC | Not guaranteed, may never run |
| GC happens immediately | GC runs when JVM decides (non-deterministic) |
| Objects are deleted | Objects are **reclaimed**, not deleted |

⭐ **Exam Facts:**
1. **You cannot force GC**, only request via `System.gc()` or `Runtime.getRuntime().gc()`
2. `finalize()` is called **at most once** per object
3. `finalize()` is **deprecated in Java 9+**
4. GC only works on **Heap memory**, not Stack

---

## 3. Wrapper Classes

### Why Wrapper Classes?
1. To treat primitives as objects
2. For use in collections (ArrayList, HashMap, etc.)
3. Utility methods (parseInt, toString, etc.)

### Wrapper Class Hierarchy

```
         Object
            |
   ┌────────┴────────┐
   │                 │
Number           Boolean
   |              Character
   ├─── Byte
   ├─── Short
   ├─── Integer
   ├─── Long
   ├─── Float
   └─── Double
```

### Primitive-Wrapper Mapping

| Primitive | Wrapper Class |
|-----------|---------------|
| byte | Byte |
| short | Short |
| int | Integer |
| long | Long |
| float | Float |
| double | Double |
| char | Character |
| boolean | Boolean |

### Creating Wrapper Objects

```java
// Old way (deprecated in Java 9+)
Integer i1 = new Integer(10);

// Recommended way
Integer i2 = Integer.valueOf(10);

// Autoboxing (automatic conversion)
Integer i3 = 10;  // Compiler converts to Integer.valueOf(10)
```

### Wrapper Class Methods

```java
// Integer
Integer i = 100;
int primitive = i.intValue();           // Wrapper → Primitive
String s = i.toString();                // "100"
int parsed = Integer.parseInt("123");   // String → int
Integer wrapped = Integer.valueOf(50);  // int → Integer

// Comparison
Integer a = 100;
Integer b = 100;
System.out.println(a == b);             // true (cached)
System.out.println(a.equals(b));        // true

Integer c = 200;
Integer d = 200;
System.out.println(c == d);             // false (not cached)
System.out.println(c.equals(d));        // true
```

### ⚠️ Integer Caching (Common MCQ Trap!)

**Java caches Integer objects from -128 to 127.**

```java
Integer a = 127;
Integer b = 127;
System.out.println(a == b);  // true (cached)

Integer c = 128;
Integer d = 128;
System.out.println(c == d);  // false (NOT cached, different objects)
```

**Why?** Performance optimization. Small integers are frequently used.

**Cache ranges for different wrappers:**

| Wrapper | Cache Range |
|---------|-------------|
| Integer | -128 to 127 |
| Byte | -128 to 127 (all values) |
| Short | -128 to 127 |
| Long | -128 to 127 |
| Character | 0 to 127 |
| Boolean | TRUE, FALSE (both cached) |
| Float | Not cached |
| Double | Not cached |

⭐ **Exam Fact:** Always use `.equals()` for wrapper comparison, not `==`.

### Autoboxing & Unboxing

```java
// Autoboxing (primitive → wrapper)
Integer i = 10;  // int → Integer

// Unboxing (wrapper → primitive)
int j = i;  // Integer → int

// In collections
List<Integer> list = new ArrayList<>();
list.add(10);  // Autoboxing
int val = list.get(0);  // Unboxing
```

⚠️ **NullPointerException Trap:**
```java
Integer i = null;
int j = i;  // NullPointerException during unboxing!
```

---

## 4. Operators

### Operator Precedence (Highest to Lowest)

| Precedence | Operator | Description |
|------------|----------|-------------|
| 1 | `()`, `[]`, `.` | Parentheses, array access, member access |
| 2 | `++`, `--` (postfix) | Postfix increment/decrement |
| 3 | `++`, `--` (prefix), `+`, `-`, `!`, `~` | Prefix, unary plus/minus, NOT |
| 4 | `*`, `/`, `%` | Multiplicative |
| 5 | `+`, `-` | Additive |
| 6 | `<<`, `>>`, `>>>` | Shift |
| 7 | `<`, `<=`, `>`, `>=`, `instanceof` | Relational |
| 8 | `==`, `!=` | Equality |
| 9 | `&` | Bitwise AND |
| 10 | `^` | Bitwise XOR |
| 11 | `|` | Bitwise OR |
| 12 | `&&` | Logical AND |
| 13 | `||` | Logical OR |
| 14 | `?:` | Ternary |
| 15 | `=`, `+=`, `-=`, etc. | Assignment |

⭐ **Mnemonic:** **P U M A S R E B L T A** (Parentheses, Unary, Multiplicative, Additive, Shift, Relational, Equality, Bitwise, Logical, Ternary, Assignment)

### 1. Unary Operators

```java
int a = 10;

// Unary plus/minus
int b = +a;  // 10
int c = -a;  // -10

// Increment/Decrement
int x = 5;
System.out.println(x++);  // 5 (post-increment: use then increment)
System.out.println(x);    // 6

int y = 5;
System.out.println(++y);  // 6 (pre-increment: increment then use)
System.out.println(y);    // 6

// Logical NOT
boolean flag = true;
System.out.println(!flag);  // false
```

⚠️ **Common MCQ Trap:**
```java
int i = 10;
int j = i++ + ++i;
// Step 1: i++ → use 10, then i becomes 11
// Step 2: ++i → i becomes 12, use 12
// Result: j = 10 + 12 = 22
```

### 2. Arithmetic Operators

```java
int a = 10, b = 3;

System.out.println(a + b);  // 13
System.out.println(a - b);  // 7
System.out.println(a * b);  // 30
System.out.println(a / b);  // 3 (integer division)
System.out.println(a % b);  // 1 (modulus)

double x = 10.0, y = 3.0;
System.out.println(x / y);  // 3.3333...
```

⚠️ **Division by Zero:**
```java
int a = 10 / 0;        // ArithmeticException
double b = 10.0 / 0;   // Infinity
double c = 0.0 / 0;    // NaN (Not a Number)
```

### 3. Relational Operators

```java
int a = 10, b = 20;

System.out.println(a > b);   // false
System.out.println(a < b);   // true
System.out.println(a >= b);  // false
System.out.println(a <= b);  // true
System.out.println(a == b);  // false
System.out.println(a != b);  // true
```

### 4. Logical Operators

```java
boolean a = true, b = false;

// AND (both must be true)
System.out.println(a && b);  // false

// OR (at least one must be true)
System.out.println(a || b);  // true

// NOT (inverts boolean)
System.out.println(!a);      // false
```

**Short-circuit evaluation:**
```java
int x = 10;
if (x > 5 && x++ < 20) {  // x++ never executes if x > 5 is false
    System.out.println(x);
}
```

⭐ **Exam Fact:**
- `&&` and `||` are **short-circuit** operators
- `&` and `|` are **non-short-circuit** (evaluate both sides)

```java
int a = 10;
if (a > 5 && ++a > 10) {  // ++a executes (short-circuit doesn't apply)
    System.out.println(a);  // 11
}

int b = 3;
if (b > 5 && ++b > 10) {  // ++b doesn't execute (short-circuit)
    System.out.println(b);
}
System.out.println(b);  // 3 (not 4)
```

### 5. Bitwise Operators

```java
int a = 5;   // 0101 in binary
int b = 3;   // 0011 in binary

System.out.println(a & b);   // 1 (0001) - AND
System.out.println(a | b);   // 7 (0111) - OR
System.out.println(a ^ b);   // 6 (0110) - XOR
System.out.println(~a);      // -6 (1010 in 2's complement) - NOT

System.out.println(a << 1);  // 10 (1010) - Left shift (multiply by 2)
System.out.println(a >> 1);  // 2 (0010) - Right shift (divide by 2)
System.out.println(a >>> 1); // 2 - Unsigned right shift
```

### 6. Ternary Operator

```java
int a = 10, b = 20;
int max = (a > b) ? a : b;  // max = 20

// Nested ternary
int x = 5;
String result = (x > 0) ? "Positive" : (x < 0) ? "Negative" : "Zero";
```

⚠️ **Common Mistake:**
```java
int a = 10;
int b = (a > 5) ? 100 : "Less than 5";  // ERROR: type mismatch
```

### 7. Assignment Operators

```java
int a = 10;

a += 5;   // a = a + 5;  → 15
a -= 3;   // a = a - 3;  → 12
a *= 2;   // a = a * 2;  → 24
a /= 4;   // a = a / 4;  → 6
a %= 4;   // a = a % 4;  → 2
```

⚠️ **Compound Assignment Type Casting:**
```java
byte b = 10;
b = b + 1;   // ERROR: incompatible types (int cannot be converted to byte)
b += 1;      // OK (implicit cast to byte)
```

---

## 5. Control Statements

### 1. if-else

```java
int score = 85;

if (score >= 90) {
    System.out.println("A");
} else if (score >= 80) {
    System.out.println("B");
} else if (score >= 70) {
    System.out.println("C");
} else {
    System.out.println("F");
}
```

### 2. switch

```java
int day = 3;

switch (day) {
    case 1:
        System.out.println("Monday");
        break;
    case 2:
        System.out.println("Tuesday");
        break;
    case 3:
        System.out.println("Wednesday");
        break;
    default:
        System.out.println("Other day");
}
```

⚠️ **Fall-through behavior:**
```java
int x = 2;
switch (x) {
    case 1:
        System.out.println("One");
    case 2:
        System.out.println("Two");  // Prints
    case 3:
        System.out.println("Three"); // Prints (no break!)
        break;
    default:
        System.out.println("Default");
}
// Output: Two Three
```

**Supported types in switch (Java 7+):**
- byte, short, int, char
- Byte, Short, Integer, Character
- String
- Enum

⭐ **Exam Fact:** switch **does NOT** support long, float, double, boolean.

### 3. Loops

#### for loop
```java
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// Enhanced for loop
int[] arr = {1, 2, 3, 4, 5};
for (int num : arr) {
    System.out.println(num);
}
```

#### while loop
```java
int i = 0;
while (i < 5) {
    System.out.println(i);
    i++;
}
```

#### do-while loop
```java
int i = 0;
do {
    System.out.println(i);
    i++;
} while (i < 5);
```

⭐ **Exam Fact:** do-while executes **at least once**, while may not execute at all.

### 4. break & continue

```java
// break (exit loop)
for (int i = 0; i < 10; i++) {
    if (i == 5) break;
    System.out.println(i);  // 0 1 2 3 4
}

// continue (skip current iteration)
for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) continue;
    System.out.println(i);  // 1 3 5 7 9
}
```

### 5. Labeled break/continue

```java
outer:
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        if (i == 1 && j == 1) break outer;
        System.out.println(i + "," + j);
    }
}
// Output: 0,0  0,1  0,2  1,0
```

---

## 🔥 Top MCQs for Session 3

### MCQ 1: GC Eligibility
**Q:** After which line is the object eligible for GC?
```java
Person p1 = new Person("A");  // Line 1
Person p2 = new Person("B");  // Line 2
p1 = p2;                      // Line 3
p2 = null;                    // Line 4
```
1. Line 1
2. Line 2
3. Line 3
4. Line 4

**Answer:** 3. Line 3  
**Explanation:** Person("A") has no references after `p1 = p2`. Person("B") still has reference p1.

---

### MCQ 2: Integer Caching
**Q:** What is the output?
```java
Integer a = 100;
Integer b = 100;
Integer c = 200;
Integer d = 200;
System.out.println((a == b) + " " + (c == d));
```
1. true true
2. true false
3. false true
4. false false

**Answer:** 2. true false  
**Explanation:** 100 is cached (-128 to 127), 200 is not.

---

### MCQ 3: Autoboxing NullPointerException
**Q:** What happens?
```java
Integer i = null;
int j = i;
```
1. j = 0
2. j = null
3. NullPointerException
4. Compile error

**Answer:** 3. NullPointerException  
**Explanation:** Unboxing null wrapper throws NPE.

---

### MCQ 4: Operator Precedence
**Q:** What is the value of x?
```java
int x = 5 + 3 * 2;
```
1. 11
2. 16
3. 13
4. 10

**Answer:** 1. 11  
**Explanation:** `*` has higher precedence than `+`. So 3 * 2 = 6, then 5 + 6 = 11.

---

### MCQ 5: Increment Operators
**Q:** What is the output?
```java
int i = 5;
int j = i++ + ++i;
System.out.println(j);
```
1. 11
2. 12
3. 13
4. 10

**Answer:** 2. 12  
**Explanation:**  
- `i++` → use 5, then i = 6
- `++i` → i = 7, use 7
- j = 5 + 7 = 12

---

### MCQ 6: Division by Zero
**Q:** What is the result?
```java
double d = 10.0 / 0;
System.out.println(d);
```
1. ArithmeticException
2. Infinity
3. NaN
4. 0.0

**Answer:** 2. Infinity  
**Explanation:** Floating-point division by zero results in Infinity (not exception).

---

### MCQ 7: Short-circuit Evaluation
**Q:** What is the output?
```java
int x = 5;
if (x < 3 && ++x > 5) {
    System.out.println("A");
}
System.out.println(x);
```
1. A 6
2. A 5
3. 6
4. 5

**Answer:** 4. 5  
**Explanation:** `x < 3` is false, so `++x` never executes (short-circuit).

---

### MCQ 8: switch Fall-through
**Q:** What is the output?
```java
int x = 2;
switch (x) {
    case 1: System.out.print("A");
    case 2: System.out.print("B");
    case 3: System.out.print("C");
    default: System.out.print("D");
}
```
1. B
2. BC
3. BCD
4. ABCD

**Answer:** 3. BCD  
**Explanation:** No break statements, so falls through from case 2 to default.

---

### MCQ 9: Wrapper Comparison
**Q:** What is the output?
```java
Integer a = new Integer(10);
Integer b = new Integer(10);
System.out.println(a == b);
System.out.println(a.equals(b));
```
1. true true
2. true false
3. false true
4. false false

**Answer:** 3. false true  
**Explanation:** `new` creates different objects (== false), but values are same (equals true).

---

### MCQ 10: do-while Loop
**Q:** How many times does the loop execute?
```java
int i = 10;
do {
    System.out.println(i);
} while (i < 5);
```
1. 0 times
2. 1 time
3. 5 times
4. 10 times

**Answer:** 2. 1 time  
**Explanation:** do-while executes **at least once** before checking condition.

---

### MCQ 11: Ternary Operator
**Q:** What is the output?
```java
int a = 10, b = 20;
int result = a > b ? a++ : ++b;
System.out.println(result + " " + a + " " + b);
```
1. 10 11 20
2. 21 10 21
3. 20 10 21
4. 21 10 20

**Answer:** 2. 21 10 21  
**Explanation:** `a > b` is false, so `++b` executes (b becomes 21). `a` remains 10.

---

### MCQ 12: finalize() Method
**Q:** Which statement is TRUE about finalize()?
1. finalize() is called before object is created
2. finalize() is guaranteed to run before GC
3. finalize() can be called multiple times
4. finalize() is deprecated in Java 9+

**Answer:** 4. finalize() is deprecated in Java 9+  
**Explanation:**  
- Called during GC (not before creation)
- Not guaranteed to run
- Called **at most once**
- Deprecated in Java 9+

---

## ⚠️ Common Mistakes Summary

1. **GC control**: Cannot force GC, only request via `System.gc()`
2. **Integer caching**: Use `.equals()` for comparison, not `==`
3. **NullPointerException**: Unboxing null wrapper causes NPE
4. **Operator precedence**: `*` and `/` before `+` and `-`
5. **Increment operators**: `i++` vs `++i` different behavior
6. **Division by zero**: int throws exception, double gives Infinity/NaN
7. **Short-circuit**: `&&` and `||` don't evaluate second operand if not needed
8. **switch fall-through**: Missing `break` causes fall-through
9. **do-while**: Executes at least once
10. **Wrapper constructors**: `new Integer(10)` deprecated, use `Integer.valueOf(10)`

---

## ⭐ One-liner Exam Facts

1. An object is **GC eligible** when it has **no references**
2. `System.gc()` is a **request**, not a guarantee
3. Integer caching range: **-128 to 127**
4. Always use **`.equals()`** for wrapper comparison
5. Unboxing **null wrapper** → **NullPointerException**
6. `*` and `/` have **higher precedence** than `+` and `-`
7. `i++` → **use then increment**, `++i` → **increment then use**
8. Floating-point division by zero → **Infinity** (not exception)
9. `&&` and `||` are **short-circuit** operators
10. switch supports: **byte, short, int, char, String, Enum** (NOT long, float, double)
11. do-while executes **at least once**
12. `finalize()` is **deprecated in Java 9+**

---

## 📚 References

### Official Documentation
- Java Language Specification: https://docs.oracle.com/javase/specs/jls/se17/html/
- Garbage Collection Tuning: https://docs.oracle.com/en/java/javase/17/gctuning/

### Man Pages
```bash
man java
```

---

**End of Session 3**
