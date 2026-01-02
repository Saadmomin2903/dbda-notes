# Java Programming - Quick Reference Card

**PG-DBDA Complete Reference** | All 17 Sessions Covered

---

## 📋 Session Index

| Session | Topics | File |
|---------|--------|------|
| 1-2 | Java Basics, JVM, OOP, Scopes | [Session_01_02](Session_01_02_Java_Basics.md) |
| 3 | Object Lifecycle, GC, Operators | [Session_03](Session_03_Object_Lifecycle_Operators.md) |
| 4 | Arrays, Strings, Encapsulation | [Session_04](Session_04_Arrays_Strings_Encapsulation.md) |
| 5 | Inheritance, Polymorphism | [Session_05](Session_05_Inheritance_Polymorphism.md) |
| 6 | Exception Handling | [Session_06](Session_06_Exception_Handling.md) |
| 7 | Enum, Autoboxing, Annotations | [Session_07](Session_07_Enum_Autoboxing_Annotations.md) |
| 8 | java.lang & java.util | [Session_08](Session_08_Java_Lang_Util.md) |
| 9-10 | Generics & Collections | [Session_09_10](Session_09_10_Generics_Collections.md) |
| 11 | Functional Programming | [Session_11](Session_11_Functional_Programming.md) |
| 12 | Streams & Date/Time API | [Session_12](Session_12_Streams_DateTime.md) |
| 13-14 | Concurrency & Multithreading | [Session_13_14](Session_13_14_Concurrency.md) |
| 15 | IO & Serialization | [Session_15](Session_15_IO_Serialization.md) |
| 16 | JVM Internals & Reflection | [Session_16](Session_16_JVM_Reflection.md) |
| 17 | JDBC | [Session_17](Session_17_JDBC.md) |

---

## 🎯 Top 100 Exam Facts

### JVM & Memory (Sessions 1-2)
1. Java is **compiled** to bytecode, then **interpreted** by JVM
2. **Heap** and **Method Area** shared among threads
3. **Stack** thread-specific, stores local variables & references
4. Only **local variables** require explicit initialization
5. `java.lang` **automatically imported**
6. Static block runs **once**, instance block **per object**
7. ClassLoader hierarchy: **Application → Extension → Bootstrap**
8. JIT compiles **hot spots** for performance
9. **javac** compiles, **java** executes
10. Stack stores **primitives + references**, Heap stores **objects**

### Object Lifecycle & GC (Session 3)
11. Object GC eligible when **no references**
12. `System.gc()` is **request**, not guarantee
13. Integer caching: **-128 to 127**
14. Unboxing **null wrapper** → **NullPointerException**
15. finalize() **deprecated in Java 9+**
16. finalize() called **at most once**

### Operators & Control (Session 3)
17. `i++` → **use then increment**, `++i` → **increment then use**
18. `*` and `/` **higher precedence** than `+` and `-`
19. Floating-point division by zero → **Infinity** (not exception)
20. `&&` and `||` are **short-circuit** operators
21. switch supports: **byte, short, int, char, String, Enum**
22. do-while executes **at least once**

### Strings & Arrays (Session 4)
23. String is **immutable**, StringBuilder is **mutable**
24. Array **length** is **property**, not method
25. Use **equals()** for value comparison, **==** for reference
26. String pool exists **only for String**, not StringBuilder/StringBuffer
27. StringBuilder **not thread-safe**, StringBuffer **thread-safe**

### Passing Data (Session 4)
28. Java is **pass-by-value** (for objects, value is reference)
29. Cannot change **where reference points**, can modify **object content**

### Inheritance & Polymorphism (Session 5)
30. Java supports **single inheritance** (class level)
31. **super()** or **this()** must be first in constructor
32. Abstract class **can have constructor**
33. Interface variables **public static final** by default
34. Override access modifier: **same or wider**, not narrower
35. **Cannot override** private, static, final methods
36. Use **instanceof** before downcasting
37. Virtual method call resolved at **runtime**
38. **Covariant return type** allowed in overriding

### Exception Handling (Session 6)
39. Checked exceptions extend **Exception** (not RuntimeException)
40. Unchecked extend **RuntimeException** or **Error**
41. catch blocks: **specific to generic**
42. finally executes **even with return**
43. Exception in finally **masks** exception from try/catch
44. **throw** = throw exception, **throws** = declare
45. try-with-resources requires **AutoCloseable**

### Enum & Autoboxing (Session 7)
46. Enum constructor **implicitly private**
47. Every enum extends **java.lang.Enum<E>**
48. Enum **cannot extend** classes, **can implement** interfaces
49. Integer cache: **-128 to 127**
50. Use **Integer.valueOf()**, not ~~new Integer()~~
51. Boolean.parseBoolean() → only **"true"** (case-insensitive) = true
52. @Override **optional** but recommended
53. @FunctionalInterface → **exactly one** abstract method

### java.lang & java.util (Session 8)
54. **java.lang** auto-imported
55. Override **hashCode() when overriding equals()**
56. Equal objects → **same hashCode** (contract)
57. getClass() returns **runtime class**
58. Math.random() returns **[0.0, 1.0)**
59. Arrays.equals() for **value comparison**
60. Collections.sort() requires **List**

### Generics (Session 9-10)
61. Generics provide **compile-time type safety**
62. Type erasure at **runtime**
63. **Cannot create generic arrays**
64. PECS: **Producer Extends, Consumer Super**
65. `List<? extends T>` → **read-only**
66. `List<? super T>` → **write-allowed**

### Collections (Session 9-10)
67. HashMap allows **one null key**
68. TreeSet/TreeMap **don't allow null**
69. HashSet → **no order**
70. LinkedHashSet → **insertion order**
71. TreeSet → **sorted order**
72. ArrayList **fast access**, LinkedList **fast insert/delete**
73. Use **iterator.remove()** during iteration
74. TreeMap sorted by **keys**, not values
75. Hashtable **synchronized**, HashMap not
76. Vector **legacy and synchronized**
77. PriorityQueue **min-heap by default**

### Functional Programming (Session 11)
78. Predicate → **boolean**, Supplier → **value**, Consumer → **void**
79. Lambda accesses **effectively final** variables
80. Function → **one in, one out**
81. BinaryOperator → **two same type in, same type out**

### Streams (Session 12)
82. Streams **lazily evaluated**
83. Stream **cannot be reused**
84. filter, map, flatMap → **intermediate** (lazy)
85. collect, forEach, reduce → **terminal** (eager)

### Date/Time (Session 12)
86. New Date API classes **immutable** and **thread-safe**
87. **Period** for dates, **Duration** for time
88. LocalDate = **date only**, LocalTime = **time only**

### Concurrency (Session 13-14)
89. **start()** starts thread, **run()** doesn't
90. **Runnable** preferred over **Thread**
91. **synchronized** prevents race condition
92. Deadlock = **circular lock dependency**

### IO & Serialization (Session 15)
93. **transient** fields NOT serialized
94. **static** fields NOT serialized
95. Character streams: **16-bit Unicode**
96. Byte streams: **8-bit bytes**

### JVM & Reflection (Session 16)
97. JIT compiles **hot spots**
98. Reflection allows **runtime** inspection
99. ClassLoader: **Application → Extension → Bootstrap**

### JDBC (Session 17)
100. PreparedStatement **prevents SQL injection**
101. executeQuery() → **ResultSet**
102. executeUpdate() → **int** (row count)

---

## 🔥 Common MCQ Traps

### Integer Caching
```java
Integer a = 127, b = 127;  // a == b → true (cached)
Integer c = 128, d = 128;  // c == d → false (not cached)
```

### String Pool
```java
String s1 = "Hello";        // Pool
String s2 = "Hello";        // Same pool object
String s3 = new String("Hello");  // Heap
// s1 == s2 → true, s1 == s3 → false
```

### Unboxing Null
```java
Integer num = null;
int i = num;  // NullPointerException!
```

### Method Overloading
```java
// INVALID - only return type different
public int add(int a, int b) { }
public double add(int a, int b) { }  // ERROR
```

### Exception Order
```java
// INVALID - generic before specific
try { }
catch (Exception e) { }
catch (IOException e) { }  // Unreachable!
```

### Stream Reuse
```java
Stream<Integer> s = list.stream();
s.forEach(System.out::println);
s.forEach(System.out::println);  // IllegalStateException!
```

### Array Length
```java
int[] arr = {1, 2, 3};
arr.length();  // ERROR - length is property, not method
arr.length;    // Correct
```

---

## 📊 Data Structure Selection

| Need | Use | Time Complexity |
|------|-----|-----------------|
| Fast random access | ArrayList | O(1) get |
| Fast insert/delete at ends | LinkedList | O(1) |
| Unique elements, fast lookup | HashSet | O(1) |
| Unique + sorted | TreeSet | O(log n) |
| Unique + insertion order | LinkedHashSet | O(1) |
| Key-value, fast lookup | HashMap | O(1) |
| Sorted key-value | TreeMap | O(log n) |
| Thread-safe list | Vector / Synchronized | O(1) |
| Priority processing | PriorityQueue | O(log n) |

---

## 🎓 Comparison Tables

### String vs StringBuilder vs StringBuffer
| Feature | String | StringBuilder | StringBuffer |
|---------|--------|---------------|--------------|
| Mutability | Immutable | Mutable | Mutable |
| Thread-Safe | Yes | No | Yes |
| Performance | Slowest | Fastest | Slower |

### ArrayList vs LinkedList
| Operation | ArrayList | LinkedList |
|-----------|-----------|------------|
| Get | O(1) | O(n) |
| Add (end) | O(1) | O(1) |
| Add (middle) | O(n) | O(1) |
| Remove | O(n) | O(1) |

### HashMap vs TreeMap vs Hashtable
| Feature | HashMap | TreeMap | Hashtable |
|---------|---------|---------|-----------|
| Order | No | Sorted (key) | No |
| Null key | 1 allowed | Not allowed | Not allowed |
| Thread-safe | No | No | Yes |
| Performance | O(1) | O(log n) | O(1) |

### Checked vs Unchecked Exceptions
| Aspect | Checked | Unchecked |
|--------|---------|-----------|
| Extends | Exception (not Runtime) | RuntimeException/Error |
| Compile check | Yes | No |
| Example | IOException | NullPointerException |

---

## ⚡ Quick Syntax Reference

### Lambda Expressions
```java
// No parameters
() -> expression

// One parameter
x -> expression
(x) -> expression

// Multiple parameters
(x, y) -> expression
(x, y) -> { statements; }
```

### Streams
```java
list.stream()
    .filter(x -> condition)     // Intermediate
    .map(x -> transform)        // Intermediate
    .sorted()                   // Intermediate
    .collect(Collectors.toList());  // Terminal
```

### Try-with-resources
```java
try (ResourceType resource = new ResourceType()) {
    // Use resource
}  // Auto-closed
```

---

## 📚 Session-wise MCQ Count

- Sessions 1-2: 12 MCQs
- Session 3: 12 MCQs
- Session 4: 10 MCQs
- Session 5: 10 MCQs
- Session 6: 10 MCQs
- Session 7: 12 MCQs
- Session 8: 10 MCQs
- Session 9-10: 10 MCQs
- Session 11: 10 MCQs (Functional Programming)
- Session 12: 10 MCQs (Streams & Date/Time)
- Session 13-14: 10 MCQs (Concurrency)
- Session 15: 10 MCQs (IO & Serialization)
- Session 16: 10 MCQs (JVM & Reflection)
- Session 17: 10 MCQs (JDBC)

**Total: 126 MCQs**

---

## 🎯 Last-Minute Revision Checklist

- [ ] JVM architecture (Class Loader, Memory Areas)
- [ ] Stack vs Heap memory
- [ ] Integer caching (-128 to 127)
- [ ] String pool vs heap strings
- [ ] equals() and hashCode() contract
- [ ] Exception hierarchy and handling rules
- [ ] Collection framework hierarchy
- [ ] HashMap vs TreeMap vs LinkedHashMap
- [ ] ArrayList vs LinkedList performance
- [ ] Generic wildcards (? extends vs ? super)
- [ ] Lambda and functional interfaces
- [ ] Stream operations (intermediate vs terminal)
- [ ] Thread creation and synchronization
- [ ] Serialization (transient, static)
- [ ] JDBC PreparedStatement

---

**For detailed coverage, refer to individual session files.**

**Good luck with your exam! 🚀**
