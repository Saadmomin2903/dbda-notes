# One-Page Visual Cheat Sheet

**Print this page for quick last-minute revision!**

---

## 🏗️ Collection Framework Hierarchy

```
Collection (I)
├── List (I) - Ordered, allows duplicates
│   ├── ArrayList - Fast access O(1), slow insert O(n)
│   ├── LinkedList - Fast insert O(1), slow access O(n)
│   └── Vector - Legacy, synchronized
│
├── Set (I) - No duplicates
│   ├── HashSet - No order, O(1) operations
│   ├── LinkedHashSet - Insertion order, O(1) operations
│   └── TreeSet - Sorted, O(log n) operations, no null
│
└── Queue (I) - FIFO
    ├── PriorityQueue - Min-heap by default
    └── Deque (I) - Double-ended queue
        └── ArrayDeque

Map (I) - Key-value pairs
├── HashMap - O(1), one null key, multiple null values
├── LinkedHashMap - Insertion order
├── TreeMap - Sorted by keys, O(log n), no null keys
└── Hashtable - Legacy, synchronized, no null
```

---

## 🎭 Exception Hierarchy

```
Throwable
├── Error - Serious issues (OutOfMemoryError)
│   └── Don't catch these
│
└── Exception
    ├── RuntimeException (Unchecked)
    │   ├── NullPointerException
    │   ├── ArrayIndexOutOfBoundsException
    │   ├── ArithmeticException
    │   ├── IllegalArgumentException
    │   └── ClassCastException
    │
    └── Others (Checked - must handle)
        ├── IOException
        ├── SQLException
        ├── ClassNotFoundException
        └── InterruptedException
```

---

## 🧵 Thread Lifecycle

```
NEW → start() → RUNNABLE ⇄ RUNNING
                   ↓          ↓
              BLOCKED    WAITING/TIMED_WAITING
                              ↓
                        TERMINATED
```

---

## 📊 Quick Comparison Tables

### String vs StringBuilder vs StringBuffer
| Feature | String | StringBuilder | StringBuffer |
|---------|--------|---------------|--------------|
| Mutable | ❌ | ✅ | ✅ |
| Thread-safe | ✅ | ❌ | ✅ |
| Speed | Slowest | Fastest | Slower |

### ArrayList vs LinkedList
| Operation | ArrayList | LinkedList |
|-----------|-----------|------------|
| get(i) | O(1) ✅ | O(n) |
| add(end) | O(1) | O(1) |
| add(middle) | O(n) | O(1) ✅ |
| remove | O(n) | O(1) ✅ |

### HashMap vs TreeMap vs Hashtable
| Feature | HashMap | TreeMap | Hashtable |
|---------|---------|---------|-----------|
| Order | None | Sorted | None |
| null key | 1 | ❌ | ❌ |
| null value | ✅ | ✅ | ❌ |
| Thread-safe | ❌ | ❌ | ✅ |
| Speed | O(1) | O(log n) | O(1) |

### Checked vs Unchecked Exception
| Aspect | Checked | Unchecked |
|--------|---------|-----------|
| Extends | Exception | RuntimeException |
| Compile check | ✅ | ❌ |
| Must handle | ✅ | ❌ |
| Example | IOException | NullPointerException |

---

## 💾 Memory Areas

```
JVM MEMORY
├── Heap (Shared) - Objects, instance variables
├── Method Area (Shared) - Class metadata, static variables
├── Stack (Per-thread) - Local variables, method calls
└── PC Register (Per-thread) - Current instruction
```

**Stack:** Local variables, method frames  
**Heap:** All objects (`new`)  
**Method Area:** Static variables, class metadata

---

## 🔢 Value Ranges & Defaults

| Type | Size | Range | Default |
|------|------|-------|---------|
| byte | 1 | -128 to 127 | 0 |
| short | 2 | -32768 to 32767 | 0 |
| int | 4 | -2³¹ to 2³¹-1 | 0 |
| long | 8 | -2⁶³ to 2⁶³-1 | 0L |
| float | 4 | ~±3.4E38 | 0.0f |
| double | 8 | ~±1.7E308 | 0.0 |
| char | 2 | 0 to 65535 | '\u0000' |
| boolean | 1 bit | true/false | false |

**Integer Cache:** -128 to 127

---

## ⚡ Quick Syntax

### Lambda
```java
() -> expression
x -> expression
(x, y) -> { statements; }
```

### Stream Pipeline
```java
list.stream()
    .filter(condition)      // Intermediate
    .map(transformation)    // Intermediate
    .collect(toList());     // Terminal
```

### Try-with-resources
```java
try (Resource r = new Resource()) {
    // Use r
}  // Auto-closed
```

### Switch (Java 14+)
```java
String result = switch(value) {
    case 1 -> "One";
    case 2 -> "Two";
    default -> "Other";
};
```

---

## 🎯 Critical Exam Facts (Top 50)

**JVM & Memory**
1. ClassLoader: Bootstrap → Extension → Application
2. JIT compiles hot spots (frequently executed code)
3. Heap shared, Stack per-thread
4. static belongs to class, not instance

**Strings & Wrappers**
5. String immutable, StringBuilder mutable
6. Integer cache: -128 to 127
7. String pool in heap (not separate area)
8. Unboxing null → NullPointerException

**OOP**
9. Java: single inheritance (class), multiple (interface)
10. super() or this() must be FIRST in constructor
11. Cannot override: private, static, final
12. Abstract class can have constructor
13. Interface variables: public static final
14. Override: same or wider access

**Collections**
15. HashMap: one null key, many null values
16. TreeSet/TreeMap: no null, sorted
17. ArrayList: fast access, LinkedList: fast insert
18. ConcurrentModificationException: modify during iteration
19. Arrays.asList() → fixed-size list
20. Type erasure at runtime

**Generics**
21. PECS: Producer Extends, Consumer Super
22. List<? extends T>: read-only
23. Cannot create generic arrays

**Functional & Streams**
24. Predicate → boolean, Supplier → T, Consumer → void
25. Lambda: effectively final variables
26. Streams: lazy evaluation
27. Stream: single-use only
28. filter/map: intermediate, collect/forEach: terminal

**Exception**
29. Checked extends Exception (not RuntimeException)
30. finally: always (except System.exit())
31. finally return overwrites try return
32. Catch order: specific to generic

**Concurrency**
33. start() creates thread, run() doesn't
34. Runnable preferred over Thread
35. synchronized prevents race condition
36. wait() releases lock, sleep() doesn't
37. Daemon threads don't prevent JVM exit

**IO & Serialization**
38. transient & static: NOT serialized
39. Byte streams: 8-bit, Character: 16-bit
40. serialVersionUID: version compatibility
41. BufferedReader: efficient text reading

**JVM Internals**
42. Method Area → Metaspace (Java 8+)
43. Reflection: runtime inspection
44. setAccessible(true): bypass access control
45. System.gc(): request, not guarantee

**JDBC**
46. PreparedStatement: prevents SQL injection
47. Type 4 driver: pure Java (most common)
48. executeQuery() → ResultSet
49. executeUpdate() → int (row count)
50. ResultSet: 1-indexed (not 0)

---

## 🚨 Common Traps

```java
Integer a = 128, b = 128;  // a == b → false (not cached)
String s = "Hi"; s.toLowerCase();  // s still "Hi" (immutable)
List<? extends Number> list;  // Cannot add (read-only)
Stream s = ...; s.count(); s.count();  // Error (single-use)
```

---

**📌 Pin this page for exam day!**
