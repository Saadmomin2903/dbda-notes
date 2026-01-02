# Session 8 – java.lang & java.util

**Topics Covered:** Important Classes from java.lang, Collections Overview from java.util, Data Structure Selection Logic

---

## 1. java.lang Package

### Why java.lang is Special?
The **only package** that is **automatically imported** in every Java program.

```java
// No need to import
String s = "Hello";  // java.lang.String
System.out.println();  // java.lang.System
Integer i = 10;       // java.lang.Integer
```

---

## 2. Object Class (Root of Hierarchy)

Every class in Java **implicitly extends Object**.

```java
// These are equivalent
class MyClass { }
class MyClass extends Object { }
```

### Object Class Methods

| Method | Description | Override? |
|--------|-------------|-----------|
| `toString()` | String representation | Usually YES |
| `equals(Object)` | Logical equality | Usually YES |
| `hashCode()` | Hash code for collections | Must if equals() overridden |
| `clone()` | Creates copy | Rarely |
| `finalize()` | Called by GC (deprecated) | NO |
| `getClass()` | Runtime class | NO (final) |
| `notify()` | Thread notification | NO |
| `notifyAll()` | Thread notification | NO |
| `wait()` | Thread waiting | NO |

---

## 3. toString() Method

### Default Implementation

```java
class Person {
    String name;
    int age;
}

Person p = new Person();
p.name = "Alice";
p.age = 25;

System.out.println(p);  // Person@15db9742 (ClassName@HashCode)
System.out.println(p.toString());  // Same as above
```

### Override toString()

```java
class Person {
    String name;
    int age;
    
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
}

Person p = new Person();
p.name = "Alice";
p.age = 25;

System.out.println(p);  // Person{name='Alice', age=25}
```

⭐ **Exam Fact:** `println(object)` internally calls `object.toString()`.

---

## 4. equals() Method

### Default Implementation (Object class)

```java
public boolean equals(Object obj) {
    return (this == obj);  // Reference equality
}
```

### Override equals()

```java
class Person {
    String name;
    int age;
    
    @Override
    public boolean equals(Object obj) {
        // 1. Check if same object
        if (this == obj) return true;
        
        // 2. Check if null
        if (obj == null) return false;
        
        // 3. Check if same class
        if (getClass() != obj.getClass()) return false;
        
        // 4. Cast and compare fields
        Person other = (Person) obj;
        return age == other.age && 
               (name == null ? other.name == null : name.equals(other.name));
    }
}

Person p1 = new Person();
p1.name = "Alice";
p1.age = 25;

Person p2 = new Person();
p2.name = "Alice";
p2.age = 25;

System.out.println(p1 == p2);        // false (different objects)
System.out.println(p1.equals(p2));   // true (same values)
```

### equals() Contract (Must Follow)

1. **Reflexive:** `x.equals(x)` must be true
2. **Symmetric:** If `x.equals(y)`, then `y.equals(x)`
3. **Transitive:** If `x.equals(y)` and `y.equals(z)`, then `x.equals(z)`
4. **Consistent:** Multiple calls return same result
5. **Null:** `x.equals(null)` must be false

---

## 5. hashCode() Method

### What is hashCode()?
Returns an **integer hash code** used by hash-based collections (HashMap, HashSet).

### hashCode() Contract

**If two objects are equal (equals() returns true), they MUST have same hashCode().**

```java
class Person {
    String name;
    int age;
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Person other = (Person) obj;
        return age == other.age && name.equals(other.name);
    }
    
    @Override
    public int hashCode() {
        // Generate hash based on fields used in equals()
        return Objects.hash(name, age);  // Java 7+ utility
    }
}
```

### hashCode() Rules

| Rule | Description |
|------|-------------|
| **Consistency** | Multiple calls on same object → same hash code |
| **Equal objects** | If `a.equals(b)` → `a.hashCode() == b.hashCode()` |
| **Not required** | Different objects MAY have same hash code (collision) |

⚠️ **Common Mistake:**
```java
// Wrong: Override equals() but not hashCode()
class Person {
    String name;
    
    @Override
    public boolean equals(Object obj) {
        Person other = (Person) obj;
        return name.equals(other.name);
    }
    // Missing hashCode() override!
}

// Problem in HashMap
Person p1 = new Person();
p1.name = "Alice";

Person p2 = new Person();
p2.name = "Alice";

Map<Person, String> map = new HashMap<>();
map.put(p1, "Value1");
System.out.println(map.get(p2));  // null (different hash codes!)
```

⭐ **Exam Fact:** **Always override hashCode() when overriding equals()**.

---

## 6. getClass() Method

```java
class Animal { }
class Dog extends Animal { }

Animal a = new Dog();

System.out.println(a.getClass().getName());        // Dog (runtime class)
System.out.println(a.getClass().getSimpleName());  // Dog
System.out.println(a instanceof Animal);            // true
System.out.println(a instanceof Dog);               // true
```

⭐ **Exam Fact:** `getClass()` returns **runtime class**, not compile-time reference type.

---

## 7. clone() Method

### Shallow Copy

```java
class Person implements Cloneable {
    String name;
    int age;
    
    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();  // Shallow copy
    }
}

Person p1 = new Person();
p1.name = "Alice";
p1.age = 25;

Person p2 = (Person) p1.clone();
System.out.println(p1 == p2);        // false (different objects)
System.out.println(p1.equals(p2));   // depends on equals() implementation
```

⚠️ **Must implement Cloneable** interface, otherwise CloneNotSupportedException.

---

## 8. String Class

### String Characteristics

| Feature | Value |
|---------|-------|
| **Immutable** | Cannot be modified after creation |
| **Final** | Cannot be subclassed |
| **Thread-safe** | Immutability guarantees safety |

### Important String Methods (Review from Session 4)

```java
String s = "Hello World";

// Length & Character access
s.length();                    // 11
s.charAt(0);                   // 'H'
s.indexOf('o');                // 4
s.lastIndexOf('o');            // 7

// Substring
s.substring(0, 5);             // "Hello"
s.substring(6);                // "World"

// Case
s.toUpperCase();               // "HELLO WORLD"
s.toLowerCase();               // "hello world"

// Trim & Replace
s.trim();                      // Removes leading/trailing whitespace
s.replace("World", "Java");    // "Hello Java"

// Split
String[] words = s.split(" "); // ["Hello", "World"]

// Comparison
s.equals("Hello World");       // true
s.equalsIgnoreCase("hello world");  // true
s.compareTo("Hello");          // positive

// Check
s.isEmpty();                   // false
s.isBlank();                   // false (Java 11+)
s.contains("World");           // true
s.startsWith("Hello");         // true
s.endsWith("World");           // true
```

---

## 9. Math Class

All methods are **static**.

```java
// Constants
Math.PI;    // 3.141592653589793
Math.E;     // 2.718281828459045

// Basic operations
Math.abs(-5);           // 5
Math.max(10, 20);       // 20
Math.min(10, 20);       // 10

// Power & Root
Math.pow(2, 3);         // 8.0 (2³)
Math.sqrt(16);          // 4.0
Math.cbrt(27);          // 3.0 (cube root)

// Rounding
Math.ceil(4.3);         // 5.0 (round up)
Math.floor(4.7);        // 4.0 (round down)
Math.round(4.5);        // 5 (nearest int)
Math.round(4.4);        // 4

// Trigonometry (radians)
Math.sin(Math.PI / 2);  // 1.0
Math.cos(0);            // 1.0
Math.tan(Math.PI / 4);  // 1.0

// Random
Math.random();          // 0.0 (inclusive) to 1.0 (exclusive)

// Generate random int from 1 to 100
int rand = (int) (Math.random() * 100) + 1;
```

⭐ **Exam Fact:** Math.random() returns **[0.0, 1.0)** (0.0 inclusive, 1.0 exclusive).

---

## 10. System Class

```java
// Standard streams
System.out.println("Standard output");
System.err.println("Error output");

// Current time in milliseconds (since Jan 1, 1970 UTC)
long millis = System.currentTimeMillis();

// Nanoseconds (more precise)
long nanos = System.nanoTime();

// Environment variables
String path = System.getenv("PATH");
Map<String, String> env = System.getenv();

// System properties
String javaVersion = System.getProperty("java.version");
String osName = System.getProperty("os.name");
String userHome = System.getProperty("user.home");

// GC request
System.gc();  // Request garbage collection (not guaranteed)

// Exit JVM
System.exit(0);  // 0 = normal exit, non-zero = error

// Array copy
int[] src = {1, 2, 3, 4, 5};
int[] dest = new int[5];
System.arraycopy(src, 0, dest, 0, 5);  // Copy src to dest
```

---

## 11. Wrapper Classes (Review from Session 7)

See Session 7 for detailed coverage of autoboxing, caching, etc.

```java
// Parsing
int i = Integer.parseInt("123");
double d = Double.parseDouble("3.14");
boolean b = Boolean.parseBoolean("true");

// valueOf (uses cache for Integer, Long, etc.)
Integer obj = Integer.valueOf(100);

// toString
String s = Integer.toString(123);

// Comparison
Integer.compare(10, 20);  // -1
```

---

## 12. java.util Package Overview

### Main Categories

1. **Collections Framework** (List, Set, Map, Queue)
2. **Utility Classes** (Arrays, Collections, Objects)
3. **Date/Time** (Date, Calendar - legacy, prefer java.time in Java 8+)
4. **Random** (Random number generation)
5. **Scanner** (Reading input)
6. **StringTokenizer** (String parsing - legacy)

---

## 13. Arrays Utility Class

```java
import java.util.Arrays;

int[] arr = {5, 2, 8, 1, 9};

// Sort
Arrays.sort(arr);  // [1, 2, 5, 8, 9]

// Binary search (array must be sorted)
int index = Arrays.binarySearch(arr, 5);  // Returns index or negative

// Fill
Arrays.fill(arr, 0);  // Fill entire array with 0

// Copy
int[] copy = Arrays.copyOf(arr, arr.length);
int[] range = Arrays.copyOfRange(arr, 1, 4);  // Copy arr[1] to arr[3]

// Compare
int[] arr1 = {1, 2, 3};
int[] arr2 = {1, 2, 3};
boolean equal = Arrays.equals(arr1, arr2);  // true

// toString
System.out.println(Arrays.toString(arr));  // [1, 2, 3, 4, 5]

// Multi-dimensional
int[][] matrix = {{1, 2}, {3, 4}};
System.out.println(Arrays.deepToString(matrix));  // [[1, 2], [3, 4]]
```

⚠️ **Common Mistake:**
```java
int[] arr1 = {1, 2, 3};
int[] arr2 = {1, 2, 3};
System.out.println(arr1.equals(arr2));  // false (reference comparison!)
System.out.println(Arrays.equals(arr1, arr2));  // true (value comparison)
```

---

## 14. Collections Utility Class

```java
import java.util.*;

List<Integer> list = new ArrayList<>(Arrays.asList(5, 2, 8, 1, 9));

// Sort
Collections.sort(list);  // [1, 2, 5, 8, 9]

// Reverse
Collections.reverse(list);  // [9, 8, 5, 2, 1]

// Shuffle
Collections.shuffle(list);  // Random order

// Binary search (list must be sorted)
int index = Collections.binarySearch(list, 5);

// Min/Max
int min = Collections.min(list);
int max = Collections.max(list);

// Frequency
int freq = Collections.frequency(list, 5);  // Count occurrences

// Fill
Collections.fill(list, 0);  // Replace all elements with 0

// Copy
List<Integer> dest = new ArrayList<>(Collections.nCopies(list.size(), 0));
Collections.copy(dest, list);

// Singleton collections (immutable, single element)
Set<String> singleton = Collections.singleton("Only");
List<String> singletonList = Collections.singletonList("Only");

// Unmodifiable collections
List<Integer> unmodifiable = Collections.unmodifiableList(list);
// unmodifiable.add(10);  // UnsupportedOperationException

// Synchronized collections (thread-safe wrappers)
List<Integer> syncList = Collections.synchronizedList(list);
```

---

## 15. Objects Utility Class (Java 7+)

```java
import java.util.Objects;

String s1 = "Hello";
String s2 = null;

// Null-safe equals
Objects.equals(s1, s2);  // false (handles null)
// s1.equals(s2);        // NullPointerException if s1 is null

// requireNonNull (validation)
String s = Objects.requireNonNull(s1, "String cannot be null");

// hashCode (null-safe)
int hash = Objects.hash(s1, s2);  // Generate hash from multiple objects

// toString (null-safe)
String str = Objects.toString(s2, "default");  // Returns "default" if s2 is null
```

---

## 16. Scanner Class

```java
import java.util.Scanner;

Scanner sc = new Scanner(System.in);

// Read different types
int i = sc.nextInt();
double d = sc.nextDouble();
String word = sc.next();      // Read single word
String line = sc.nextLine();  // Read entire line
boolean b = sc.nextBoolean();

// Check if input available
if (sc.hasNextInt()) {
    int num = sc.nextInt();
}

// Close scanner
sc.close();
```

⚠️ **Common Trap:**
```java
Scanner sc = new Scanner(System.in);
int num = sc.nextInt();
String line = sc.nextLine();  // Empty! (reads newline after number)

// Fix: Consume newline
int num = sc.nextInt();
sc.nextLine();  // Consume newline
String line = sc.nextLine();  // Now reads actual line
```

---

## 17. Data Structure Selection Guide

### When to Use What?

| Need | Use | Reason |
|------|-----|--------|
| Fast random access by index | **ArrayList** | O(1) access |
| Fast insert/delete | **LinkedList** | O(1) at ends |
| Unique elements, fast lookup | **HashSet** | O(1) lookup, no duplicates |
| Unique + sorted elements | **TreeSet** | O(log n) lookup, sorted |
| Unique + insertion order | **LinkedHashSet** | Maintains order |
| Key-value pairs, fast lookup | **HashMap** | O(1) lookup |
| Sorted key-value pairs | **TreeMap** | O(log n), sorted by key |
| Thread-safe list | **Vector** or synchronized wrapper | Legacy |
| Priority-based processing | **PriorityQueue** | Min-heap by default |

### Time Complexity Comparison

| Operation | ArrayList | LinkedList | HashSet | TreeSet |
|-----------|-----------|------------|---------|---------|
| **Add** | O(1)* | O(1) | O(1) | O(log n) |
| **Remove** | O(n) | O(1)** | O(1) | O(log n) |
| **Get** | O(1) | O(n) | N/A | N/A |
| **Contains** | O(n) | O(n) | O(1) | O(log n) |

*Amortized, **At ends

---

## 🔥 Top MCQs for Session 8

### MCQ 1: java.lang Import
**Q:** Which package is auto-imported?
1. java.util
2. java.io
3. java.lang
4. java.net

**Answer:** 3. java.lang  
**Explanation:** java.lang is the only package automatically imported.

---

### MCQ 2: equals() and hashCode()
**Q:** If equals() returns true, hashCode() must:
1. Return different values
2. Return same value
3. Return 0
4. Throw exception

**Answer:** 2. Return same value  
**Explanation:** hashCode() contract: equal objects must have same hash code.

---

### MCQ 3: Math.random()
**Q:** Math.random() returns range:
1. 0.0 to 1.0 (both inclusive)
2. 0.0 (inclusive) to 1.0 (exclusive)
3. 0.0 to 100.0
4. 1.0 to 10.0

**Answer:** 2. 0.0 (inclusive) to 1.0 (exclusive)  
**Explanation:** Returns [0.0, 1.0) range.

---

### MCQ 4: getClass() vs instanceof
**Q:** What is the output?
```java
class Animal { }
class Dog extends Animal { }

Animal a = new Dog();
System.out.println(a.getClass().getName());
```
1. Animal
2. Dog
3. Object
4. Compile error

**Answer:** 2. Dog  
**Explanation:** getClass() returns runtime class (Dog), not reference type (Animal).

---

### MCQ 5: Array Comparison
**Q:** What is the output?
```java
int[] arr1 = {1, 2, 3};
int[] arr2 = {1, 2, 3};
System.out.println(arr1.equals(arr2));
```
1. true
2. false
3. Compile error

**Answer:** 2. false  
**Explanation:** equals() on array compares references. Use Arrays.equals() for value comparison.

---

### MCQ 6: toString() Default
**Q:** What does default toString() return?
1. Field values
2. ClassName@HashCode
3. Null
4. Empty string

**Answer:** 2. ClassName@HashCode  
**Explanation:** Object.toString() returns ClassName@HashCode format.

---

### MCQ 7: Collections.sort()
**Q:** Collections.sort() works on:
1. Array
2. List
3. Set
4. Map

**Answer:** 2. List  
**Explanation:** Collections.sort() requires List interface.

---

### MCQ 8: Math.round()
**Q:** What is Math.round(4.5)?
1. 4
2. 5
3. 4.0
4. 5.0

**Answer:** 2. 5  
**Explanation:** Rounds to nearest integer (ties round up).

---

### MCQ 9: System.gc()
**Q:** Does System.gc() guarantee garbage collection?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** gc() is a request, JVM may ignore it.

---

### MCQ 10: clone() Requirement
**Q:** To use clone(), class must implement:
1. Serializable
2. Cloneable
3. Comparable
4. Runnable

**Answer:** 2. Cloneable  
**Explanation:** Must implement Cloneable, otherwise CloneNotSupportedException.

---

## ⚠️ Common Mistakes

1. **Not overriding hashCode() with equals()**
2. **Using equals() on arrays** instead of Arrays.equals()
3. **Assuming System.gc() runs immediately**
4. **Math.random() range** misunderstanding
5. **Scanner nextLine() after nextInt()** without consuming newline
6. **Forgetting Cloneable** interface for clone()
7. **getClass() vs instanceof** confusion

---

## ⭐ One-liner Exam Facts

1. **java.lang** is auto-imported
2. Override **hashCode() when overriding equals()**
3. hashCode() contract: **equal objects → same hashCode**
4. getClass() returns **runtime class**
5. Math.random() returns **[0.0, 1.0)**
6. Math.round() rounds **.5** upward
7. System.gc() is a **request**, not guarantee
8. Arrays.equals() for **value comparison** of arrays
9. Collections.sort() requires **List** interface
10. Scanner.nextLine() after nextInt() **reads newline**
11. clone() requires **Cloneable** interface
12. Objects.equals() is **null-safe**

---

**End of Session 8**
