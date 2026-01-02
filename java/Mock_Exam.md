# Mock Exam – Java Programming PG-DBDA

**Total Questions:** 50  
**Time:** 60 minutes  
**Passing Score:** 70% (35/50)

---

## Instructions
1. Each question has 4 options, only ONE is correct
2. Mark your answers on a separate sheet
3. No negative marking
4. Use the answer key at the end to check your score

---

## Section A: Java Fundamentals (Questions 1-15)

**Q1.** Which of the following is NOT a valid JVM component?
1. Class Loader
2. Execution Engine
3. Compiler
4. Garbage Collector

**Q2.** What is the output?
```java
int x = 10;
System.out.println(x++);
System.out.println(++x);
```
1. 10, 11
2. 10, 12
3. 11, 12
4. 11, 11

**Q3.** Integer caching in Java works for which range?
1. -256 to 255
2. -128 to 127
3. 0 to 255
4. -100 to 100

**Q4.** What happens when you call `run()` instead of `start()` on a Thread?
1. Compile error
2. Runtime exception
3. Runs in current thread (no new thread created)
4. Same as start()

**Q5.** Which is true about String in Java?
1. Mutable
2. Thread-safe due to immutability
3. Extends StringBuffer
4. Stored only in heap

**Q6.** What is the output?
```java
String s1 = "Hello";
String s2 = "Hello";
String s3 = new String("Hello");
System.out.println(s1 == s2);
System.out.println(s1 == s3);
```
1. true, true
2. true, false
3. false, true
4. false, false

**Q7.** Which access modifier has the widest scope?
1. private
2. default (package-private)
3. protected
4. public

**Q8.** Method overloading is resolved at:
1. Compile time
2. Runtime
3. Load time
4. Both compile and runtime

**Q9.** Can an abstract class have a constructor?
1. Yes
2. No
3. Only if it has no abstract methods
4. Only private constructor

**Q10.** What is the output?
```java
Integer a = 128;
Integer b = 128;
System.out.println(a == b);
```
1. true
2. false
3. Compile error
4. Runtime exception

**Q11.** Which exception is checked?
1. NullPointerException
2. ArrayIndexOutOfBoundsException
3. IOException
4. ArithmeticException

**Q12.** finally block executes:
1. Only if no exception
2. Only if exception occurs
3. Always (except System.exit())
4. Never if return in try

**Q13.** Enum constructor must be:
1. public
2. protected
3. private or package-private
4. static

**Q14.** Which is NOT serialized?
1. Instance variables
2. transient variables
3. final variables
4. All are serialized

**Q15.** JVM memory area shared among ALL threads?
1. Stack
2. PC Register
3. Heap
4. Native Method Stack

---

## Section B: OOP & Collections (Questions 16-30)

**Q16.** Java supports multiple inheritance through:
1. Classes
2. Interfaces
3. Abstract classes
4. Not supported

**Q17.** What is the output?
```java
class Parent {
    void method() { System.out.println("Parent"); }
}
class Child extends Parent {
    void method() { System.out.println("Child"); }
}
Parent obj = new Child();
obj.method();
```
1. Parent
2. Child
3. Compile error
4. Runtime error

**Q18.** HashMap allows:
1. No null key or value
2. One null key, multiple null values
3. Multiple null keys
4. One null key and one null value

**Q19.** TreeSet does NOT allow:
1. Duplicates
2. null
3. Custom objects
4. Integers

**Q20.** Which gives O(1) random access?
1. LinkedList
2. ArrayList
3. TreeSet
4. LinkedHashSet

**Q21.** What is the output?
```java
List<Integer> list = Arrays.asList(1, 2, 3);
list.add(4);
```
1. [1, 2, 3, 4]
2. UnsupportedOperationException
3. Compile error
4. NullPointerException

**Q22.** PECS principle means:
1. Producer Extends, Consumer Super
2. Producer Equals, Consumer Same
3. Push Extends, Consumer Super
4. Pull Extends, Create Super

**Q23.** Type erasure occurs at:
1. Compile time
2. Runtime
3. Load time
4. Never

**Q24.** Which can store primitives directly?
1. ArrayList
2. HashMap
3. int[]
4. List<Integer>

**Q25.** ConcurrentModificationException occurs when:
1. Thread conflict
2. Modifying collection during iteration
3. Null value added
4. Memory full

**Q26.** LinkedHashSet maintains:
1. No order
2. Sorted order
3. Insertion order
4. Reverse order

**Q27.** Which is functional interface?
1. Comparable
2. Runnable
3. Serializable
4. Cloneable

**Q28.** Lambda can access:
1. Any local variable
2. Only final variables
3. Effectively final variables
4. Only instance variables

**Q29.** Predicate<T> returns:
1. T
2. void
3. boolean
4. Object

**Q30.** Which is terminal stream operation?
1. filter()
2. map()
3. sorted()
4. collect()

---

## Section C: Advanced Topics (Questions 31-45)

**Q31.** Streams are evaluated:
1. Eagerly
2. Lazily
3. Randomly
4. Sequentially only

**Q32.** Can a stream be reused?
1. Yes
2. No
3. Only parallel streams
4. Only sequential streams

**Q33.** New Date API classes are:
1. Mutable
2. Immutable and thread-safe
3. Thread-unsafe
4. Deprecated

**Q34.** Period is used for:
1. Time duration
2. Date duration
3. Both
4. Neither

**Q35.** synchronized prevents:
1. Compile errors
2. Race condition
3. Deadlock
4. Memory leak

**Q36.** Daemon threads:
1. Prevent JVM from exiting
2. Don't prevent JVM from exiting
3. Must be started first
4. Cannot be created

**Q37.** wait() vs sleep():
1. Same functionality
2. wait() releases lock, sleep() doesn't
3. sleep() releases lock, wait() doesn't
4. Both release lock

**Q38.** Character streams use:
1. 8-bit
2. 16-bit
3. 32-bit
4. Variable bit

**Q39.** serialVersionUID is used for:
1. Performance
2. Security
3. Version compatibility
4. Compression

**Q40.** JIT compiles:
1. All code
2. Hot spots (frequently executed)
3. Only main method
4. Static methods only

**Q41.** ClassLoader hierarchy (parent to child):
1. Application → Extension → Bootstrap
2. Bootstrap → Extension → Application
3. Extension → Bootstrap → Application
4. Bootstrap → Application → Extension

**Q42.** Reflection allows:
1. Compile-time checking
2. Runtime inspection/modification
3. Faster execution
4. Better security

**Q43.** PreparedStatement prevents:
1. Syntax errors
2. SQL injection
3. Connection timeout
4. Data loss

**Q44.** executeQuery() returns:
1. int
2. boolean
3. ResultSet
4. void

**Q45.** JDBC Type 4 driver is:
1. JDBC-ODBC bridge
2. Native API
3. Network protocol
4. Pure Java (thin driver)

---

## Section D: Tricky Scenarios (Questions 46-50)

**Q46.** What is the output?
```java
Integer num = null;
int value = num;
```
1. 0
2. null
3. Compile error
4. NullPointerException

**Q47.** What is the output?
```java
String s = "ABC";
s.toLowerCase();
System.out.println(s);
```
1. abc
2. ABC
3. Compile error
4. Runtime error

**Q48.** What is the output?
```java
try {
    System.out.println("Try");
    return;
} finally {
    System.out.println("Finally");
}
```
1. Try
2. Finally
3. Try Finally
4. Compile error

**Q49.** Which is valid?
```java
List<? extends Number> list = new ArrayList<Integer>();
```
1. list.add(10)
2. list.add(10.5)
3. Number n = list.get(0)
4. All invalid

**Q50.** What is the output?
```java
Stream.of(1, 2, 3)
      .filter(n -> n > 2)
      .count();
```
1. 0
2. 1
3. 2
4. 3

---

## ANSWER KEY

### Section A (1-15)
1. **C** - Compiler (not part of JVM, separate tool)
2. **B** - 10, 12 (x++ uses then increments; ++x increments then uses)
3. **B** - -128 to 127
4. **C** - Runs in current thread
5. **B** - Thread-safe due to immutability
6. **B** - true, false (pool vs heap)
7. **D** - public
8. **A** - Compile time
9. **A** - Yes
10. **B** - false (128 not cached)
11. **C** - IOException
12. **C** - Always (except System.exit())
13. **C** - private or package-private
14. **B** - transient variables
15. **C** - Heap

### Section B (16-30)
16. **B** - Interfaces
17. **B** - Child (runtime polymorphism)
18. **B** - One null key, multiple null values
19. **B** - null
20. **B** - ArrayList
21. **B** - UnsupportedOperationException
22. **A** - Producer Extends, Consumer Super
23. **B** - Runtime
24. **C** - int[] (array stores primitives)
25. **B** - Modifying collection during iteration
26. **C** - Insertion order
27. **B** - Runnable
28. **C** - Effectively final variables
29. **C** - boolean
30. **D** - collect()

### Section C (31-45)
31. **B** - Lazily
32. **B** - No
33. **B** - Immutable and thread-safe
34. **B** - Date duration
35. **B** - Race condition
36. **B** - Don't prevent JVM from exiting
37. **B** - wait() releases lock, sleep() doesn't
38. **B** - 16-bit
39. **C** - Version compatibility
40. **B** - Hot spots
41. **B** - Bootstrap → Extension → Application
42. **B** - Runtime inspection/modification
43. **B** - SQL injection
44. **C** - ResultSet
45. **D** - Pure Java (thin driver)

### Section D (46-50)
46. **D** - NullPointerException (unboxing null)
47. **B** - ABC (String is immutable, toLowerCase() returns new string)
48. **C** - Try Finally (finally always executes)
49. **C** - Number n = list.get(0) (can read, cannot write)
50. **B** - 1 (only 3 passes filter)

---

## Scoring Guide

| Score | Grade | Remarks |
|-------|-------|---------|
| 45-50 | A+ | Excellent! Exam ready |
| 40-44 | A | Very good understanding |
| 35-39 | B+ | Good, review weak areas |
| 30-34 | B | Pass, more practice needed |
| 25-29 | C | Borderline, significant review required |
| <25 | F | Need thorough revision |

---

## Topic-wise Analysis

Track your performance by topic:

| Topic | Questions | Score | % |
|-------|-----------|-------|---|
| JVM & Memory | 1, 15, 40, 41 | __/4 | __% |
| Strings & Wrappers | 5, 6, 10, 47 | __/4 | __% |
| OOP & Inheritance | 7, 8, 9, 16, 17 | __/5 | __% |
| Collections | 18-26 | __/9 | __% |
| Functional & Streams | 27-32, 50 | __/7 | __% |
| Concurrency | 4, 35-37 | __/4 | __% |
| IO & Serialization | 14, 38, 39 | __/3 | __% |
| JDBC & Reflection | 42-45 | __/4 | __% |
| Exceptions | 11, 12 | __/2 | __% |
| Tricky Scenarios | 2, 46, 48, 49 | __/4 | __% |

**Weak areas (< 70%):** _________________

---

**End of Mock Exam**
