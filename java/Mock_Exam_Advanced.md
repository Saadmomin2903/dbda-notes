# Mock Exam – Advanced (Tricky & Edge Cases)

**Total Questions:** 50  
**Difficulty:** Advanced - Edge Cases & Tricky Concepts  
**Time:** 75 minutes  
**Target:** Students aiming for 90%+

> ⚠️ **WARNING:** This exam focuses on subtle Java behaviors, edge cases, and common misconceptions. Every question is designed to test deep understanding.

---

## Section A: Tricky Fundamentals (1-12)

**Q1.** What is the output?
```java
int i = 1;
i = i++ + ++i;
System.out.println(i);
```
1. 3
2. 4
3. 5
4. Undefined behavior

**Q2.** What is the output?
```java
System.out.println(0.1 + 0.2 == 0.3);
```
1. true
2. false
3. Compile error
4. Depends on JVM

**Q3.** What happens?
```java
byte b = 127;
b++;
System.out.println(b);
```
1. 128
2. -128
3. Compile error
4. Runtime exception

**Q4.** What is the output?
```java
String s = new String("Hello");
s.intern();
System.out.println(s == "Hello");
```
1. true
2. false
3. Compile error
4. NullPointerException

**Q5.** What is the output?
```java
Integer a = 1000;
Integer b = 1000;
System.out.println(a == b);
System.out.println(a.equals(b));
```
1. true, true
2. false, true
3. true, false
4. false, false

**Q6.** What is the output?
```java
int[] arr1 = {1, 2, 3};
int[] arr2 = {1, 2, 3};
System.out.println(arr1 == arr2);
System.out.println(arr1.equals(arr2));
```
1. true, true
2. false, false
3. true, false
4. false, true

**Q7.** What is the output?
```java
String s1 = "a" + "b";
String s2 = "ab";
System.out.println(s1 == s2);
```
1. true (compile-time constant folding)
2. false
3. Compile error
4. Depends on JVM

**Q8.** What is the output?
```java
final String s1 = "a";
String s2 = s1 + "b";
String s3 = "ab";
System.out.println(s2 == s3);
```
1. true
2. false
3. Compile error
4. Runtime error

**Q9.** What is the output?
```java
System.out.println(10 / 0.0);
```
1. ArithmeticException
2. Infinity
3. NaN
4. Compile error

**Q10.** What is the output?
```java
System.out.println(0.0 / 0.0);
```
1. 0.0
2. Infinity
3. NaN
4. ArithmeticException

**Q11.** What is the output?
```java
char c = 'A';
c++;
System.out.println(c);
```
1. A
2. B
3. 66
4. Compile error

**Q12.** What is the output?
```java
boolean b = true;
if (b = false) {
    System.out.println("True");
} else {
    System.out.println("False");
}
```
1. True
2. False
3. Compile error
4. Nothing printed

---

## Section B: OOP Edge Cases (13-24)

**Q13.** What is the output?
```java
class Parent {
    static void display() { System.out.println("Parent"); }
}
class Child extends Parent {
    static void display() { System.out.println("Child"); }
}
Parent p = new Child();
p.display();
```
1. Parent (static method hiding, not overriding)
2. Child
3. Compile error
4. Runtime error

**Q14.** What happens?
```java
class Test {
    {
        System.out.println("Instance block");
    }
    static {
        System.out.println("Static block");
    }
    Test() {
        System.out.println("Constructor");
    }
}
Test t = new Test();
```
1. Static, Instance, Constructor
2. Instance, Static, Constructor
3. Constructor, Instance, Static
4. Static, Constructor, Instance

**Q15.** What is valid?
```java
class Outer {
    static class Inner {
        void display() {
            System.out.println(this);  // Line X
        }
    }
}
```
1. Line X prints Outer's this
2. Line X prints Inner's this
3. Compile error (static nested class)
4. Runtime error

**Q16.** What is the output?
```java
class A {
    A() {
        display();
    }
    void display() {
        System.out.println("A");
    }
}
class B extends A {
    int x = 10;
    void display() {
        System.out.println(x);
    }
}
B obj = new B();
```
1. 10
2. 0 (x not initialized when display() called from A's constructor)
3. A
4. Compile error

**Q17.** What happens?
```java
interface I {
    int x = 10;
}
class C implements I {
    public void test() {
        x = 20;  // Can we modify?
    }
}
```
1. Valid, x becomes 20
2. Compile error (interface variables are final)
3. Runtime error
4. x remains 10

**Q18.** What is the output?
```java
class Test {
    private Test() { }
    public static void main(String[] args) {
        Test t = new Test();
        System.out.println("Created");
    }
}
```
1. Compile error
2. Created (private constructor accessible in same class)
3. Runtime error
4. Nothing printed

**Q19.** Can this compile?
```java
abstract class A {
    abstract void method();
}
class B extends A {
    // No method() implementation
}
```
1. Yes, if B is also abstract
2. No, must implement method()
3. Yes, always valid
4. Depends on JVM

**Q20.** What is the output?
```java
class Parent {
    void method() throws IOException {
        System.out.println("Parent");
    }
}
class Child extends Parent {
    void method() throws Exception {  // Valid?
        System.out.println("Child");
    }
}
```
1. Compiles fine
2. Compile error (cannot throw broader exception)
3. Runtime error
4. Depends on usage

**Q21.** What is the output?
```java
class Test {
    int x;
    Test(int x) {
        this.x = x;
    }
}
Test t = new Test();  // Valid?
```
1. Valid, x = 0
2. Compile error (no no-arg constructor)
3. Runtime error
4. Valid, x = garbage value

**Q22.** What is the output?
```java
class A {
    final void show() {
        System.out.println("A");
    }
}
class B extends A {
    void show() {
        System.out.println("B");
    }
}
```
1. Compiles fine
2. Compile error (cannot override final)
3. Runtime error
4. B's show() hides A's

**Q23.** What is the output?
```java
Object o = new String("Hello");
System.out.println(o.length());  // Valid?
```
1. 5
2. Compile error (Object has no length())
3. Runtime error
4. ClassCastException

**Q24.** What is the output?
```java
String s = null;
System.out.println(s instanceof String);
```
1. true
2. false
3. NullPointerException
4. Compile error

---

## Section C: Collections & Generics Traps (25-36)

**Q25.** What is the output?
```java
List<Integer> list = new ArrayList<>();
list.add(1);
list.add(2);
list.add(3);
list.remove(1);  // What's removed?
System.out.println(list);
```
1. [2, 3] (removes value 1)
2. [1, 3] (removes index 1)
3. [1, 2] (removes last occurrence)
4. Compile error

**Q26.** What happens?
```java
List<String> list = new ArrayList<>();
list.add("A");
list.add(null);
list.add("B");
Collections.sort(list);
```
1. [null, A, B]
2. [A, B, null]
3. NullPointerException
4. Compile error

**Q27.** What is the output?
```java
Set<String> set = new HashSet<>();
set.add(null);
set.add(null);
System.out.println(set.size());
```
1. 0
2. 1 (Set allows one null)
3. 2
4. NullPointerException

**Q28.** What happens?
```java
TreeSet<String> set = new TreeSet<>();
set.add(null);
```
1. Valid, null added
2. NullPointerException (TreeSet doesn't allow null)
3. Compile error
4. null ignored

**Q29.** What is the output?
```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
map.put("A", 2);
System.out.println(map.size());
```
1. 1 (key overwritten)
2. 2
3. Compile error
4. Runtime error

**Q30.** Can this compile?
```java
List<int> list = new ArrayList<int>();
```
1. Yes
2. No (cannot use primitives with generics)
3. Only in Java 10+
4. Depends on compiler

**Q31.** What is valid?
```java
List<? extends Number> list = new ArrayList<Integer>();
// Which is valid?
```
1. list.add(10)
2. list.add(10.5)
3. Number n = list.get(0)
4. All invalid

**Q32.** What is the output?
```java
List<String> list1 = new ArrayList<>();
List<Integer> list2 = new ArrayList<>();
System.out.println(list1.getClass() == list2.getClass());
```
1. false
2. true (type erasure - both are ArrayList at runtime)
3. Compile error
4. ClassCastException

**Q33.** What happens?
```java
List<String> list = Arrays.asList("A", "B");
list.set(0, "C");  // Valid?
list.add("D");     // Valid?
```
1. Both valid
2. set valid, add throws exception
3. add valid, set throws exception
4. Both throw exception

**Q34.** What is the output?
```java
Integer[] arr = {1, 2, 3};
List<Integer> list = Arrays.asList(arr);
arr[0] = 100;
System.out.println(list.get(0));
```
1. 1
2. 100 (backed by original array)
3. Compile error
4. ConcurrentModificationException

**Q35.** What is the output?
```java
List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3));
for (Integer i : list) {
    if (i == 2) {
        list.remove(i);
    }
}
```
1. [1, 3]
2. ConcurrentModificationException
3. Infinite loop
4. Compile error

**Q36.** What is the output?
```java
Map<String, String> map = new HashMap<>();
map.put("A", "1");
System.out.println(map.get("B"));
```
1. ""
2. null
3. NullPointerException
4. NoSuchElementException

---

## Section D: Concurrency & Advanced (37-50)

**Q37.** What is guaranteed output?
```java
class Counter {
    int count = 0;
    void increment() {
        count++;
    }
}
// Two threads call increment() 1000 times each
```
1. 2000
2. Less than 2000 (race condition)
3. More than 2000
4. Compile error

**Q38.** What is the output?
```java
String s = "Hello";
synchronized(s) {
    s = "World";  // Valid?
}
```
1. Compiles fine (but synchronizing on mutable reference is bad practice)
2. Compile error
3. Runtime error
4. Deadlock

**Q39.** What happens?
```java
Thread t = new Thread();
t.start();
t.start();  // Call start() twice?
```
1. Starts twice
2. IllegalThreadStateException
3. Compile error
4. Does nothing second time

**Q40.** What is the output?
```java
try {
    return 1;
} finally {
    return 2;
}
```
1. 1
2. 2 (finally return overwrites)
3. Compile error
4. Runtime error

**Q41.** What happens?
```java
try {
    throw new IOException();
} catch (Exception e) {
    System.out.println("Caught");
} catch (IOException e) {
    System.out.println("IO Exception");
}
```
1. Caught
2. IO Exception
3. Compile error (unreachable catch)
4. Both printed

**Q42.** What is transient + final?
```java
class Test implements Serializable {
    transient final int x = 10;
}
// After serialization-deserialization, x = ?
```
1. 0
2. 10 (final fields are serialized even if transient)
3. Compile error
4. Runtime error

**Q43.** What is the output?
```java
String s = new String("Hello");
WeakReference<String> wr = new WeakReference<>(s);
s = null;
System.gc();
System.out.println(wr.get());
```
1. Hello
2. null (weak reference cleared)
3. Depends on GC
4. Runtime error

**Q44.** What is the output?
```java
Stream<Integer> s = Stream.of(1, 2, 3);
long count = s.filter(n -> n > 1).count();
long count2 = s.filter(n -> n > 2).count();
System.out.println(count + count2);
```
1. 3
2. IllegalStateException (stream already used)
3. 0
4. Compile error

**Q45.** What is the output?
```java
Optional<String> opt = Optional.of(null);
```
1. Empty Optional
2. Optional with null
3. NullPointerException
4. Compile error

**Q46.** What is the output?
```java
List<Integer> list = List.of(1, 2, 3);
list.add(4);
```
1. [1, 2, 3, 4]
2. UnsupportedOperationException (immutable list)
3. Compile error
4. NullPointerException

**Q47.** What is the output?
```java
int x = 5;
Supplier<Integer> s = () -> x;
x = 10;
System.out.println(s.get());
```
1. 5
2. 10
3. Compile error (x not effectively final)
4. Runtime error

**Q48.** What is the output?
```java
Connection conn = DriverManager.getConnection(url);
ResultSet rs = conn.createStatement()
    .executeQuery("SELECT * FROM users");
System.out.println(rs.getInt(0));  // Valid index?
```
1. First column value
2. SQLException (ResultSet is 1-indexed)
3. 0
4. Compile error

**Q49.** What is ClassCastException?
```java
Object obj = new Integer(10);
String s = (String) obj;
```
1. Compile error
2. ClassCastException at runtime
3. "10"
4. null

**Q50.** What is the output?
```java
System.out.println(Math.round(-1.5));
```
1. -2
2. -1
3. -2.0
4. Depends on mode

---

---

# ANSWER KEY

## Section A: Tricky Fundamentals

**Q1: B - 4**  
Explanation: `i = i++ + ++i` → i=1 initially, i++ returns 1 (then i=2), ++i returns 3 (i=3), 1+3=4

**Q2: B - false**  
Explanation: Floating-point precision issue. 0.1+0.2 = 0.30000000000000004, not exactly 0.3

**Q3: B - -128**  
Explanation: Byte overflow. 127 is max byte, 127+1 wraps to -128

**Q4: B - false**  
Explanation: intern() returns reference to pool, but doesn't change s. Need: s = s.intern()

**Q5: B - false, true**  
Explanation: 1000 not cached (outside -128 to 127), so == is false. equals() compares values (true)

**Q6: B - false, false**  
Explanation: Arrays don't override equals(), so both use reference comparison

**Q7: A - true**  
Explanation: Compiler concatenates string literals at compile-time, both in pool

**Q8: A - true**  
Explanation: final String s1 = "a" is compile-time constant, so s1+"b" → "ab" at compile-time (pool)

**Q9: B - Infinity**  
Explanation: Floating-point division by zero → Infinity (not exception)

**Q10: C - NaN**  
Explanation: 0.0/0.0 = NaN (Not a Number)

**Q11: B - B**  
Explanation: char can be incremented. 'A' (65) + 1 = 'B' (66)

**Q12: B - False**  
Explanation: Assignment b = false in condition, so condition is false

## Section B: OOP Edge Cases

**Q13: A - Parent**  
Explanation: Static methods are hidden, not overridden. Resolved at compile-time based on reference type

**Q14: A - Static, Instance, Constructor**  
Explanation: Static block when class loads, instance block before constructor, then constructor

**Q15: B - Prints Inner's this**  
Explanation: Static nested class has its own 'this', not Outer's

**Q16: B - 0**  
Explanation: display() called from A's constructor before B's instance variables initialized. x = 0 (default)

**Q17: B - Compile error**  
Explanation: Interface variables are public static final (cannot be modified)

**Q18: B - Created**  
Explanation: Private constructor accessible within same class

**Q19: A - Yes, if B is also abstract**  
Explanation: Abstract class can extend abstract class without implementing abstract methods (must be abstract itself)

**Q20: B - Compile error**  
Explanation: Overriding method cannot throw broader checked exception than parent

**Q21: B - Compile error**  
Explanation: No no-arg constructor available (parameterized constructor defined)

**Q22: B - Compile error**  
Explanation: Cannot override final method

**Q23: B - Compile error**  
Explanation: Object reference type has no length() method (compile-time check)

**Q24: B - false**  
Explanation: null instanceof anything is always false (not NullPointerException)

## Section C: Collections & Generics

**Q25: B - [1, 3]**  
Explanation: remove(int index) removes at index 1 (second element = 2)

**Q26: C - NullPointerException**  
Explanation: Collections.sort() cannot compare null

**Q27: B - 1**  
Explanation: HashSet allows one null (duplicate ignored)

**Q28: B - NullPointerException**  
Explanation: TreeSet requires comparison, cannot compare null

**Q29: A - 1**  
Explanation: Map keys unique, second put() overwrites first value

**Q30: B - No**  
Explanation: Generics require reference types, not primitives

**Q31: C - Number n = list.get(0)**  
Explanation: `? extends Number` is read-only (PECS: Producer Extends)

**Q32: B - true**  
Explanation: Type erasure - both are ArrayList at runtime

**Q33: B - set valid, add throws exception**  
Explanation: Arrays.asList() returns fixed-size list (can modify, cannot add/remove)

**Q34: B - 100**  
Explanation: Arrays.asList() backed by original array (changes reflect)

**Q35: B - ConcurrentModificationException**  
Explanation: Cannot modify collection during for-each iteration

**Q36: B - null**  
Explanation: HashMap.get() returns null for missing key (not exception)

## Section D: Concurrency & Advanced

**Q37: B - Less than 2000**  
Explanation: Race condition without synchronization (lost updates)

**Q38: A - Compiles fine**  
Explanation: Valid syntax, but bad practice (synchronizing on mutable reference)

**Q39: B - IllegalThreadStateException**  
Explanation: Cannot call start() twice on same thread

**Q40: B - 2**  
Explanation: finally block's return overwrites try's return

**Q41: C - Compile error**  
Explanation: Second catch (IOException) unreachable (already caught by Exception)

**Q42: B - 10**  
Explanation: final fields serialized even if transient (JVM ensures consistency)

**Q43: C - Depends on GC**  
Explanation: WeakReference may or may not be cleared (GC dependent)

**Q44: B - IllegalStateException**  
Explanation: Stream can only be used once

**Q45: C - NullPointerException**  
Explanation: Optional.of(null) throws NPE. Use Optional.ofNullable(null)

**Q46: B - UnsupportedOperationException**  
Explanation: List.of() creates immutable list

**Q47: C - Compile error**  
Explanation: Lambda requires effectively final variables (x is modified)

**Q48: B - SQLException**  
Explanation: ResultSet is 1-indexed (not 0-indexed like arrays)

**Q49: B - ClassCastException**  
Explanation: Runtime error - Integer cannot be cast to String

**Q50: B - -1**  
Explanation: Math.round(-1.5) = -1 (rounds towards positive infinity for .5)

---

## Scoring Guide

| Score | Level |
|-------|-------|
| 45-50 | Expert - Exceptional understanding |
| 40-44 | Advanced - Excellent |
| 35-39 | Good - Strong foundation |
| 30-34 | Average - Need more practice |
| <30 | Review fundamentals |

---

## Topic-wise Difficulty Analysis

| Topic | Questions | Common Pitfalls |
|-------|-----------|-----------------|
| Floating Point | 2, 9, 10 | Precision, NaN, Infinity |
| Overflow | 3 | Byte wraparound |
| String Pool | 4, 7, 8 | intern(), compile-time constants |
| Integer Cache | 1, 5 | -128 to 127 only |
| Static Hiding | 13 | Not polymorphic |
| Initialization | 14, 16 | Order, premature method calls |
| final/transient | 42 | Serialization edge case |
| Generics | 25, 30-35 | Type erasure, PECS |
| Collections | 26-29, 33 | null handling varies |
| Streams | 44 | Single-use only |
| Concurrency | 37-39 | Race conditions |

---

**End of Advanced Mock Exam**
