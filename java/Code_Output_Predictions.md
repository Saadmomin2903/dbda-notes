# Code Output Prediction Problems

**Total Problems:** 30  
**Skill:** Predicting program output  
**Difficulty:** Easy to Tricky

> ⚠️ **Common in exams!** Practice predicting output WITHOUT running code.

---

## Instructions
1. Read each code snippet carefully
2. Write the predicted output or error
3. Check answer at the end
4. Understand WHY if you got it wrong

---

## PROBLEM 1: Post vs Pre Increment
```java
int x = 10;
System.out.println(x++);
System.out.println(++x);
System.out.println(x);
```
**Your Answer:** ____________

---

## PROBLEM 2: String Pool
```java
String s1 = "Java";
String s2 = "Java";
String s3 = new String("Java");
System.out.println(s1 == s2);
System.out.println(s1 == s3);
System.out.println(s1.equals(s3));
```
**Your Answer:** ____________

---

## PROBLEM 3: Integer Caching
```java
Integer a = 127;
Integer b = 127;
Integer c = 128;
Integer d = 128;
System.out.println(a == b);
System.out.println(c == d);
```
**Your Answer:** ____________

---

## PROBLEM 4: String Immutability
```java
String s = "Hello";
s.concat(" World");
System.out.println(s);
```
**Your Answer:** ____________

---

## PROBLEM 5: final static method keyword
```java
class Parent {
    static void display() {
        System.out.println("Parent");
    }
}
class Child extends Parent {
    static void display() {
        System.out.println("Child");
    }
}
Parent obj = new Child();
obj.display();
```
**Your Answer:** ____________

---

## PROBLEM 6: Polymorphism
```java
class Animal {
    void sound() { System.out.println("Animal sound"); }
}
class Dog extends Animal {
    void sound() { System.out.println("Bark"); }
}
Animal a = new Dog();
a.sound();
```
**Your Answer:** ____________

---

## PROBLEM 7: Constructor Chaining
```java
class Test {
    Test() {
        this(10);
        System.out.println("No-arg");
    }
    Test(int x) {
        System.out.println("Parameterized: " + x);
    }
}
Test t = new Test();
```
**Your Answer:** ____________

---

## PROBLEM 8: finally with return
```java
public static int test() {
    try {
        return 1;
    } finally {
        return 2;
    }
}
System.out.println(test());
```
**Your Answer:** ____________

---

## PROBLEM 9: Exception Handling
```java
try {
    int x = 10 / 0;
} catch (ArithmeticException e) {
    System.out.println("Caught");
} finally {
    System.out.println("Finally");
}
System.out.println("End");
```
**Your Answer:** ____________

---

## PROBLEM 10: Array Length
```java
int[] arr = {1, 2, 3};
System.out.println(arr.length);
arr = new int[5];
System.out.println(arr.length);
```
**Your Answer:** ____________

---

## PROBLEM 11: ArrayList Modification
```java
List<String> list = Arrays.asList("A", "B", "C");
list.add("D");
```
**Your Answer:** ____________

---

## PROBLEM 12: HashMap null
```java
Map<String, String> map = new HashMap<>();
map.put(null, "value1");
map.put(null, "value2");
map.put("key", null);
System.out.println(map.size());
```
**Your Answer:** ____________

---

## PROBLEM 13: TreeSet Ordering
```java
Set<Integer> set = new TreeSet<>();
set.add(5);
set.add(1);
set.add(3);
System.out.println(set);
```
**Your Answer:** ____________

---

## PROBLEM 14: Type Erasure
```java
List<String> list1 = new ArrayList<>();
List<Integer> list2 = new ArrayList<>();
System.out.println(list1.getClass() == list2.getClass());
```
**Your Answer:** ____________

---

## PROBLEM 15: Wrapper Unboxing
```java
Integer num = null;
int value = num;
```
**Your Answer:** ____________

---

## PROBLEM 16: Lambda Effectively Final
```java
int x = 10;
Consumer<Integer> c = n -> System.out.println(n + x);
x = 20;
c.accept(5);
```
**Your Answer:** ____________

---

## PROBLEM 17: Stream Lazy Evaluation
```java
List<Integer> list = Arrays.asList(1, 2, 3);
Stream<Integer> stream = list.stream()
    .filter(n -> {
        System.out.println("Filter: " + n);
        return n > 1;
    });
System.out.println("Stream created");
```
**Your Answer:** ____________

---

## PROBLEM 18: Stream Reuse
```java
Stream<Integer> stream = Stream.of(1, 2, 3);
stream.forEach(System.out::println);
stream.forEach(System.out::println);
```
**Your Answer:** ____________

---

## PROBLEM 19: Thread start vs run
```java
Thread t = new Thread(() -> System.out.println("Thread: " + Thread.currentThread().getName()));
t.run();
```
**Your Answer:** ____________

---

## PROBLEM 20: synchronized
```java
class Counter {
    private int count = 0;
    public void increment() {
        count++;
    }
}
// Two threads call increment() 1000 times each
// Final count = ?
```
**Your Answer:** ____________

---

## PROBLEM 21: Static vs Instance
```java
class Test {
    static int x = 10;
    int y = 20;
}
Test t1 = new Test();
Test t2 = new Test();
t1.x = 100;
t1.y = 200;
System.out.println(t2.x + " " + t2.y);
```
**Your Answer:** ____________

---

## PROBLEM 22: Method Overloading
```java
public void print(int x) {
    System.out.println("int");
}
public void print(Integer x) {
    System.out.println("Integer");
}
print(10);
```
**Your Answer:** ____________

---

## PROBLEM 23: Enum ordinal
```java
enum Day {
    MON, TUE, WED
}
System.out.println(Day.WED.ordinal());
```
**Your Answer:** ____________

---

## PROBLEM 24: Interface Default Method
```java
interface A {
    default void print() {
        System.out.println("A");
    }
}
interface B {
    default void print() {
        System.out.println("B");
    }
}
class C implements A, B {
    // What must C do?
}
```
**Your Answer:** ____________

---

## PROBLEM 25: Transient Serialization
```java
class Employee implements Serializable {
    String name = "Alice";
    transient int age = 25;
}
// Serialize then deserialize
// age value = ?
```
**Your Answer:** ____________

---

## PROBLEM 26: Collection Sort
```java
List<Integer> list = Arrays.asList(3, 1, 2);
Collections.sort(list);
System.out.println(list);
```
**Your Answer:** ____________

---

## PROBLEM 27: String concat in loop
```java
String s = "";
for (int i = 0; i < 3; i++) {
    s += i;
}
System.out.println(s);
```
**Your Answer:** ____________

---

## PROBLEM 28: Equality
```java
Integer a = Integer.valueOf(100);
Integer b = Integer.valueOf(100);
System.out.println(a == b);
```
**Your Answer:** ____________

---

## PROBLEM 29: Local Variable Initialization
```java
public void test() {
    int x;
    System.out.println(x);
}
```
**Your Answer:** ____________

---

## PROBLEM 30: Generic Wildcard
```java
List<? extends Number> list = new ArrayList<Integer>();
list.add(10);
```
**Your Answer:** ____________

---

---

# ANSWER KEY

## PROBLEM 1: Post vs Pre Increment
```
10
12
12
```
**Explanation:** `x++` uses current value (10) then increments to 11. `++x` increments 11 to 12 then uses it.

## PROBLEM 2: String Pool
```
true
false
true
```
**Explanation:** s1 and s2 reference same pool object. s3 is in heap (new). equals() compares values.

## PROBLEM 3: Integer Caching
```
true
false
```
**Explanation:** -128 to 127 cached. 127 == 127 (same object), 128 != 128 (different objects).

## PROBLEM 4: String Immutability
```
Hello
```
**Explanation:** String is immutable. concat() returns NEW string, doesn't modify original.

## PROBLEM 5: Static Method Hiding
```
Parent
```
**Explanation:** Static methods are NOT polymorphic (hidden, not overridden). Resolved at compile time based on reference type.

## PROBLEM 6: Polymorphism
```
Bark
```
**Explanation:** Runtime polymorphism. Actual object is Dog, so Dog's sound() executes.

## PROBLEM 7: Constructor Chaining
```
Parameterized: 10
No-arg
```
**Explanation:** No-arg constructor calls parameterized first using this(10).

## PROBLEM 8: finally with return
```
2
```
**Explanation:** finally's return overwrites try's return.

## PROBLEM 9: Exception Handling
```
Caught
Finally
End
```
**Explanation:** Exception caught, finally executes, program continues.

## PROBLEM 10: Array Length
```
3
5
```
**Explanation:** First array has 3 elements, new array has 5 (all 0).

## PROBLEM 11: ArrayList Modification
```
UnsupportedOperationException
```
**Explanation:** Arrays.asList() returns fixed-size list. Cannot add/remove.

## PROBLEM 12: HashMap null
```
2
```
**Explanation:** HashMap allows one null key (second put overwrites first) and multiple null values.

## PROBLEM 13: TreeSet Ordering
```
[1, 3, 5]
```
**Explanation:** TreeSet maintains sorted order.

## PROBLEM 14: Type Erasure
```
true
```
**Explanation:** Type erasure at runtime. Both are just ArrayList at runtime.

## PROBLEM 15: Wrapper Unboxing
```
NullPointerException
```
**Explanation:** Cannot unbox null wrapper (null.intValue()).

## PROBLEM 16: Lambda Effectively Final
```
Compile error
```
**Explanation:** x is not effectively final (modified after lambda). Compile error.

## PROBLEM 17: Stream Lazy Evaluation
```
Stream created
```
**Explanation:** Intermediate operations (filter) are lazy. Nothing executes without terminal operation.

## PROBLEM 18: Stream Reuse
```
1
2
3
IllegalStateException
```
**Explanation:** Stream can only be used once. Second forEach() throws exception.

## PROBLEM 19: Thread start vs run
```
Thread: main
```
**Explanation:** run() executes in current thread (main), doesn't create new thread.

## PROBLEM 20: synchronized
```
Unpredictable (likely less than 2000)
```
**Explanation:** Without synchronized, race condition occurs. Likely lose some increments.

## PROBLEM 21: Static vs Instance
```
100 20
```
**Explanation:** Static variable shared (x=100 for all). Instance variable separate (y=20 for t2).

## PROBLEM 22: Method Overloading
```
int
```
**Explanation:** Primitive preferred over autoboxing for overloading.

## PROBLEM 23: Enum ordinal
```
2
```
**Explanation:** ordinal() is 0-indexed. MON=0, TUE=1, WED=2.

## PROBLEM 24: Interface Default Method
```
Must override print() to resolve conflict
```
**Explanation:** Diamond problem. Class must explicitly choose which default method or provide own.

## PROBLEM 25: Transient Serialization
```
0
```
**Explanation:** transient fields NOT serialized. int defaults to 0 after deserialization.

## PROBLEM 26: Collection Sort
```
[1, 2, 3]
```
**Explanation:** Collections.sort() sorts in ascending order (natural ordering).

## PROBLEM 27: String concat in loop
```
012
```
**Explanation:** String concatenation in loop creates "0", "01", "012".

## PROBLEM 28: Equality
```
true
```
**Explanation:** 100 is in cache range (-128 to 127). valueOf() returns cached object.

## PROBLEM 29: Local Variable Initialization
```
Compile error
```
**Explanation:** Local variables must be explicitly initialized.

## PROBLEM 30: Generic Wildcard
```
Compile error
```
**Explanation:** Cannot add to `List<? extends Number>` (read-only).

---

## Scoring
- 27-30: Excellent! Strong understanding
- 23-26: Very Good
- 19-22: Good, review some concepts
- 15-18: Average, more practice needed
- <15: Need significant review

**Your Score:** ___/30

---

**End of Code Output Prediction Problems**
