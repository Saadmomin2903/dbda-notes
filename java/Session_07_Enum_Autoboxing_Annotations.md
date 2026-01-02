# Session 7 – Enum, Autoboxing & Annotations

**Topics Covered:** Enum Internals, Autoboxing & Unboxing, Wrapper Method Behaviors, Annotations Basics

---

## 1. Enum Internals

### What is Enum?
A special Java class that represents a **group of named constants**.

### Why Enum?
Before enums, constants were defined using `public static final`:

```java
// Old way (error-prone)
public class Day {
    public static final int MONDAY = 1;
    public static final int TUESDAY = 2;
    public static final int WEDNESDAY = 3;
}

// Problem: Type safety
int day = 100;  // Valid but meaningless!
```

```java
// Enum way (type-safe)
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

// Type-safe
Day day = Day.MONDAY;
// Day day = 100;  // Compile error!
```

---

## 2. Enum Declaration & Usage

### Basic Enum

```java
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

// Usage
Day today = Day.MONDAY;
System.out.println(today);  // MONDAY

// Switch statement
switch (today) {
    case MONDAY:
        System.out.println("Start of week");
        break;
    case FRIDAY:
        System.out.println("End of week");
        break;
    default:
        System.out.println("Midweek");
}
```

---

## 3. Enum Built-in Methods

```java
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

public class EnumDemo {
    public static void main(String[] args) {
        Day today = Day.WEDNESDAY;
        
        // 1. values() - Returns array of all enum constants
        Day[] allDays = Day.values();
        for (Day d : allDays) {
            System.out.println(d);
        }
        
        // 2. valueOf(String) - Returns enum constant by name
        Day day = Day.valueOf("MONDAY");  // MONDAY
        // Day invalid = Day.valueOf("INVALID");  // IllegalArgumentException
        
        // 3. ordinal() - Returns position (0-indexed)
        System.out.println(Day.MONDAY.ordinal());     // 0
        System.out.println(Day.WEDNESDAY.ordinal());  // 2
        
        // 4. name() - Returns name as string
        System.out.println(Day.MONDAY.name());  // "MONDAY"
        
        // 5. compareTo() - Compares based on ordinal
        int result = Day.MONDAY.compareTo(Day.FRIDAY);  // Negative (0 < 4)
    }
}
```

### Method Summary Table

| Method | Return Type | Description | Example |
|--------|-------------|-------------|---------|
| `values()` | `EnumType[]` | Returns array of all constants | `Day.values()` |
| `valueOf(String)` | `EnumType` | Returns constant by name | `Day.valueOf("MONDAY")` |
| `ordinal()` | `int` | Returns position (0-indexed) | `Day.MONDAY.ordinal()` → 0 |
| `name()` | `String` | Returns constant name | `Day.MONDAY.name()` → "MONDAY" |
| `toString()` | `String` | Returns constant name (can override) | `Day.MONDAY.toString()` |
| `compareTo()` | `int` | Compares ordinals | `Day.MONDAY.compareTo(Day.FRIDAY)` |

⚠️ **Common MCQ Trap:**
```java
Day d = Day.valueOf("monday");  // IllegalArgumentException (case-sensitive!)
Day d = Day.valueOf("MONDAY");  // Correct
```

---

## 4. Enum with Fields, Methods & Constructors

### Enum with Fields

```java
enum Size {
    SMALL(10), MEDIUM(20), LARGE(30), XLARGE(40);
    
    private int value;  // Field
    
    // Constructor (MUST be private or package-private)
    private Size(int value) {
        this.value = value;
    }
    
    // Getter method
    public int getValue() {
        return value;
    }
}

// Usage
Size size = Size.MEDIUM;
System.out.println(size.getValue());  // 20
```

### Enum with Multiple Fields

```java
enum Pizza {
    SMALL(8, 300), MEDIUM(10, 450), LARGE(12, 600);
    
    private int diameter;    // in inches
    private int calories;
    
    private Pizza(int diameter, int calories) {
        this.diameter = diameter;
        this.calories = calories;
    }
    
    public int getDiameter() { return diameter; }
    public int getCalories() { return calories; }
    
    // Custom method
    public void displayInfo() {
        System.out.println(name() + ": " + diameter + "\" - " + calories + " cal");
    }
}

// Usage
Pizza pizza = Pizza.LARGE;
System.out.println(pizza.getDiameter());  // 12
System.out.println(pizza.getCalories());  // 600
pizza.displayInfo();  // LARGE: 12" - 600 cal
```

### Enum with Abstract Methods

```java
enum Operation {
    PLUS {
        public double apply(double x, double y) {
            return x + y;
        }
    },
    MINUS {
        public double apply(double x, double y) {
            return x - y;
        }
    },
    MULTIPLY {
        public double apply(double x, double y) {
            return x * y;
        }
    },
    DIVIDE {
        public double apply(double x, double y) {
            return x / y;
        }
    };
    
    // Abstract method (each constant MUST implement)
    public abstract double apply(double x, double y);
}

// Usage
double result = Operation.PLUS.apply(5, 3);      // 8.0
double result2 = Operation.MULTIPLY.apply(4, 7);  // 28.0
```

---

## 5. Enum Internals & Restrictions

### What Enum Actually Is

```java
// When you write:
enum Day {
    MONDAY, TUESDAY
}

// Compiler generates (simplified):
final class Day extends Enum<Day> {
    public static final Day MONDAY = new Day("MONDAY", 0);
    public static final Day TUESDAY = new Day("TUESDAY", 1);
    
    private Day(String name, int ordinal) {
        super(name, ordinal);
    }
    
    public static Day[] values() { ... }
    public static Day valueOf(String name) { ... }
}
```

### Enum Hierarchy

```
        java.lang.Object
               |
        java.lang.Enum<E>
               |
          Your Enum Class
```

### Enum Restrictions

| Restriction | Reason |
|-------------|--------|
| **Cannot extend other classes** | Already extends `java.lang.Enum` (single inheritance) |
| **Can implement interfaces** | Multiple interface implementation allowed |
| **Constructor is implicitly private** | Cannot instantiate enum from outside |
| **Cannot be instantiated with `new`** | Enum constants created by compiler |
| **Implicitly final** | Cannot be subclassed |
| **Cannot be abstract** | Because it's final |

```java
// ❌ INVALID
enum Day extends SomeClass {  // ERROR: Cannot extend
}

// ✅ VALID
interface Displayable {
    void display();
}

enum Day implements Displayable {
    MONDAY, TUESDAY;
    
    public void display() {
        System.out.println(this.name());
    }
}
```

⭐ **Exam Fact:** Enum constructor is **implicitly private**. Making it public/protected is a compile error.

```java
enum Size {
    SMALL(10);
    
    public Size(int value) { }  // ERROR: Modifier 'public' not allowed here
}
```

---

## 6. Enum Comparison

```java
enum Day {
    MONDAY, TUESDAY, WEDNESDAY
}

Day d1 = Day.MONDAY;
Day d2 = Day.MONDAY;
Day d3 = Day.WEDNESDAY;

// Using ==
System.out.println(d1 == d2);        // true (same instance)
System.out.println(d1 == d3);        // false

// Using equals()
System.out.println(d1.equals(d2));   // true
System.out.println(d1.equals(d3));   // false

// Using compareTo()
System.out.println(d1.compareTo(d2)); // 0 (same ordinal)
System.out.println(d1.compareTo(d3)); // -2 (0 - 2)
```

⭐ **Exam Fact:** For enums, `==` and `equals()` give same result (enum constants are singletons). But `==` is preferred (null-safe).

---

## 7. Autoboxing & Unboxing

### What is Autoboxing?
Automatic conversion of **primitive → wrapper object**.

### What is Unboxing?
Automatic conversion of **wrapper object → primitive**.

### Before Java 5 (Manual Boxing)

```java
// Manual boxing
Integer intObj = new Integer(10);  // Deprecated in Java 9+
Integer intObj = Integer.valueOf(10);

// Manual unboxing
int primitive = intObj.intValue();
```

### After Java 5 (Autoboxing/Unboxing)

```java
// Autoboxing (primitive → wrapper)
Integer intObj = 10;  // Compiler: Integer.valueOf(10)

// Unboxing (wrapper → primitive)
int primitive = intObj;  // Compiler: intObj.intValue()
```

---

## 8. Autoboxing Examples

### In Collections

```java
List<Integer> list = new ArrayList<>();

// Autoboxing
list.add(10);     // int → Integer
list.add(20);     // int → Integer

// Unboxing
int first = list.get(0);  // Integer → int
int total = 0;
for (Integer num : list) {
    total += num;  // Unboxing in each iteration
}
```

### In Method Calls

```java
public void printNumber(Integer num) {
    System.out.println(num);
}

printNumber(42);  // Autoboxing: int → Integer
```

### In Expressions

```java
Integer a = 10;
Integer b = 20;

// Unboxing for arithmetic
int sum = a + b;  // a.intValue() + b.intValue()

// Autoboxing result
Integer result = a + b;  // Unbox, add, then autobox
```

---

## 9. Wrapper Class Caching (CRITICAL for MCQs!)

### Integer Cache

Java caches Integer objects in range **-128 to 127**.

```java
Integer a = 127;
Integer b = 127;
System.out.println(a == b);  // true (cached, same object)

Integer c = 128;
Integer d = 128;
System.out.println(c == d);  // false (not cached, different objects)

// Always safe with equals()
System.out.println(c.equals(d));  // true
```

### Memory Diagram

```
Integer Cache (Method Area)
┌──────────────────────────┐
│ -128 → Integer(-128)     │
│ -127 → Integer(-127)     │
│  ...                     │
│  126 → Integer(126)      │
│  127 → Integer(127)      │ ← a and b point here
└──────────────────────────┘

Heap (non-cached)
┌──────────────────────────┐
│ Integer(128) ← c points  │
│ Integer(128) ← d points  │
└──────────────────────────┘
```

### Cache Ranges for All Wrappers

| Wrapper | Cache Range | Reason |
|---------|-------------|--------|
| **Integer** | -128 to 127 | Performance (common values) |
| **Byte** | -128 to 127 | All possible values |
| **Short** | -128 to 127 | Performance |
| **Long** | -128 to 127 | Performance |
| **Character** | 0 to 127 | ASCII range |
| **Boolean** | true, false | Both values cached |
| **Float** | None | Not cached |
| **Double** | None | Not cached |

⚠️ **Common MCQ Trap:**
```java
Integer a = 100;
Integer b = 100;
System.out.println(a == b);  // true (cached)

Integer c = new Integer(100);  // Force new object
Integer d = new Integer(100);
System.out.println(c == d);  // false (different objects)

Integer e = Integer.valueOf(100);
System.out.println(a == e);  // true (valueOf uses cache)
```

⭐ **Exam Fact:** Use `Integer.valueOf()` (uses cache) instead of `new Integer()` (deprecated, always creates new object).

---

## 10. Autoboxing/Unboxing Performance Issues

### Performance Impact

```java
// Inefficient (boxing/unboxing in loop)
Long sum = 0L;
for (long i = 0; i < 1000000; i++) {
    sum += i;  // Unbox sum, add, box result - 1 million times!
}

// Efficient (no boxing/unboxing)
long sum = 0L;
for (long i = 0; i < 1000000; i++) {
    sum += i;  // Pure primitive arithmetic
}
```

⭐ **Exam Fact:** Avoid autoboxing/unboxing in **tight loops** for performance.

---

## 11. NullPointerException in Unboxing

### Critical Trap

```java
Integer num = null;
int primitive = num;  // NullPointerException! (null.intValue())

// Safe check
Integer num = null;
if (num != null) {
    int primitive = num;
}
```

### Common Scenarios

```java
// NPE in comparison
Integer a = null;
if (a > 0) {  // NullPointerException (unboxing null)
    System.out.println("Positive");
}

// NPE in arithmetic
Integer x = null;
Integer y = 10;
int sum = x + y;  // NullPointerException

// NPE in collections
List<Integer> list = new ArrayList<>();
list.add(null);
int value = list.get(0);  // NullPointerException
```

⚠️ **Common MCQ Trap:**
```java
Integer a = null;
Integer b = 10;
System.out.println(a == b);  // false (no NPE, reference comparison)
System.out.println(a > b);   // NullPointerException (unboxing for comparison)
```

---

## 12. Wrapper Class Methods

### Integer Methods

```java
// Parsing
int num = Integer.parseInt("123");           // String → int
Integer obj = Integer.valueOf("123");        // String → Integer
Integer obj2 = Integer.valueOf(123);         // int → Integer (cached)

// Conversion
String s = Integer.toString(123);            // int → String
String binary = Integer.toBinaryString(10);  // "1010"
String hex = Integer.toHexString(255);       // "ff"
String octal = Integer.toOctalString(8);     // "10"

// Comparison
Integer.compare(10, 20);  // -1 (10 < 20)
Integer.compare(20, 10);  // 1 (20 > 10)
Integer.compare(10, 10);  // 0 (equal)

// Min/Max
System.out.println(Integer.MAX_VALUE);  // 2147483647
System.out.println(Integer.MIN_VALUE);  // -2147483648

// Checking
Integer.isFinite(10);  // N/A (only for Float/Double)
```

### Double/Float Methods

```java
// Special values
System.out.println(Double.POSITIVE_INFINITY);  // Infinity
System.out.println(Double.NEGATIVE_INFINITY);  // -Infinity
System.out.println(Double.NaN);                // NaN (Not a Number)

// Checking special values
Double.isInfinite(10.0 / 0);  // true
Double.isNaN(0.0 / 0);        // true
Double.isFinite(10.5);        // true

// Comparison
Double.compare(1.5, 2.5);     // -1
```

### Boolean Methods

```java
Boolean b = Boolean.valueOf("true");   // String → Boolean
boolean primitive = b.booleanValue();  // Boolean → boolean

Boolean.parseBoolean("true");   // true
Boolean.parseBoolean("TRUE");   // true (case-insensitive)
Boolean.parseBoolean("yes");    // false (only "true" is true)
```

⚠️ **Common Trap:**
```java
Boolean.parseBoolean("1");      // false (not "true")
Boolean.parseBoolean("yes");    // false (not "true")
Boolean.parseBoolean("True");   // true (case-insensitive)
```

---

## 13. Annotations

### What are Annotations?
Metadata that provides information about code **to compiler, runtime, or tools**.

### Syntax

```java
@AnnotationName
@AnnotationName(value = "something")
@AnnotationName(value1 = "x", value2 = 10)
```

---

## 14. Built-in Annotations

### @Override

```java
class Parent {
    void display() { }
}

class Child extends Parent {
    @Override
    void display() {  // Compiler checks if method actually overrides
        System.out.println("Child");
    }
    
    @Override
    void show() {  // ERROR: method does not override from superclass
    }
}
```

**Purpose:** Compile-time check to ensure method is actually overriding.

⭐ **Exam Fact:** @Override is **optional** but highly recommended.

---

### @Deprecated

```java
class OldAPI {
    @Deprecated
    public void oldMethod() {
        System.out.println("Old method");
    }
    
    @Deprecated(since = "2.0", forRemoval = true)
    public void veryOldMethod() {
        System.out.println("Very old");
    }
}

// Usage
OldAPI api = new OldAPI();
api.oldMethod();  // Warning: 'oldMethod()' is deprecated
```

**Purpose:** Marks code as outdated, compiler shows warning.

---

### @SuppressWarnings

```java
@SuppressWarnings("unchecked")
List list = new ArrayList();  // Raw type warning suppressed
list.add("String");

@SuppressWarnings({"unchecked", "deprecation"})
public void method() {
    // Multiple warnings suppressed
}

// Common values
@SuppressWarnings("unused")       // Unused variable
@SuppressWarnings("rawtypes")     // Raw type usage
@SuppressWarnings("all")          // All warnings
```

**Purpose:** Suppress compiler warnings.

---

### @FunctionalInterface (Java 8+)

```java
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);
    
    // int multiply(int a, int b);  // ERROR: Multiple abstract methods
}

// Valid (default/static methods allowed)
@FunctionalInterface
interface MyInterface {
    void method();  // Single abstract method
    
    default void defaultMethod() { }
    static void staticMethod() { }
}
```

**Purpose:** Marks interface as functional (single abstract method), compiler enforces.

⭐ **Exam Fact:** @FunctionalInterface enforces **exactly one abstract method** (default and static don't count).

---

### @SafeVarargs

```java
@SafeVarargs
public static <T> void printItems(T... items) {
    for (T item : items) {
        System.out.println(item);
    }
}
```

**Purpose:** Suppress warnings about varargs with generic types.

---

## 15. Custom Annotations

### Simple Annotation

```java
@interface MyAnnotation {
    // Annotation body
}

// Usage
@MyAnnotation
public class MyClass { }
```

### Annotation with Elements

```java
@interface RequestMapping {
    String value();           // Required
    String method() default "GET";  // Optional (has default)
    int timeout() default 5000;
}

// Usage
@RequestMapping(value = "/api/users")
@RequestMapping(value = "/api/data", method = "POST", timeout = 10000)
```

### Annotation with Single Value

```java
@interface Version {
    String value();
}

// Usage
@Version("1.0")  // Shorthand when only 'value' element
@Version(value = "1.0")  // Explicit
```

---

## 16. Meta-Annotations

Annotations that annotate other annotations.

### @Retention

Specifies how long annotation is retained.

```java
import java.lang.annotation.*;

@Retention(RetentionPolicy.RUNTIME)
@interface MyAnnotation { }
```

| Policy | Description |
|--------|-------------|
| `RetentionPolicy.SOURCE` | Discarded by compiler (e.g., @Override) |
| `RetentionPolicy.CLASS` | Stored in .class file, not available at runtime (default) |
| `RetentionPolicy.RUNTIME` | Available at runtime via reflection |

---

### @Target

Specifies where annotation can be applied.

```java
@Target(ElementType.METHOD)
@interface MethodOnly { }

@Target({ElementType.FIELD, ElementType.METHOD})
@interface FieldOrMethod { }
```

| Element Type | Usage |
|--------------|-------|
| `TYPE` | Class, interface, enum |
| `FIELD` | Field |
| `METHOD` | Method |
| `PARAMETER` | Parameter |
| `CONSTRUCTOR` | Constructor |
| `LOCAL_VARIABLE` | Local variable |
| `ANNOTATION_TYPE` | Annotation |
| `PACKAGE` | Package |

---

### @Documented

```java
@Documented
@interface Important { }
```

**Purpose:** Include in JavaDoc documentation.

---

### @Inherited

```java
@Inherited
@interface ParentAnnotation { }

@ParentAnnotation
class Parent { }

class Child extends Parent { }  // Inherits @ParentAnnotation
```

**Purpose:** Allow subclasses to inherit annotation.

---

## 🔥 Top MCQs for Session 7

### MCQ 1: Enum Constructor
**Q:** Can enum constructor be public?
```java
enum Size {
    SMALL(10);
    public Size(int value) { }
}
```
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** Enum constructor must be private or package-private (implicitly private).

---

### MCQ 2: Enum Extends
**Q:** Which class does every enum extend?
1. Object
2. Enum
3. Class
4. None

**Answer:** 2. Enum  
**Explanation:** Every enum extends java.lang.Enum<E> implicitly.

---

### MCQ 3: Integer Caching
**Q:** What is the output?
```java
Integer a = 127;
Integer b = 127;
Integer c = 128;
Integer d = 128;
System.out.println((a == b) + " " + (c == d));
```
1. true true
2. true false
3. false true
4. false false

**Answer:** 2. true false  
**Explanation:** 127 is cached (-128 to 127), 128 is not.

---

### MCQ 4: Unboxing NullPointerException
**Q:** What happens?
```java
Integer num = null;
int value = num;
```
1. value = 0
2. value = null
3. Compile error
4. NullPointerException

**Answer:** 4. NullPointerException  
**Explanation:** Unboxing null wrapper throws NPE (null.intValue()).

---

### MCQ 5: Autoboxing Performance
**Q:** Which is more efficient?
```java
// Option A
Long sum = 0L;
for (long i = 0; i < 1000; i++) sum += i;

// Option B
long sum = 0L;
for (long i = 0; i < 1000; i++) sum += i;
```
1. Option A
2. Option B
3. Same performance

**Answer:** 2. Option B  
**Explanation:** Option A has boxing/unboxing in loop, Option B uses pure primitives.

---

### MCQ 6: Wrapper Comparison
**Q:** What is the output?
```java
Integer a = new Integer(10);
Integer b = new Integer(10);
System.out.println(a == b);
```
1. true
2. false
3. Compile error
4. Runtime error

**Answer:** 2. false  
**Explanation:** `new` creates different objects. Always use equals() for value comparison.

---

### MCQ 7: Boolean Parsing
**Q:** What is the result?
```java
Boolean.parseBoolean("yes");
```
1. true
2. false
3. Compile error
4. Runtime error

**Answer:** 2. false  
**Explanation:** Only the string "true" (case-insensitive) returns true.

---

### MCQ 8: Enum ordinal()
**Q:** What is the output?
```java
enum Day {
    MONDAY, TUESDAY, WEDNESDAY
}
System.out.println(Day.WEDNESDAY.ordinal());
```
1. 0
2. 1
3. 2
4. 3

**Answer:** 3. 2  
**Explanation:** ordinal() returns 0-indexed position (MONDAY=0, TUESDAY=1, WEDNESDAY=2).

---

### MCQ 9: @Override
**Q:** Is @Override mandatory for overriding methods?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** @Override is optional but recommended for compile-time checking.

---

### MCQ 10: @FunctionalInterface
**Q:** How many abstract methods in functional interface?
```java
@FunctionalInterface
interface MyInterface {
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
**Explanation:** Only abstract methods count. Default and static methods don't count.

---

### MCQ 11: valueOf vs new
**Q:** Which uses Integer cache?
1. `new Integer(100)`
2. `Integer.valueOf(100)`
3. Both
4. Neither

**Answer:** 2. Integer.valueOf(100)  
**Explanation:** valueOf() uses cache, new always creates new object (deprecated).

---

### MCQ 12: Enum Implements
**Q:** Can enum implement interface?
```java
interface Printable {
    void print();
}

enum Day implements Printable {
    MONDAY;
    public void print() { }
}
```
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** Enum can implement interfaces (but cannot extend classes).

---

## ⚠️ Common Mistakes Summary

1. **Enum constructor public** → Must be private/package-private
2. **Integer caching** → Use equals(), not ==
3. **Unboxing null** → Always causes NullPointerException
4. **Autoboxing in loops** → Performance issue
5. **new Integer()** → Deprecated, use valueOf()
6. **Boolean.parseBoolean()** → Only "true" (case-insensitive) is true
7. **@Override optional** → But highly recommended
8. **Enum extends** → Cannot extend other classes
9. **== with wrappers** → Use equals() for value comparison
10. **valueOf case-sensitive** → Day.valueOf("monday") throws exception

---

## ⭐ One-liner Exam Facts

1. Enum constructor is **implicitly private**
2. Every enum extends **java.lang.Enum<E>**
3. Enum **cannot extend** other classes (single inheritance)
4. Enum **can implement** interfaces
5. Integer cache range: **-128 to 127**
6. Use **Integer.valueOf()**, not ~~new Integer()~~ (deprecated)
7. Unboxing **null wrapper** → **NullPointerException**
8. Autoboxing/unboxing in **loops** → performance hit
9. **==** compares references, **equals()** compares values
10. Boolean.parseBoolean() → only **"true"** (case-insensitive) is true
11. @Override is **optional** but recommended
12. @FunctionalInterface → **exactly one** abstract method
13. enum.ordinal() returns **0-indexed** position
14. enum.valueOf() is **case-sensitive**
15. Float and Double are **not cached**

---

## 📚 References

### Official Documentation
- Oracle Java Enum Documentation: https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html
- Autoboxing Documentation: https://docs.oracle.com/javase/tutorial/java/data/autoboxing.html
- Annotations Tutorial: https://docs.oracle.com/javase/tutorial/java/annotations/

---

**End of Session 7**
