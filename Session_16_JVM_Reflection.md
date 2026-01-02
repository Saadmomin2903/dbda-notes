# Session 16 – JVM Internals & Reflection

**Topics Covered:** JVM Architecture Deep Dive, Class Loaders, JIT Compiler, Garbage Collection, Reflection API, Dynamic Class Loading

---

## 1. JVM Architecture (Complete)

```
┌────────────────────────────────────────────────────────────┐
│                    JVM ARCHITECTURE                         │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │              CLASS LOADER SUBSYSTEM                   │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  1. Bootstrap ClassLoader (loads rt.jar)             │  │
│  │  2. Extension ClassLoader (loads ext/)               │  │
│  │  3. Application ClassLoader (loads classpath)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           RUNTIME DATA AREAS                          │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Heap (shared)       - Objects                     │  │
│  │  • Method Area (shared) - Class metadata, statics    │  │
│  │  • Stack (per-thread)  - Local vars, method calls    │  │
│  │  • PC Register (per-thread) - Current instruction    │  │
│  │  • Native Method Stack - Native method data          │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            EXECUTION ENGINE                           │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Interpreter - Executes bytecode line-by-line      │  │
│  │  • JIT Compiler - Compiles hot spots to native       │  │
│  │  • Garbage Collector - Reclaims memory               │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Class Loader Subsystem

### ClassLoader Hierarchy

```
Bootstrap ClassLoader (C/C++)
         ├── Loads: rt.jar, core Java classes
         └── Parent: null
              ↑
Extension ClassLoader (Java)
         ├── Loads: jre/lib/ext/
         └── Parent: Bootstrap
              ↑
Application ClassLoader (Java)
         ├── Loads: CLASSPATH
         └── Parent: Extension
```

### Delegation Model

```java
// Example class loading
class MyClass { }

// Steps:
// 1. Application ClassLoader checks cache → not found
// 2. Delegates to Extension → not found
// 3. Extension delegates to Bootstrap → not found
// 4. Bootstrap tries to load → fails (not core class)
// 5. Extension tries to load → fails (not in ext/)
// 6. Application tries to load → SUCCESS (in classpath)
```

⭐ **Exam Fact:** ClassLoader follows **parent delegation** model (always asks parent first).

### ClassLoader Methods

```java
// Get classloader
ClassLoader cl = MyClass.class.getClassLoader();
System.out.println(cl);  // sun.misc.Launcher$AppClassLoader

// Get parent
ClassLoader parent = cl.getParent();  // ExtClassLoader
ClassLoader grandParent = parent.getParent();  // null (Bootstrap)

// Load class dynamically
Class<?> clazz = Class.forName("com.example.MyClass");
```

---

## 3. Runtime Data Areas

### Heap (Shared Among Threads)

```
┌─────────────────────────────┐
│          HEAP               │
├─────────────────────────────┤
│  Young Generation           │
│  ├── Eden Space             │
│  ├── Survivor Space 0 (S0)  │
│  └── Survivor Space 1 (S1)  │
│                             │
│  Old Generation (Tenured)   │
└─────────────────────────────┘
```

**Stores:** All objects and instance variables.

### Method Area / Metaspace (Shared)

**Stores:**
- Class metadata
- Static variables
- Constant pool
- Method bytecode

⭐ **Exam Fact:** In Java 8+, **PermGen replaced by Metaspace** (native memory, not heap).

### Stack (Per-Thread)

```
Thread Stack
┌──────────────┐
│ Frame 3      │ ← method3() call
├──────────────┤
│ Frame 2      │ ← method2() call
├──────────────┤
│ Frame 1      │ ← method1() call
├──────────────┤
│ Main Frame   │ ← main() method
└──────────────┘
```

**Each Frame Contains:**
- Local variables
- Operand stack
- Reference to runtime constant pool

### PC Register (Per-Thread)

Stores address of current instruction being executed.

---

## 4. JIT Compiler

### What is JIT?
**Just-In-Time Compiler** = Compiles frequently executed bytecode (**hot spots**) to native machine code for better performance.

### How It Works

```
Bytecode → Interpreter (initially)
    ↓
Execution profiling (JVM monitors)
    ↓
Hot spot detected (method called 1000+ times)
    ↓
JIT compiles to native code
    ↓
Future calls execute native code (MUCH faster)
```

### JIT Compiler Types

1. **C1 Compiler** (Client) - Fast compilation, less optimization
2. **C2 Compiler** (Server) - Slow compilation, aggressive optimization
3. **Tiered Compilation** - Uses both (C1 first, then C2 for hot spots)

⭐ **Exam Fact:** JIT compiles **hot spots** (frequently executed code), not all code.

---

## 5. Garbage Collection

### How GC Works

```
1. Mark Phase: Mark all reachable objects
2. Sweep Phase: Delete unreachable objects
3. Compact Phase (optional): Compact memory
```

### GC Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Serial GC** | Single thread, small apps |
| **Parallel GC** | Multiple threads (throughput) |
| **CMS** | Concurrent Mark Sweep (low pause) |
| **G1 GC** | Garbage First (default Java  9+) |
| **ZGC** | Low-latency (Java 11+) |

⚠️ **System.gc()** is a **request**, not guarantee. JVM may ignore it.

---

## 6. Reflection API

### What is Reflection?
Ability to **inspect and modify** classes, methods, fields **at runtime**.

### Use Cases
- Frameworks (Spring, Hibernate)
- Testing (JUnit)
- Dynamic proxies
- Serialization

⚠️ **Drawbacks:**
- Performance overhead
- Security issues
- Breaks encapsulation

---

## 7. Getting Class Object

```java
// Method 1: Class.forName()
Class<?> clazz1 = Class.forName("java.lang.String");

// Method 2: .class literal
Class<?> clazz2 = String.class;

// Method 3: getClass()
String s = "Hello";
Class<?> clazz3 = s.getClass();

// All three return same Class object
System.out.println(clazz1 == clazz2);  // true
System.out.println(clazz2 == clazz3);  // true
```

---

## 8. Class Information

```java
Class<?> clazz = String.class;

// Names
String name = clazz.getName();  // java.lang.String
String simpleName = clazz.getSimpleName();  // String
Package pkg = clazz.getPackage();  // java.lang

// Modifiers
int modifiers = clazz.getModifiers();
boolean isFinal = Modifier.isFinal(modifiers);
boolean isPublic = Modifier.isPublic(modifiers);

// Hierarchy
Class<?> superClass = clazz.getSuperclass();  // Object
Class<?>[] interfaces = clazz.getInterfaces();

// Check
boolean isInterface = clazz.isInterface();
boolean isArray = clazz.isArray();
boolean isPrimitive = clazz.isPrimitive();
```

---

## 9. Accessing Methods

```java
Class<?> clazz = String.class;

// All public methods (including inherited)
Method[] methods = clazz.getMethods();

// All declared methods (only this class, all access levels)
Method[] declaredMethods = clazz.getDeclaredMethods();

// Specific method
Method method = clazz.getMethod("substring", int.class, int.class);

// Method info
String methodName = method.getName();           // substring
Class<?> returnType = method.getReturnType();   // String
Class<?>[] params = method.getParameterTypes(); // [int, int]
```

---

## 10. Invoking Methods

```java
Class<?> clazz = String.class;
Method method = clazz.getMethod("substring", int.class, int.class);

// Invoke method
String str = "Hello World";
Object result = method.invoke(str, 0, 5);  // "Hello"
System.out.println(result);
```

### Invoking Static Methods

```java
Class<?> clazz = Integer.class;
Method method = clazz.getMethod("parseInt", String.class);

// For static methods, first parameter is null
Object result = method.invoke(null, "123");  // 123
System.out.println(result);
```

---

## 11. Accessing Fields

```java
class Person {
    public String name = "Alice";
    private int age = 25;
}

Class<?> clazz = Person.class;

// Public field
Field nameField = clazz.getField("name");
Person p = new Person();
String name = (String) nameField.get(p);  // "Alice"

// Private field
Field ageField = clazz.getDeclaredField("age");
ageField.setAccessible(true);  // Bypass access control
int age = (int) ageField.get(p);  // 25

// Modify field
ageField.set(p, 30);
System.out.println(p.age);  // 30 (if accessible)
```

⚠️ **setAccessible(true)** bypasses normal access control (use carefully!).

---

## 12. Creating Instances

```java
// Using Class.newInstance() (deprecated)
Class<?> clazz = String.class;
// Object obj = clazz.newInstance();  // Deprecated

// Using Constructor
Constructor<?> constructor = clazz.getConstructor(String.class);
Object obj = constructor.newInstance("Hello");
System.out.println(obj);  // Hello

// No-arg constructor
Class<?> clazz2 = ArrayList.class;
Constructor<?> constructor2 = clazz2.getConstructor();
ArrayList<String> list = (ArrayList<String>) constructor2.newInstance();
```

---

## 🔥 Top MCQs for Session 16

### MCQ 1: ClassLoader Hierarchy
**Q:** ClassLoader hierarchy (top to bottom)?
1. Application → Extension → Bootstrap
2. Bootstrap → Extension → Application
3. Extension → Bootstrap → Application
4. Bootstrap → Application → Extension

**Answer:** 2. Bootstrap → Extension → Application  
**Explanation:** Parent delegation: Bootstrap at top, Application at bottom.

---

### MCQ 2: JIT Compiles
**Q:** JIT compiles:
1. All code
2. Hot spots (frequently executed)
3. Only main method
4. Static methods

**Answer:** 2. Hot spots  
**Explanation:** JIT optimizes frequently executed code, not all bytecode.

---

### MCQ 3: Heap Shared?
**Q:** Heap is shared among threads?
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** Heap and Method Area are shared; Stack is per-thread.

---

### MCQ 4: setAccessible(true)
**Q:** setAccessible(true) allows:
1. Making class public
2. Accessing private members
3. Changing method signature
4. Creating instances

**Answer:** 2. Accessing private members  
**Explanation:** Bypasses access control for fields/methods.

---

### MCQ 5: getClass() Returns
**Q:** getClass() returns:
1. Class name as String
2. Compile-time type
3. Runtime class
4. Package name

**Answer:** 3. Runtime class  
**Explanation:** Returns actual runtime class, not compile-time reference type.

---

### MCQ 6: Method Invocation
**Q:** To invoke static method via reflection:
1. Pass null as first parameter
2. Pass class object
3. Don't pass anything
4. Cannot invoke static methods

**Answer:** 1. Pass null as first parameter  
**Explanation:** method.invoke(null, args) for static methods.

---

### MCQ 7: Metaspace
**Q:** Java 8+ replaced PermGen with:
1. Heap
2. Stack
3. Metaspace
4. Eden Space

**Answer:** 3. Metaspace  
**Explanation:** Metaspace (native memory) replaced PermGen in Java 8.

---

### MCQ 8: Parent Delegation
**Q:** ClassLoader follows:
1. Child delegation
2. Parent delegation
3. No delegation
4. Random selection

**Answer:** 2. Parent delegation  
**Explanation:** Always delegates to parent first.

---

### MCQ 9: Reflection Performance
**Q:** Reflection has:
1. Better performance
2. Same performance
3. Performance overhead
4. No impact

**Answer:** 3. Performance overhead  
**Explanation:** Reflection is slower due to runtime checks.

---

### MCQ 10: GC Request
**Q:** System.gc() guarantees garbage collection?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** It's a request; JVM may ignore it.

---

## ⚠️ Common Mistakes

1. **Assuming System.gc()** runs immediately
2. **Using reflection** when not needed (performance)
3. **setAccessible(true)** security implications
4. **Confusing compile-time type** with **runtime class**
5. **Not handling** ClassNotFoundException
6. **Forgetting null** for static method invocation
7. **Mixing up** ClassLoader hierarchy order

---

## ⭐ One-liner Exam Facts

1. ClassLoader hierarchy: **Bootstrap → Extension → Application**
2. **Parent delegation**: Always ask parent first
3. Bootstrap loads **rt.jar** (core Java)
4. JIT compiles **hot spots** (frequently executed code)
5. **Heap** and **Method Area** shared among threads
6. **Stack** per-thread (local vars, method calls)
7. Java 8+ **Metaspace** replaced **PermGen**
8. Metaspace uses **native memory**, not heap
9. **System.gc()** is **request**, not guarantee
10. Reflection allows **runtime** inspection/modification
11. **setAccessible(true)** bypasses access control
12. getClass() returns **runtime class**
13. **Three ways** to get Class: forName, .class, getClass()
14.  Static method invoke: **null** as first parameter
15. Reflection has **performance overhead**

---

**End of Session 16**
