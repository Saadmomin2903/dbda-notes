# Session 5 – Inheritance & Polymorphism

**Topics Covered:** Inheritance, Abstract Classes, Interfaces, Inner Classes, Anonymous Inner Classes, Polymorphism, Upcasting/Downcasting, Virtual Methods, Method Overriding

---

## 1. Inheritance

### Definition
Mechanism where one class acquires properties and behaviors of another class.

```java
// Parent class (Superclass)
public class Animal {
    String name;
    
    public void eat() {
        System.out.println("Animal is eating");
    }
}

// Child class (Subclass)
public class Dog extends Animal {
    public void bark() {
        System.out.println("Dog is barking");
    }
}

// Usage
Dog dog = new Dog();
dog.eat();   // Inherited from Animal
dog.bark();  // Dog's own method
```

### Types of Inheritance (Class Level)

```
SUPPORTED:
1. Single Inheritance:    A → B
2. Multilevel Inheritance: A → B → C
3. Hierarchical: A → B
                 A → C

NOT SUPPORTED (Diamond Problem):
4. Multiple Inheritance (class): B ← A → C
                                    ↓
                                    D (Ambiguity!)
```

⭐ **Exam Fact:** Java supports **single inheritance** at class level, **multiple inheritance** via interfaces.

### super Keyword

```java
class Parent {
    int x = 10;
    
    public void display() {
        System.out.println("Parent");
    }
}

class Child extends Parent {
    int x = 20;
    
    public void display() {
        System.out.println(super.x);    // 10 (parent's x)
        System.out.println(this.x);     // 20 (child's x)
        super.display();                // Calls parent's display()
    }
}
```

### Constructor Chaining in Inheritance

```java
class Parent {
    public Parent() {
        System.out.println("Parent constructor");
    }
}

class Child extends Parent {
    public Child() {
        // super(); // Implicit call to parent's no-arg constructor
        System.out.println("Child constructor");
    }
}

Child c = new Child();
// Output:
// Parent constructor
// Child constructor
```

⚠️ **Common Mistake:**
```java
class Parent {
    public Parent(int x) { }
}

class Child extends Parent {
    public Child() {  // ERROR: implicit super() call fails
    }
}

// FIX:
class Child extends Parent {
    public Child() {
        super(10);  // Explicit call to Parent(int)
    }
}
```

⭐ **Exam Fact:** First statement in constructor is either `this()` or `super()` (implicitly added if not present).

---

## 2. Abstract Classes

### Definition
Class that **cannot be instantiated**, used as a blueprint.

```java
abstract class Shape {
    String color;
    
    // Abstract method (no body)
    abstract void draw();
    
    // Concrete method
    public void setColor(String color) {
        this.color = color;
    }
}

class Circle extends Shape {
    // MUST implement abstract method
    void draw() {
        System.out.println("Drawing Circle");
    }
}

// Usage
// Shape s = new Shape();  // ERROR: Cannot instantiate
Shape s = new Circle();    // OK (polymorphism)
s.draw();
```

### Abstract Class Rules

| Rule | Description |
|------|-------------|
| **Cannot be instantiated** | `new Shape()` → ERROR |
| **Can have constructors** | Used by subclasses |
| **Can have abstract methods** | No body, ended with `;` |
| **Can have concrete methods** | With implementation |
| **Can have instance variables** | Yes |
| **Can have static members** | Yes |
| **Subclass must implement all abstract methods** | Or be abstract itself |

```java
abstract class Vehicle {
    int wheels;
    
    // Constructor
    public Vehicle(int wheels) {
        this.wheels = wheels;
    }
    
    // Abstract method
    abstract void start();
    
    // Concrete method
    void stop() {
        System.out.println("Vehicle stopped");
    }
}

class Car extends Vehicle {
    public Car() {
        super(4);  // Call abstract class constructor
    }
    
    void start() {
        System.out.println("Car started");
    }
}
```

---

## 3. Interfaces

### Definition
Blueprint of a class containing **only abstract methods** (Java 7) and **constants**.

```java
interface Drawable {
    // All methods are public abstract by default
    void draw();
    
    // All variables are public static final
    int MAX_SIZE = 100;
}

class Rectangle implements Drawable {
    public void draw() {  // MUST be public
        System.out.println("Drawing Rectangle");
    }
}
```

### Java 8+ Interface Features

```java
interface MyInterface {
    // Abstract method
    void abstractMethod();
    
    // Default method (Java 8+)
    default void defaultMethod() {
        System.out.println("Default implementation");
    }
    
    // Static method (Java 8+)
    static void staticMethod() {
        System.out.println("Static method");
    }
    
    // Private method (Java 9+)
    private void privateHelper() {
        System.out.println("Private helper");
    }
}
```

### Multiple Inheritance via Interfaces

```java
interface A {
    void methodA();
}

interface B {
    void methodB();
}

class C implements A, B {
    public void methodA() { }
    public void methodB() { }
}
```

### Diamond Problem Resolution

```java
interface A {
    default void display() {
        System.out.println("A");
    }
}

interface B {
    default void display() {
        System.out.println("B");
    }
}

class C implements A, B {
    // MUST override to resolve ambiguity
    public void display() {
        A.super.display();  // Explicitly call A's implementation
        // OR
        B.super.display();  // Explicitly call B's implementation
    }
}
```

### Abstract Class vs Interface

| Feature | Abstract Class | Interface |
|---------|----------------|-----------|
| **Methods** | abstract + concrete | abstract + default + static (Java 8+) |
| **Variables** | Any type | public static final only |
| **Constructor** | Yes | No |
| **Multiple Inheritance** | No | Yes |
| **Access Modifiers** | All | public (methods must be public) |
| **Keyword** | `extends` | `implements` |
| **Use Case** | IS-A relationship with shared code | Contract/capability |

⭐ **Exam Fact:** Use **abstract class** when you have common code. Use **interface** when you want to define a contract.

---

## 4. Inner Classes

### Types of Inner Classes

#### 1. Member Inner Class

```java
class Outer {
    private int x = 10;
    
    class Inner {
        void display() {
            System.out.println(x);  // Can access outer's private members
        }
    }
}

// Usage
Outer outer = new Outer();
Outer.Inner inner = outer.new Inner();
inner.display();
```

#### 2. Static Nested Class

```java
class Outer {
    static int x = 10;
    
    static class StaticNested {
        void display() {
            System.out.println(x);  // Can access outer's static members only
        }
    }
}

// Usage
Outer.StaticNested nested = new Outer.StaticNested();
nested.display();
```

#### 3. Local Inner Class

```java
class Outer {
    void method() {
        int localVar = 20;  // Effectively final
        
        class LocalInner {
            void display() {
                System.out.println(localVar);
            }
        }
        
        LocalInner inner = new LocalInner();
        inner.display();
    }
}
```

#### 4. Anonymous Inner Class

```java
interface Greeting {
    void greet();
}

public class Test {
    public static void main(String[] args) {
        // Anonymous inner class
        Greeting g = new Greeting() {
            public void greet() {
                System.out.println("Hello!");
            }
        };
        g.greet();
    }
}
```

---

## 5. Polymorphism

### Definition
Ability of an object to take many forms.

### Types of Polymorphism

#### 1. Compile-time (Static) Polymorphism - Method Overloading

```java
class Calculator {
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
}
```

#### 2. Runtime (Dynamic) Polymorphism - Method Overriding

```java
class Animal {
    void sound() {
        System.out.println("Animal makes sound");
    }
}

class Dog extends Animal {
    void sound() {
        System.out.println("Dog barks");
    }
}

// Runtime polymorphism
Animal a = new Dog();  // Upcasting
a.sound();  // "Dog barks" (decided at runtime)
```

---

## 6. Method Overriding

### Rules

```java
class Parent {
    void display() {
        System.out.println("Parent");
    }
}

class Child extends Parent {
    @Override  // Optional but recommended
    void display() {
        System.out.println("Child");
    }
}
```

### Override Rules (5-5-5 Rule)

| Aspect | Rule |
|--------|------|
| **Method signature** | MUST be same |
| **Return type** | Same or covariant (subtype) |
| **Access modifier** | Same or wider (not narrower) |
| **Exceptions** | Same or narrower (not broader) for checked exceptions |
| **static/final/private** | Cannot be overridden |

```java
class Parent {
    protected Number getValue() { return 10; }
    void display() throws IOException { }
}

class Child extends Parent {
    // Valid: covariant return type
    public Integer getValue() { return 20; }  // Integer is subtype of Number
    
    // Valid: narrower exception
    void display() throws FileNotFoundException { }  // FileNotFoundException extends IOException
}
```

⚠️ **Common MCQ Traps:**

```java
class Parent {
    void display() { }
}

class Child extends Parent {
    private void display() { }  // ERROR: Cannot reduce visibility
}
```

```java
class Parent {
    static void display() { }
}

class Child extends Parent {
    void display() { }  // ERROR: Cannot override static with non-static
}
```

---

## 7. Upcasting & Downcasting

### Upcasting (Implicit)

```java
Animal a = new Dog();  // Upcasting (implicit)
a.eat();  // OK (Animal has eat())
// a.bark();  // ERROR: Animal doesn't have bark()
```

### Downcasting (Explicit)

```java
Animal a = new Dog();
Dog d = (Dog) a;  // Downcasting (explicit)
d.bark();  // OK now

// Dangerous downcasting
Animal a = new Animal();
Dog d = (Dog) a;  // ClassCastException at runtime!
```

### instanceof Operator

```java
Animal a = new Dog();

if (a instanceof Dog) {
    Dog d = (Dog) a;
    d.bark();
}
```

⭐ **Exam Fact:** Use `instanceof` before downcasting to avoid ClassCastException.

---

## 8. Virtual Methods

### Definition
Method call resolved at **runtime** based on actual object type.

```java
class Parent {
    void display() {
        System.out.println("Parent");
    }
}

class Child extends Parent {
    void display() {
        System.out.println("Child");
    }
}

Parent p = new Child();
p.display();  // "Child" (virtual method call)
```

### Method Dispatch

```
Compile Time: Check if method exists in reference type (Parent)
Runtime: Execute method from actual object type (Child)
```

### Example

```java
class A {
    void m1() { System.out.print("A"); }
}

class B extends A {
    void m1() { System.out.print("B"); }
}

class C extends B {
    void m1() { System.out.print("C"); }
}

A obj = new C();
obj.m1();  // "C" (virtual method call)
```

---

## 🔥 Top MCQs for Session 5

### MCQ 1: Inheritance Type
**Q:** Which inheritance is NOT supported in Java?
1. Single
2. Multilevel
3. Multiple (class level)
4. Hierarchical

**Answer:** 3. Multiple (class level)  
**Explanation:** Diamond problem. Use interfaces for multiple inheritance.

---

### MCQ 2: Constructor Chaining
**Q:** What is the output?
```java
class A {
    A() { System.out.print("A"); }
}

class B extends A {
    B() { System.out.print("B"); }
}

new B();
```
1. A
2. B
3. AB
4. BA

**Answer:** 3. AB  
**Explanation:** Parent constructor called first (implicit super()).

---

### MCQ 3: Abstract Class
**Q:** Which is TRUE?
1. Abstract class can be instantiated
2. Abstract class cannot have constructor
3. Abstract class can have concrete methods
4. Abstract class must have abstract methods

**Answer:** 3. Abstract class can have concrete methods  
**Explanation:** Abstract class can have 0 or more abstract methods and concrete methods.

---

### MCQ 4: Interface Methods
**Q:** In Java 8+, which method can have implementation in interface?
1. Abstract method
2. Default method
3. Both
4. None

**Answer:** 2. Default method  
**Explanation:** Default and static methods can have implementation.

---

### MCQ 5: Multiple Inheritance
**Q:** A class can implement how many interfaces?
1. 0
2. 1
3. 2
4. Multiple

**Answer:** 4. Multiple  
**Explanation:** Class can implement unlimited interfaces.

---

### MCQ 6: Method Overriding
**Q:** Can we override private method?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** Private methods are not visible to subclass, hence cannot be overridden.

---

### MCQ 7: Access Modifier in Override
**Q:** Is this valid override?
```java
class Parent {
    protected void display() { }
}

class Child extends Parent {
    public void display() { }
}
```
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** Access can be widened (protected → public). Cannot be narrowed.

---

### MCQ 8: Upcasting/Downcasting
**Q:** What is the output?
```java
class A {
    void m() { System.out.print("A"); }
}

class B extends A {
    void m() { System.out.print("B"); }
}

A a = new B();
a.m();
```
1. A
2. B
3. Compile error
4. Runtime error

**Answer:** 2. B  
**Explanation:** Virtual method call. Method resolved at runtime based on actual object (B).

---

### MCQ 9: instanceof
**Q:** What is the output?
```java
class A { }
class B extends A { }

A a = new B();
System.out.println(a instanceof B);
```
1. true
2. false
3. Compile error
4. Runtime error

**Answer:** 1. true  
**Explanation:** Instance of B, so instanceof B returns true.

---

### MCQ 10: Covariant Return Type
**Q:** Is this valid override?
```java
class Parent {
    Number getValue() { return 10; }
}

class Child extends Parent {
    Integer getValue() { return 20; }
}
```
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** Covariant return type allowed (Integer is subtype of Number).

---

## ⚠️ Common Mistakes

1. **Trying to instantiate abstract class**
2. **Not implementing all abstract methods** in subclass
3. **Narrowing access modifier** in override
4. **Overriding static/final/private** methods
5. **Downcasting without instanceof** check
6. **Forgetting super()** when parent has parameterized constructor
7. **Multiple class inheritance** (use interfaces)

---

## ⭐ One-liner Exam Facts

1. Java supports **single inheritance** at class level
2. **super()** or **this()** must be first statement in constructor
3. Abstract class **can have constructor** (used by subclass)
4. Interface variables are **public static final** by default
5. Override access modifier can be **same or wider**, not narrower
6. **Cannot override** private, static, final methods
7. Use **instanceof** before downcasting
8. Virtual method call resolved at **runtime**
9. **Covariant return type** allowed in overriding
10. Default methods in interface (Java 8+) allow **implementation**

---

**End of Session 5**
