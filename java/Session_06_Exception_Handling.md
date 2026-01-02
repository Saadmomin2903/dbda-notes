# Session 6 – Exception Handling

**Topics Covered:** Exception Hierarchy, Checked vs Unchecked, try-catch-finally, throw vs throws, Multiple & Nested catch, Custom Exceptions, Errors vs Exceptions

---

## 1. Exception Hierarchy

```
                    Throwable
                        |
        ┌───────────────┴───────────────┐
        |                               |
     Error                          Exception
        |                               |
  ┌─────┴─────┐           ┌─────────────┴──────────────┐
  |           |           |                            |
OutOfMemory  StackOver  IOException           RuntimeException
Error        flowError  (Checked)                 (Unchecked)
                            |                          |
                        ┌───┴───┐        ┌───────── ───┴─────────────┐
                   FileNot  SQL   NullPointer ArrayIndex  Arithmetic
                   Found  Exception  Exception  OutOfBounds Exception
```

### Key Classes

| Class | Type | Description |
|-------|------|-------------|
| **Throwable** | Root | Base class for all exceptions |
| **Error** | Unchecked | Serious JVM problems (OutOfMemoryError) |
| **Exception** | Checked/Unchecked | Application-level problems |
| **RuntimeException** | Unchecked | Programming errors |
| **IOException** | Checked | I/O failures |

---

## 2. Checked vs Unchecked Exceptions

### Checked Exceptions
**Must be handled** at compile time (try-catch or throws).

```java
// IOException (checked)
public void readFile() throws IOException {
    FileReader fr = new FileReader("file.txt");  // Must handle or declare
}

// Handling with try-catch
public void readFile() {
    try {
        FileReader fr = new FileReader("file.txt");
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

### Unchecked Exceptions
**Not required** to handle at compile time.

```java
// NullPointerException (unchecked)
public void method() {
    String s = null;
    s.length();  // No compile-time check, throws NPE at runtime
}

// ArithmeticException (unchecked)
int result = 10 / 0;  // No compile-time check
```

### Comparison Table

| Aspect | Checked | Unchecked |
|--------|---------|-----------|
| **Subclass of** | Exception (but not RuntimeException) | RuntimeException or Error |
| **Compile-time check** | Yes (must handle or declare) | No |
| **Examples** | IOException, SQLException | NullPointerException, ArrayIndexOutOfBoundsException |
| **Use Case** | Recoverable conditions | Programming errors |

⭐ **Exam Fact:** Checked exceptions extend **Exception** (but not RuntimeException). Unchecked extend **RuntimeException** or **Error**.

---

## 3. try-catch-finally

### Basic Syntax

```java
try {
    // Code that may throw exception
    int result = 10 / 0;
} catch (ArithmeticException e) {
    // Handle exception
    System.out.println("Cannot divide by zero");
} finally {
    // Always executes (even if exception occurs or return statement)
    System.out.println("Cleanup code");
}
```

### Execution Flow

```
┌─────────────────────────────────────────┐
│ try block executed                      │
├─────────────────────────────────────────┤
│ Exception thrown?                       │
│   YES → catch block → finally           │
│   NO → finally block                    │
└─────────────────────────────────────────┘
```

### finally Always Executes

```java
public static int method() {
    try {
        return 1;
    } finally {
        // Executes before return
        System.out.println("Finally block");
    }
}
```

⚠️ **Only exception:** JVM crash (`System.exit(0)`) prevents finally execution.

---

## 4. Multiple & Nested catch

### Multiple catch Blocks

```java
try {
    int[] arr = {1, 2, 3};
    System.out.println(arr[5]);
    int result = 10 / 0;
} catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("Array index error");
} catch (ArithmeticException e) {
    System.out.println("Arithmetic error");
} catch (Exception e) {  // Generic catch (must be last)
    System.out.println("Some error");
}
```

⭐ **Exam Fact:** catch blocks must go from **specific to generic**. Generic Exception catch must be last.

⚠️ **Common MCQ Trap:**
```java
try {
    // code
} catch (Exception e) {  // Generic first
} catch (IOException e) {  // ERROR: unreachable catch block
}
```

### Multi-catch (Java 7+)

```java
try {
    // code
} catch (IOException | SQLException e) {
    // Handle both exceptions
    System.out.println(e.getMessage());
}
```

### Nested try-catch

```java
try {
    try {
        int result = 10 / 0;
    } catch (ArithmeticException e) {
        System.out.println("Inner catch");
        throw e;  // Re-throw to outer catch
    }
} catch (ArithmeticException e) {
    System.out.println("Outer catch");
}
```

---

## 5. throw vs throws

### throw (Statement)
Used to **explicitly throw** an exception.

```java
public void validate(int age) {
    if (age < 18) {
        throw new IllegalArgumentException("Age must be 18+");
    }
}
```

### throws (Clause)
Used to **declare** that a method may throw exceptions.

```java
public void readFile() throws IOException {
    FileReader fr = new FileReader("file.txt");
}

// Caller must handle
public void caller() {
    try {
        readFile();
    } catch (IOException e) {
        e.printStackTrace();
    }
}
```

### Comparison Table

| Aspect | throw | throws |
|--------|-------|--------|
| **Purpose** | Throw exception | Declare exception |
| **Location** | Inside method body | Method signature |
| **Syntax** | `throw new Exception()` | `throws Exception` |
| **Count** | One exception per throw | Multiple exceptions |
| **Example** | `throw new IOException()` | `throws IOException, SQLException` |

⚠️ **Common Mistake:**
```java
// Cannot throw checked exception without declaring
public void method() {
    throw new IOException();  // ERROR: must declare with throws
}

// FIX:
public void method() throws IOException {
    throw new IOException();  // OK
}
```

---

## 6. Custom Exceptions

```java
// Custom checked exception
public class InsufficientBalanceException extends Exception {
    public InsufficientBalanceException(String message) {
        super(message);
    }
}

// Custom unchecked exception
public class InvalidAgeException extends RuntimeException {
    public InvalidAgeException(String message) {
        super(message);
    }
}

// Usage
public void withdraw(double amount) throws InsufficientBalanceException {
    if (amount > balance) {
        throw new InsufficientBalanceException("Balance too low");
    }
}
```

⭐ **Exam Fact:** 
- Extend **Exception** for checked custom exception
- Extend **RuntimeException** for unchecked custom exception

---

## 7. Errors vs Exceptions

| Aspect | Error | Exception |
|--------|-------|-----------|
| **Recovery** | Cannot recover (JVM issue) | Can recover (application issue) |
| **Examples** | OutOfMemoryError, StackOverflowError | IOException, NullPointerException |
| **Handling** | Should NOT catch | Should catch and handle |
| **Cause** | External (JVM, system) | Internal (application logic) |

```java
// DON'T DO THIS (catching Error)
try {
    // code
} catch (OutOfMemoryError e) {
    // Cannot recover from OOM
}

// DO THIS (catching Exception)
try {
    // code
} catch (IOException e) {
    // Can recover from IO error
}
```

---

## 8. Exception Flow Control

### Case 1: Exception in try, no catch match

```java
try {
    throw new IOException();
} catch (ArithmeticException e) {  // No match
    System.out.println("Catch");
} finally {
    System.out.println("Finally");
}
// Output: Finally
// Then IOException propagates to caller
```

### Case 2: Exception in catch block

```java
try {
    throw new IOException();
} catch (IOException e) {
    throw new RuntimeException();  // New exception
} finally {
    System.out.println("Finally");
}
// Output: Finally
// Then RuntimeException propagates
```

### Case 3: Exception masking

```java
try {
    throw new IOException();
} finally {
    throw new RuntimeException();  // Masks IOException!
}
// RuntimeException propagates, IOException lost
```

⚠️ **Common MCQ Trap:** Exception in finally block **masks** exception from try/catch.

---

## 9. Try-with-resources (Java 7+)

```java
// Old way
BufferedReader br = null;
try {
    br = new BufferedReader(new FileReader("file.txt"));
    String line = br.readLine();
} catch (IOException e) {
    e.printStackTrace();
} finally {
    if (br != null) {
        try {
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// New way (try-with-resources)
try (BufferedReader br = new BufferedReader(new FileReader("file.txt"))) {
    String line = br.readLine();
} catch (IOException e) {
    e.printStackTrace();
}
// br.close() called automatically
```

⭐ **Exam Fact:** Resource must implement **AutoCloseable** interface.

---

## 🔥 Top MCQs for Session 6

### MCQ 1: Checked vs Unchecked
**Q:** Which is a checked exception?
1. NullPointerException
2. IOException
3. ArithmeticException
4. ArrayIndexOutOfBoundsException

**Answer:** 2. IOException  
**Explanation:** IOException extends Exception (not RuntimeException).

---

### MCQ 2: catch Block Order
**Q:** Is this valid?
```java
try {
} catch (Exception e) {
} catch (IOException e) {
}
```
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** Specific exception (IOException) after generic (Exception) is unreachable.

---

### MCQ 3: finally Execution
**Q:** Will finally execute?
```java
try {
    return 1;
} finally {
    System.out.println("Finally");
}
```
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** finally executes even if return statement in try.

---

### MCQ 4: throw vs throws
**Q:** Which throws an exception?
1. throw
2. throws

**Answer:** 1. throw  
**Explanation:** throw is a statement that throws exception. throws is a declaration.

---

### MCQ 5: Exception Masking
**Q:** What is the output?
```java
try {
    throw new IOException("IO");
} finally {
    throw new RuntimeException("Runtime");
}
```
1. IOException propagates
2. RuntimeException propagates
3. Both propagate
4. Compile error

**Answer:** 2. RuntimeException propagates  
**Explanation:** Exception in finally masks exception from try.

---

### MCQ 6: Multi-catch
**Q:** Is this valid (Java 7+)?
```java
try {
} catch (IOException | SQLException e) {
}
```
1. Yes
2. No

**Answer:** 1. Yes  
**Explanation:** Multi-catch syntax allowed in Java 7+.

---

### MCQ 7: Custom Exception
**Q:** To create checked custom exception, extend:
1. Error
2. Exception
3. RuntimeException
4. Throwable

**Answer:** 2. Exception  
**Explanation:** Checked exceptions extend Exception (not RuntimeException).

---

### MCQ 8: finally vs return
**Q:** What is the output?
```java
public static int method() {
    try {
        return 1;
    } finally {
        return 2;
    }
}
```
1. 1
2. 2
3. Compile error
4. Runtime error

**Answer:** 2  
**Explanation:** finally return overwrites try return.

---

### MCQ 9: Unchecked Exception
**Q:** Which extends RuntimeException?
1. IOException
2. SQLException
3. NullPointerException
4. FileNotFoundException

**Answer:** 3. NullPointerException  
**Explanation:** NPE extends RuntimeException (unchecked).

---

### MCQ 10: try-with-resources
**Q:** Resource in try-with-resources must implement:
1. Closeable
2. AutoCloseable
3. Serializable
4. Runnable

**Answer:** 2. AutoCloseable  
**Explanation:** try-with-resources requires AutoCloseable interface.

---

## ⚠️ Common Mistakes

1. **Generic catch before specific** catch
2. **Catching Error** (should not catch JVM errors)
3. **Not declaring checked exceptions** with throws
4. **Exception masking** in finally block
5. **Assuming finally won't execute** with return
6. **Using throw without s** in method signature for checked exceptions

---

## ⭐ One-liner Exam Facts

1. Checked exceptions extend **Exception** (not RuntimeException)
2. Unchecked exceptions extend **RuntimeException** or **Error**
3. catch blocks must go **specific to generic**
4. finally executes **even with return** statement
5. Exception in finally **masks** exception from try/catch
6. **throw** = throw exception, **throws** = declare exception
7. Custom checked exception extends **Exception**
8. Custom unchecked exception extends **RuntimeException**
9. **Error** should NOT be caught (JVM issues)
10. try-with-resources requires **AutoCloseable** interface

---

**End of Session 6**
