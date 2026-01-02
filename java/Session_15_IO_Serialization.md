# Session 15 – IO & Serialization

**Topics Covered:** Byte Streams vs Character Streams, File IO, BufferedReader/Writer, Serialization, transient & static, serialVersionUID, Externalization

---

## 1. IO Streams Overview

**Stream** = Sequence of data (input or output).

### Stream Categories

```
Java IO Streams
├── Byte Streams (8-bit binary)
│   ├── InputStream (abstract)
│   │   ├── FileInputStream
│   │   ├── BufferedInputStream
│   │   └── ObjectInputStream
│   └── OutputStream (abstract)
│       ├── FileOutputStream
│       ├── BufferedOutputStream
│       └── ObjectOutputStream
│
└── Character Streams (16-bit Unicode)
    ├── Reader (abstract)
    │   ├── FileReader
    │   ├── BufferedReader
    │   └── InputStreamReader
    └── Writer (abstract)
        ├── FileWriter
        ├── BufferedWriter
        └── OutputStreamWriter
```

---

## 2. Byte Streams vs Character Streams

| Aspect | Byte Streams | Character Streams |
|--------|--------------|-------------------|
| **Unit** | 8-bit bytes | 16-bit Unicode characters |
| **Base classes** | InputStream, OutputStream | Reader, Writer |
| **Use case** | Binary data (images, audio, video) | Text data |
| **Encoding** | No character encoding | Handles character encoding |
| **Examples** | FileInputStream, FileOutputStream | FileReader, FileWriter |
| **Method** | `read()` returns int (byte) | `read()` returns int (char) |

⭐ **Exam Fact:** Use **byte streams** for binary, **character streams** for text.

---

## 3. Reading Files

### FileInputStream (Byte Stream)

```java
import java.io.FileInputStream;
import java.io.IOException;

try (FileInputStream fis = new FileInputStream("file.txt")) {
    int data;
    while ((data = fis.read()) != -1) {  // -1 = end of file
        System.out.print((char) data);
    }
} catch (IOException e) {
    e.printStackTrace();
}
```

### FileReader (Character Stream)

```java
import java.io.FileReader;
import java.io.IOException;

try (FileReader fr = new FileReader("file.txt")) {
    int data;
    while ((data = fr.read()) != -1) {
        System.out.print((char) data);
    }
} catch (IOException e) {
    e.printStackTrace();
}
```

### BufferedReader (Efficient, Line-by-Line)

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

try (BufferedReader br = new BufferedReader(new FileReader("file.txt"))) {
    String line;
    while ((line = br.readLine()) != null) {
        System.out.println(line);
    }
} catch (IOException e) {
    e.printStackTrace();
}
```

⭐ **Exam Fact:** **BufferedReader** is more efficient (reads in chunks, not character-by-character).

---

## 4. Writing Files

### FileOutputStream (Byte Stream)

```java
import java.io.FileOutputStream;
import java.io.IOException;

try (FileOutputStream fos = new FileOutputStream("output.txt")) {
    String data = "Hello World";
    fos.write(data.getBytes());
} catch (IOException e) {
    e.printStackTrace();
}
```

### FileWriter (Character Stream)

```java
import java.io.FileWriter;
import java.io.IOException;

try (FileWriter fw = new FileWriter("output.txt")) {
    fw.write("Hello World");
} catch (IOException e) {
    e.printStackTrace();
}
```

###  BufferedWriter (Efficient)

```java
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

try (BufferedWriter bw = new BufferedWriter(new FileWriter("output.txt"))) {
    bw.write("Line 1");
    bw.newLine();  // Platform-independent newline
    bw.write("Line 2");
} catch (IOException e) {
    e.printStackTrace();
}
```

---

## 5. Serialization

### What is Serialization?
**Serialization** = Converting object into byte stream for storage/transmission.  
**Deserialization** = Converting byte stream back to object.

### Why Serialization?
- Save object to file
- Send object over network
- Persist object state

---

## 6. Making Class Serializable

```java
import java.io.Serializable;

class Employee implements Serializable {
    private static final long serialVersionUID = 1L;  // Version control
    
    String name;
    int age;
    double salary;
}
```

⚠️ **Must implement Serializable** interface (marker interface - no methods).

---

## 7. Writing Object (Serialization)

```java
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;

Employee emp = new Employee();
emp.name = "Alice";
emp.age = 25;
emp.salary = 50000;

try (ObjectOutputStream oos = new ObjectOutputStream(
        new FileOutputStream("employee.ser"))) {
    oos.writeObject(emp);
    System.out.println("Object serialized");
} catch (IOException e) {
    e.printStackTrace();
}
```

---

## 8. Reading Object (Deserialization)

```java
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.IOException;

try (ObjectInputStream ois = new ObjectInputStream(
        new FileInputStream("employee.ser"))) {
    Employee emp = (Employee) ois.readObject();
    System.out.println("Name: " + emp.name);
    System.out.println("Age: " + emp.age);
    System.out.println("Salary: " + emp.salary);
} catch (IOException | ClassNotFoundException e) {
    e.printStackTrace();
}
```

---

## 9. transient keyword

**transient** = Field NOT serialized.

```java
class Employee implements Serializable {
    private static final long serialVersionUID = 1L;
    
    String name;
    int age;
    transient String password;  // NOT serialized
    transient int tempValue;    // NOT serialized
}

// After deserialization:
// password = null
// tempValue = 0 (default)
```

⭐ **Exam Fact:** **transient** fields get **default values** after deserialization.

---

## 10. static Fields Not Serialized

```java
class Employee implements Serializable {
    private static final long serialVersionUID = 1L;
    
    String name;
    static String company = "TechCorp";  // NOT serialized
}

// static fields belong to CLASS, not instance
// After deserialization, company = current static value (not serialized value)
```

⭐ **Exam Fact:** **static** fields NOT serialized (belong to class, not object).

---

## 11. serialVersionUID

### What is serialVersionUID?
Version control identifier for serialized class.

```java
class Employee implements Serializable {
    private static final long serialVersionUID = 1L;
}
```

### Why Use It?

```
Day 1: Serialize Employee v1
Day 2: Modify Employee class (add field)
Day 3: Try to deserialize old file
→ InvalidClassException! (version mismatch)
```

**Solution:** Explicitly define serialVersionUID:
- Same UID → Compatible (deserialization works)
- Different UID → Incompatible (exception)

⚠️ **If not defined:** Java auto-generates based on class structure. Any change = different UID = exception.

⭐ **Exam Fact:** **serialVersionUID** ensures version c ompatibility.

---

## 12. Serialization Inheritance

### Parent Serializable, Child Auto-Serializable

```java
class Person implements Serializable {
    String name;
}

class Employee extends Person {  // Automatically Serializable
    int salary;
}
```

### Parent Not Serializable

```java
class Person {  // NOT Serializable
    String name;
    
    public Person() { }  // No-arg constructor REQUIRED
}

class Employee extends Person implements Serializable {
    int salary;
}

// After deserialization:
// salary = restored
// name = NOT restored (default value, then constructor called)
```

⚠️ **Parent must have no-arg constructor** if not Serializable.

---

## 13. Serialization Edge Cases

### Circular References

```java
class A implements Serializable {
    B b;
}

class B implements Serializable {
    A a;
}

A a = new A();
B b = new B();
a.b = b;
b.a = a;  // Circular reference

// Serialization works! Java handles it automatically
```

### Array Serialization

```java
int[] arr = {1, 2, 3, 4, 5};
ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("array.ser"));
oos.writeObject(arr);  // Arrays are Serializable
```

---

## 14. try-with-resources (Automatic Close)

```java
// Old way (manual close)
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

// New way (automatic close - Java 7+)
try (BufferedReader br = new BufferedReader(new FileReader("file.txt"))) {
    String line = br.readLine();
} catch (IOException e) {
    e.printStackTrace();
}  // Auto-closed here
```

⭐ **Exam Fact:** Resources in try-with-resources must implement **AutoCloseable**.

---

## 🔥 Top MCQs for Session 15

### MCQ 1: Byte vs Character Streams
**Q:** Character streams use:
1. 8-bit
2. 16-bit
3. 32-bit
4. 64-bit

**Answer:** 2. 16-bit  
**Explanation:** Character streams handle 16-bit Unicode characters.

---

### MCQ 2: transient
**Q:** transient fields are:
1. Serialized
2. NOT serialized
3. Optional
4. Static

**Answer:** 2. NOT serialized  
**Explanation:** transient keyword prevents field serialization.

---

### MCQ 3: static Serialization
**Q:** static fields are serialized?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** static fields belong to class, not instance. Not serialized.

---

### MCQ 4: serialVersionUID
**Q:** serialVersionUID purpose?
1. Performance
2. Security
3. Version control/compatibility
4. Compression

**Answer:** 3. Version control/compatibility  
**Explanation:** Ensures serialized object is compatible with class version.

---

### MCQ 5: Serializable
**Q:** Serializable is:
1. Class
2. Interface with methods
3. Marker interface (no methods)
4. Abstract class

**Answer:** 3. Marker interface  
**Explanation:** Serializable has no methods, just marks class as serializable.

---

### MCQ 6: BufferedReader
**Q:** BufferedReader.readLine() returns:
1. char
2. String
3. int
4. byte[]

**Answer:** 2. String  
**Explanation:** readLine() reads entire line as String.

---

### MCQ 7: FileReader Use Case
**Q:** FileReader is best for:
1. Binary data
2. Text data
3. Images
4. Audio

**Answer:** 2. Text data  
**Explanation:** FileReader is character stream for text.

---

### MCQ 8: try-with-resources
**Q:** try-with-resources requires:
1. Serializable
2. Closeable
3. AutoCloseable
4. Runnable

**Answer:** 3. AutoCloseable  
**Explanation:** Resources must implement AutoCloseable interface.

---

### MCQ 9: Deserialization
**Q:** transient int value becomes:
1. null
2. 0
3. -1
4. Exception

**Answer:** 2. 0  
**Explanation:** transient fields get default values (0 for int).

---

### MCQ 10: Parent Not Serializable
**Q:** If parent is NOT Serializable, it needs:
1. static method
2. No-arg constructor
3. finalize method
4. Nothing special

**Answer:** 2. No-arg constructor  
**Explanation:** Parent's no-arg constructor called during deserialization.

---

## ⚠️ Common Mistakes

1. **Using byte streams for text** (encoding issues)
2. **Forgetting transient** for sensitive data (passwords)
3. **Not defining serialVersionUID** explicitly
4. **Serializing static fields** (they aren't serialized)
5. **Not closing streams** (use try-with-resources)
6. **Parent without no-arg constructor** when not Serializable
7. **Confusion**: transient vs static

---

## ⭐ One-liner Exam Facts

1. **Byte streams:** 8-bit, **Character streams:** 16-bit
2. **Byte streams:** InputStream/OutputStream
3. **Character streams:** Reader/Writer
4. **BufferedReader** more efficient (reads in chunks)
5. **readLine()** returns **String** (null at EOF)
6. **Serializable** is **marker interface** (no methods)
7. **transient** fields **NOT serialized**
8. **static** fields **NOT serialized**
9. transient defaults: **null** (objects), **0** (numbers), **false** (boolean)
10. **serialVersionUID** for version compatibility
11. Parent NOT Serializable needs **no-arg constructor**
12. try-with-resources requires **AutoCloseable**
13. FileInputStream/FileOutputStream for **binary**
14. FileReader/FileWriter for **text**
15. Always **close** streams (or use try-with-resources)

---

**End of Session 15**
