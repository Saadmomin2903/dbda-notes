# Lab Exercise Solutions

**Total Programs:** 15  
**Purpose:** Complete working Java programs for lab exam preparation  
**Coverage:** File IO, JDBC, Collections, Multithreading, OOP

> ✅ **All programs tested and working**

---

## LAB 1: File Reading & Writing

### Task: Read from one file, write to another (uppercase conversion)

```java
import java.io.*;

public class FileCopyUpperCase {
    public static void main(String[] args) {
        try (BufferedReader br = new BufferedReader(new FileReader("input.txt"));
             BufferedWriter bw = new BufferedWriter(new FileWriter("output.txt"))) {
            
            String line;
            while ((line = br.readLine()) != null) {
                bw.write(line.toUpperCase());
                bw.newLine();
            }
            
            System.out.println("File copied successfully with uppercase conversion");
            
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
```

**Input (input.txt):**
```
Hello World
Java Programming
PG-DBDA
```

**Output (output.txt):**
```
HELLO WORLD
JAVA PROGRAMMING
PG-DBDA
```

---

## LAB 2: Word Count in File

### Task: Count words, lines, and characters in a file

```java
import java.io.*;

public class WordCount {
    public static void main(String[] args) {
        int wordCount = 0, lineCount = 0, charCount = 0;
        
        try (BufferedReader br = new BufferedReader(new FileReader("input.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                lineCount++;
                charCount += line.length();
                
                String[] words = line.trim().split("\\s+");
                if (!line.trim().isEmpty()) {
                    wordCount += words.length;
                }
            }
            
            System.out.println("Lines: " + lineCount);
            System.out.println("Words: " + wordCount);
            System.out.println("Characters: " + charCount);
            
        } catch (IOException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
```

---

## LAB 3: Object Serialization

### Task: Serialize and deserialize Employee object

```java
import java.io.*;

class Employee implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private int id;
    private String name;
    private double salary;
    private transient String password;  // Not serialized
    
    public Employee(int id, String name, double salary, String password) {
        this.id = id;
        this.name = name;
        this.salary = salary;
        this.password = password;
    }
    
    @Override
    public String toString() {
        return "Employee{id=" + id + ", name=" + name + 
               ", salary=" + salary + ", password=" + password + "}";
    }
}

public class SerializationDemo {
    public static void main(String[] args) {
        // Serialize
        Employee emp = new Employee(101, "Alice", 50000, "secret123");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream("employee.ser"))) {
            oos.writeObject(emp);
            System.out.println("Serialized: " + emp);
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        // Deserialize
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream("employee.ser"))) {
            Employee readEmp = (Employee) ois.readObject();
            System.out.println("Deserialized: " + readEmp);
            System.out.println("Password after deserialization: null (transient)");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
```

**Output:**
```
Serialized: Employee{id=101, name=Alice, salary=50000.0, password=secret123}
Deserialized: Employee{id=101, name=Alice, salary=50000.0, password=null}
Password after deserialization: null (transient)
```

---

## LAB 4: ArrayList Operations

### Task: Add, remove, search, and sort elements

```java
import java.util.*;

public class ArrayListOperations {
    public static void main(String[] args) {
        List<String> fruits = new ArrayList<>();
        
        // Add
        fruits.add("Apple");
        fruits.add("Banana");
        fruits.add("Cherry");
        fruits.add("Date");
        System.out.println("Initial: " + fruits);
        
        // Add at index
        fruits.add(1, "Mango");
        System.out.println("After insert: " + fruits);
        
        // Remove
        fruits.remove("Banana");
        System.out.println("After remove: " + fruits);
        
        // Search
        if (fruits.contains("Apple")) {
            int index = fruits.indexOf("Apple");
            System.out.println("Apple found at index: " + index);
        }
        
        // Sort
        Collections.sort(fruits);
        System.out.println("Sorted: " + fruits);
        
        // Iterate
        System.out.println("\nUsing forEach:");
        fruits.forEach(fruit -> System.out.println("- " + fruit));
    }
}
```

---

## LAB 5: HashMap - Student Database

### Task: Store and retrieve student records

```java
import java.util.*;

class Student {
    int rollNo;
    String name;
    double marks;
    
    public Student(int rollNo, String name, double marks) {
        this.rollNo = rollNo;
        this.name = name;
        this.marks = marks;
    }
    
    @Override
    public String toString() {
        return "Student{rollNo=" + rollNo + ", name=" + name + ", marks=" + marks + "}";
    }
}

public class StudentDatabase {
    public static void main(String[] args) {
        Map<Integer, Student> students = new HashMap<>();
        
        // Add students
        students.put(101, new Student(101, "Alice", 85.5));
        students.put(102, new Student(102, "Bob", 78.0));
        students.put(103, new Student(103, "Charlie", 92.5));
        
        // Display all
        System.out.println("All Students:");
        students.forEach((rollNo, student) -> 
            System.out.println(rollNo + " -> " + student));
        
        // Search
        int searchRoll = 102;
        if (students.containsKey(searchRoll)) {
            System.out.println("\nFound: " + students.get(searchRoll));
        }
        
        // Remove
        students.remove(102);
        System.out.println("\nAfter removing 102: " + students.keySet());
        
        // Sort by marks
        System.out.println("\nSorted by marks:");
        students.values().stream()
            .sorted((s1, s2) -> Double.compare(s2.marks, s1.marks))
            .forEach(System.out::println);
    }
}
```

---

## LAB 6: TreeSet - Sorted Unique Elements

### Task: Store integers in sorted order, demonstrate operations

```java
import java.util.*;

public class TreeSetDemo {
    public static void main(String[] args) {
        TreeSet<Integer> numbers = new TreeSet<>();
        
        // Add (duplicates ignored, sorted automatically)
        numbers.add(50);
        numbers.add(20);
        numbers.add(80);
        numbers.add(20);  // Duplicate, ignored
        numbers.add(35);
        
        System.out.println("TreeSet (sorted): " + numbers);
        
        // First and last
        System.out.println("First: " + numbers.first());
        System.out.println("Last: " + numbers.last());
        
        // Range operations
        System.out.println("HeadSet (<50): " + numbers.headSet(50));
        System.out.println("TailSet (>=35): " + numbers.tailSet(35));
        System.out.println("SubSet [20,80): " + numbers.subSet(20, 80));
        
        // Navigation
        System.out.println("Lower than 50: " + numbers.lower(50));
        System.out.println("Higher than 35: " + numbers.higher(35));
        
        // Descending
        System.out.println("Descending: " + numbers.descendingSet());
    }
}
```

**Output:**
```
TreeSet (sorted): [20, 35, 50, 80]
First: 20
Last: 80
HeadSet (<50): [20, 35]
TailSet (>=35): [35, 50, 80]
SubSet [20,80): [20, 35, 50]
Lower than 50: 35
Higher than 35: 50
Descending: [80, 50, 35, 20]
```

---

## LAB 7: Simple Producer-Consumer (Single Buffer)

### Task: Implement producer-consumer using wait/notify

```java
class SharedBuffer {
    private int data;
    private boolean available = false;
    
    public synchronized void produce(int value) throws InterruptedException {
        while (available) {
            wait();  // Wait if buffer full
        }
        data = value;
        available = true;
        System.out.println("Produced: " + value);
        notify();
    }
    
    public synchronized int consume() throws InterruptedException {
        while (!available) {
            wait();  // Wait if buffer empty
        }
        available = false;
        System.out.println("Consumed: " + data);
        notify();
        return data;
    }
}

public class ProducerConsumerDemo {
    public static void main(String[] args) {
        SharedBuffer buffer = new SharedBuffer();
        
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                try {
                    buffer.produce(i);
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                try {
                    buffer.consume();
                    Thread.sleep(150);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
        
        producer.start();
        consumer.start();
    }
}
```

---

## LAB 8: Thread Synchronization - Bank Account

### Task: Prevent race condition with synchronized

```java
class BankAccount {
    private int balance = 1000;
    
    // WITHOUT synchronized - race condition
    public void withdrawUnsafe(int amount) {
        if (balance >= amount) {
            System.out.println(Thread.currentThread().getName() + " checking balance: " + balance);
            balance -= amount;
            System.out.println(Thread.currentThread().getName() + " withdrew: " + amount + 
                             ", remaining: " + balance);
        } else {
            System.out.println(Thread.currentThread().getName() + " insufficient funds");
        }
    }
    
    // WITH synchronized - thread-safe
    public synchronized void withdrawSafe(int amount) {
        if (balance >= amount) {
            System.out.println(Thread.currentThread().getName() + " checking balance: " + balance);
            balance -= amount;
            System.out.println(Thread.currentThread().getName() + " withdrew: " + amount + 
                             ", remaining: " + balance);
        } else {
            System.out.println(Thread.currentThread().getName() + " insufficient funds");
        }
    }
}

public class BankThreadDemo {
    public static void main(String[] args) {
        BankAccount account = new BankAccount();
        
        Runnable task = () -> account.withdrawSafe(600);
        
        Thread t1 = new Thread(task, "User1");
        Thread t2 = new Thread(task, "User2");
        
        t1.start();
        t2.start();
    }
}
```

---

## LAB 9: JDBC - Create Table & Insert

### Task: Create database table and insert records

```java
import java.sql.*;

public class JDBCCreateInsert {
    static final String DB_URL = "jdbc:mysql://localhost:3306/testdb";
    static final String USER = "root";
    static final String PASS = "password";
    
    public static void main(String[] args) {
        // Create table
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASS);
             Statement stmt = conn.createStatement()) {
            
            String sql = "CREATE TABLE IF NOT EXISTS students (" +
                        "id INT PRIMARY KEY, " +
                        "name VARCHAR(50), " +
                        "marks DOUBLE)";
            stmt.executeUpdate(sql);
            System.out.println("Table created successfully");
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
        
        // Insert records
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASS);
             PreparedStatement pstmt = conn.prepareStatement(
                 "INSERT INTO students VALUES (?, ?, ?)")) {
            
            // Insert student 1
            pstmt.setInt(1, 101);
            pstmt.setString(2, "Alice");
            pstmt.setDouble(3, 85.5);
            pstmt.executeUpdate();
            
            // Insert student 2
            pstmt.setInt(1, 102);
            pstmt.setString(2, "Bob");
            pstmt.setDouble(3, 78.0);
            pstmt.executeUpdate();
            
            System.out.println("Records inserted successfully");
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

---

## LAB 10: JDBC - Select & Display

### Task: Retrieve and display all records

```java
import java.sql.*;

public class JDBCSelectDisplay {
    static final String DB_URL = "jdbc:mysql://localhost:3306/testdb";
    static final String USER = "root";
    static final String PASS = "password";
    
    public static void main(String[] args) {
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASS);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT * FROM students")) {
            
            System.out.printf("%-5s %-20s %-10s%n", "ID", "Name", "Marks");
            System.out.println("----------------------------------------");
            
            while (rs.next()) {
                int id = rs.getInt("id");
                String name = rs.getString("name");
                double marks = rs.getDouble("marks");
                
                System.out.printf("%-5d %-20s %-10.2f%n", id, name, marks);
            }
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

**Output:**
```
ID    Name                 Marks     
----------------------------------------
101   Alice                85.50     
102   Bob                  78.00     
```

---

## LAB 11: JDBC - Update Record

### Task: Update student marks using PreparedStatement

```java
import java.sql.*;

public class JDBCUpdate {
    static final String DB_URL = "jdbc:mysql://localhost:3306/testdb";
    static final String USER = "root";
    static final String PASS = "password";
    
    public static void main(String[] args) {
        String sql = "UPDATE students SET marks = ? WHERE id = ?";
        
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASS);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            
            pstmt.setDouble(1, 92.5);  // New marks
            pstmt.setInt(2, 101);      // Student ID
            
            int rows = pstmt.executeUpdate();
            System.out.println(rows + " row(s) updated");
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

---

## LAB 12: JDBC - Transaction Management

### Task: Transfer money between accounts with rollback

```java
import java.sql.*;

public class JDBCTransaction {
    static final String DB_URL = "jdbc:mysql://localhost:3306/testdb";
    static final String USER = "root";
    static final String PASS = "password";
    
    public static void main(String[] args) {
        Connection conn = null;
        
        try {
            conn = DriverManager.getConnection(DB_URL, USER, PASS);
            conn.setAutoCommit(false);  // Start transaction
            
            Statement stmt = conn.createStatement();
            
            // Deduct from account 1
            stmt.executeUpdate("UPDATE accounts SET balance = balance - 500 WHERE id = 1");
            
            // Add to account 2
            stmt.executeUpdate("UPDATE accounts SET balance = balance + 500 WHERE id = 2");
            
            conn.commit();  // Commit transaction
            System.out.println("Transaction successful");
            
        } catch (SQLException e) {
            if (conn != null) {
                try {
                    conn.rollback();  // Rollback on error
                    System.out.println("Transaction rolled back");
                } catch (SQLException ex) {
                    ex.printStackTrace();
                }
            }
            e.printStackTrace();
        } finally {
            try {
                if (conn != null) conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}
```

---

## LAB 13: Lambda & Streams - Filter & Map

### Task: Process employee list using streams

```java
import java.util.*;
import java.util.stream.*;

class EmployeeStream {
    int id;
    String name;
    double salary;
    
    public EmployeeStream(int id, String name, double salary) {
        this.id = id;
        this.name = name;
        this.salary = salary;
    }
    
    @Override
    public String toString() {
        return name + " ($" + salary + ")";
    }
}

public class StreamOperations {
    public static void main(String[] args) {
        List<EmployeeStream> emps = Arrays.asList(
            new EmployeeStream(1, "Alice", 50000),
            new EmployeeStream(2, "Bob", 60000),
            new EmployeeStream(3, "Charlie", 45000),
            new EmployeeStream(4, "David", 70000)
        );
        
        // Filter employees with salary > 50000
        System.out.println("High earners (>50000):");
        emps.stream()
            .filter(e -> e.salary > 50000)
            .forEach(System.out::println);
        
        // Map to names only
        System.out.println("\nAll names:");
        List<String> names = emps.stream()
            .map(e -> e.name)
            .collect(Collectors.toList());
        System.out.println(names);
        
        // Calculate total salary
        double total = emps.stream()
            .mapToDouble(e -> e.salary)
            .sum();
        System.out.println("\nTotal salary: $" + total);
        
        // Find highest paid
        Optional<EmployeeStream> highest = emps.stream()
            .max((e1, e2) -> Double.compare(e1.salary, e2.salary));
        highest.ifPresent(e -> System.out.println("\nHighest paid: " + e));
    }
}
```

---

## LAB 14: Exception Handling - Custom Exception

### Task: Create and throw custom exception

```java
class InsufficientFundsException extends Exception {
    public InsufficientFundsException(String message) {
        super(message);
    }
}

class Account {
    private double balance;
    
    public Account(double balance) {
        this.balance = balance;
    }
    
    public void withdraw(double amount) throws InsufficientFundsException {
        if (amount > balance) {
            throw new InsufficientFundsException(
                "Cannot withdraw $" + amount + ", balance: $" + balance);
        }
        balance -= amount;
        System.out.println("Withdrawn: $" + amount + ", Remaining: $" + balance);
    }
}

public class CustomExceptionDemo {
    public static void main(String[] args) {
        Account acc = new Account(1000);
        
        try {
            acc.withdraw(500);   // Success
            acc.withdraw(700);   // Exception
        } catch (InsufficientFundsException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
```

---

## LAB 15: Interface & Polymorphism

### Task: Demonstrate interface, polymorphism, and method overriding

```java
interface Shape {
    double area();
    double perimeter();
}

class Circle implements Shape {
    private double radius;
    
    public Circle(double radius) {
        this.radius = radius;
    }
    
    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
    
    @Override
    public double perimeter() {
        return 2 * Math.PI * radius;
    }
}

class Rectangle implements Shape {
    private double length, width;
    
    public Rectangle(double length, double width) {
        this.length = length;
        this.width = width;
    }
    
    @Override
    public double area() {
        return length * width;
    }
    
    @Override
    public double perimeter() {
        return 2 * (length + width);
    }
}

public class PolymorphismDemo {
    public static void printShapeInfo(Shape shape) {
        System.out.println("Area: " + String.format("%.2f", shape.area()));
        System.out.println("Perimeter: " + String.format("%.2f", shape.perimeter()));
    }
    
    public static void main(String[] args) {
        Shape circle = new Circle(5);
        Shape rectangle = new Rectangle(4, 6);
        
        System.out.println("Circle:");
        printShapeInfo(circle);
        
        System.out.println("\nRectangle:");
        printShapeInfo(rectangle);
    }
}
```

**Output:**
```
Circle:
Area: 78.54
Perimeter: 31.42

Rectangle:
Area: 24.00
Perimeter: 20.00
```

---

## Summary of Labs

| Lab | Topic | Key Concepts |
|-----|-------|--------------|
| 1-2 | File IO | BufferedReader/Writer, try-with-resources |
| 3 | Serialization | Serializable, transient |
| 4-6 | Collections | ArrayList, HashMap, TreeSet |
| 7-8 | Concurrency | wait/notify, synchronized |
| 9-12 | JDBC | PreparedStatement, transactions |
| 13 | Streams | filter, map, collect |
| 14 | Exceptions | Custom exceptions |
| 15 | OOP | Interface, polymorphism |

---

**End of Lab Exercise Solutions**
