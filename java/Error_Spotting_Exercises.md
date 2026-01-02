# Error Spotting Exercises

**Total Problems:** 20  
**Skill:** Identifying compile-time and runtime errors  
**Instructions:** Find the error(s) in each code snippet

> ⚠️ **Critical Skill!** Exams often test ability to spot bugs at a glance.

---

## PROBLEM 1: Access Modifier
```java
class Test {
    private void display() {
        System.out.println("Private method");
    }
}
class Demo {
    public static void main(String[] args) {
        Test t = new Test();
        t.display();
    }
}
```
**Error:** _______________

---

## PROBLEM 2: Method Overloading
```java
class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    public double add(int a, int b) {
        return a + b + 0.0;
    }
}
```
**Error:** _______________

---

## PROBLEM 3: Constructor
```java
class Person {
    String name;
    public void Person(String name) {
        this.name = name;
    }
}
```
**Error:** _______________

---

## PROBLEM 4: Abstract Class
```java
abstract class Animal {
    abstract void sound();
}
Animal a = new Animal();
```
**Error:** _______________

---

## PROBLEM 5: Interface
```java
interface Printable {
    void print();
}
class Document implements Printable {
    // No print() implementation
}
```
**Error:** _______________

---

## PROBLEM 6: Exception Handling
```java
public void test() {
    FileReader fr = new FileReader("file.txt");
}
```
**Error:** _______________

---

## PROBLEM 7: Exception Order
```java
try {
    // code
} catch (Exception e) {
    // handle
} catch (IOException e) {
    // handle
}
```
**Error:** _______________

---

## PROBLEM 8: finally
```java
try {
    return;
} finally {
    System.out.println("Finally");
    // No return here
}
```
**Error:** _______________

---

## PROBLEM 9: Array
```java
int[] arr = new int[3];
arr[0] = 1;
arr[1] = 2;
arr[2] = 3;
arr[3] = 4;
```
**Error:** _______________

---

## PROBLEM 10: String
```java
String s;
s.concat("Hello");
System.out.println(s);
```
**Error:** _______________

---

## PROBLEM 11: Inheritance
```java
class A {
    private void show() {
        System.out.println("A");
    }
}
class B extends A {
    @Override
    private void show() {
        System.out.println("B");
    }
}
```
**Error:** _______________

---

## PROBLEM 12: Generic
```java
List<int> list = new ArrayList<int>();
```
**Error:** _______________

---

## PROBLEM 13: Collection
```java
Set<String> set = new HashSet<>();
set.add("A");
set.add("B");
set.add(0, "C");
```
**Error:** _______________

---

## PROBLEM 14: HashMap
```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
int value = map.get("B");
```
**Error:** _______________

---

## PROBLEM 15: Thread
```java
class MyThread extends Thread {
    public void start() {
        System.out.println("Custom start");
    }
}
MyThread t = new MyThread();
t.start();
```
**Error:** _______________

---

## PROBLEM 16: Lambda
```java
int x = 10;
Runnable r = () -> {
    x = 20;
    System.out.println(x);
};
```
**Error:** _______________

---

## PROBLEM 17: Stream
```java
List<String> list = Arrays.asList("A", "B", "C");
list.stream()
    .filter(s -> s.startsWith("A"))
    .map(s -> s.toLowerCase())
    // Missing terminal operation
String result = ???
```
**Error:** _______________

---

## PROBLEM 18: Serialization
```java
class Employee implements Serializable {
    String name;
    Address address;  // Address is NOT Serializable
}
```
**Error:** _______________

---

## PROBLEM 19: JDBC
```java
Connection conn = DriverManager.getConnection(url);
Statement stmt = conn.createStatement();
String username = request.getParameter("username");
String query = "SELECT * FROM users WHERE username = '" + username + "'";
ResultSet rs = stmt.executeQuery(query);
```
**Error:** _______________

---

## PROBLEM 20: Switch
```java
String day = "Monday";
switch(day) {
    case "Monday":
        System.out.println("Mon");
    case "Tuesday":
        System.out.println("Tue");
        break;
    default:
        System.out.println("Other");
}
```
**Error:** _______________

---

---

# ANSWER KEY

## PROBLEM 1: Access Modifier
**Error:** Cannot access private method from outside class  
**Fix:** Make `display()` public or call it from within Test class

## PROBLEM 2: Method Overloading
**Error:** Cannot overload based on return type alone  
**Fix:** Parameters must differ (name, number, or type)

## PROBLEM 3: Constructor
**Error:** Constructor has return type (void) - it's a method, not constructor  
**Fix:** Remove `void`: `public Person(String name) { }`

## PROBLEM 4: Abstract Class
**Error:** Cannot instantiate abstract class  
**Fix:** Create concrete subclass or make Animal non-abstract

## PROBLEM 5: Interface
**Error:** Document must implement all interface methods  
**Fix:** Add `public void print() { }` in Document class

## PROBLEM 6: Exception Handling
**Error:** FileNotFoundException (checked) not handled  
**Fix:** Add `throws IOException` or try-catch block

## PROBLEM 7: Exception Order
**Error:** IOException is unreachable (Exception already catches it)  
**Fix:** Put specific exception (IOException) BEFORE generic (Exception)

## PROBLEM 8: finally
**Error:** None! This is valid. finally executes even with return in try.  
**Trick Question:** Code compiles and runs fine

## PROBLEM 9: Array
**Error:** ArrayIndexOutOfBoundsException at runtime (index 3 invalid)  
**Fix:** Array size is 3, valid indices are 0-2

## PROBLEM 10: String
**Error:** Variable 's' might not have been initialized  
**Fix:** Initialize: `String s = "";` or `String s = null;`

## PROBLEM 11: Inheritance
**Error:** Cannot override private method (not visible to subclass)  
**Fix:** Make method protected or public, or remove @Override

## PROBLEM 12: Generic
**Error:** Cannot use primitive types with generics  
**Fix:** Use wrapper: `List<Integer> list = new ArrayList<>();`

## PROBLEM 13: Collection
**Error:** Set doesn't have `add(index, element)` method (only List)  
**Fix:** Use `set.add("C");` without index

## PROBLEM 14: HashMap
**Error:** get("B") returns null (not in map), cannot assign to  int  
**Fix:** Handle null: `Integer value = map.get("B");` or check first

## PROBLEM 15: Thread
**Error:** Logic error - overriding start() prevents thread from actually starting  
**Fix:** Override `run()` method, not `start()`

## PROBLEM 16: Lambda
**Error:** Local variable x must be effectively final  
**Fix:** Don't modify x inside lambda, or use instance variable

## PROBLEM 17: Stream
**Error:** No terminal operation - stream not executed  
**Fix:** Add: `.collect(Collectors.toList())` or `.findFirst()` etc.

## PROBLEM 18: Serialization
**Error:** NotSerializableException - Address must also be Serializable  
**Fix:** Make Address implement Serializable, or mark field transient

## PROBLEM 19: JDBC
**Error:** SQL injection vulnerability  
**Fix:** Use PreparedStatement with parameterized query

## PROBLEM 20: Switch
**Error:** Missing break in first case - falls through to print both  
**Fix:** Add `break;` after first println (or intentional fall-through comment)

---

## Category-wise Errors

### Compile-Time Errors (14)
- Problems: 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17

### Runtime Errors (3)
- Problems: 9 (ArrayIndexOutOfBoundsException)
- Problem 18 (NotSerializableException)

### Logic Errors (3)
- Problem 15 (Thread logic)
- Problem 19 (SQL injection - security)
- Problem 20 (Switch fall-through - logic)

---

## Error Type Distribution

| Error Type | Count | Problems |
|------------|-------|----------|
| OOP Concepts | 5 | 1, 3, 4, 5, 11 |
| Exception Handling | 3 | 6, 7, 18 |
| Collections/Generics | 4 | 12, 13, 14, 17 |
| Overloading/Overriding | 2 | 2, 11 |
| Initialization | 1 | 10 |
| Arrays | 1 | 9 |
| Concurrency | 2 | 15, 16 |
| JDBC | 1 | 19 |
| Control Flow | 1 | 20 |

---

## Scoring
- 18-20: Excellent error detection skills
- 15-17: Very good
- 12-14: Good understanding
- 9-11: Average, review concepts
- <9: Need more practice

**Your Score:** ___/20

---

**End of Error Spotting Exercises**
