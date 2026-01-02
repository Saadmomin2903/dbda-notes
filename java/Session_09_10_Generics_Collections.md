# Session 9-10 – Generics & Collections

**Topics Covered:** Generics (Type Parameters, Bounds, Wildcards), Collection Framework Hierarchy, List, Set, Map, Queue, Iterator

---

## PART 1: GENERICS

## 1. Why Generics?

### Problem Before Generics (Java 1.4)

```java
// Without generics - type safety issues
List list = new ArrayList();
list.add("String");
list.add(10);        // Allowed! (runtime error later)
list.add(new Date());

// Type casting required
String s = (String) list.get(0);  // OK
String s2 = (String) list.get(1); // ClassCastException at runtime!
```

### Solution: Generics (Java 5+)

```java
// With generics - compile-time type safety
List<String> list = new ArrayList<>();
list.add("String");
// list.add(10);     // Compile error!
// list.add(new Date()); // Compile error!

// No type casting needed
String s = list.get(0);  // No cast required
```

⭐ **Exam Fact:** Generics provide **compile-time type safety** and eliminate **type casting**.

---

## 2. Generic Class

```java
// Generic class with type parameter T
public class Box<T> {
    private T value;
    
    public void set(T value) {
        this.value = value;
    }
    
    public T get() {
        return value;
    }
}

// Usage
Box<Integer> intBox = new Box<>();
intBox.set(10);
int value = intBox.get();  // No casting

Box<String> strBox = new Box<>();
strBox.set("Hello");
String s = strBox.get();
```

### Multiple Type Parameters

```java
public class Pair<K, V> {
    private K key;
    private V value;
    
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }
    
    public K getKey() { return key; }
    public V getValue() { return value; }
}

// Usage
Pair<String, Integer> pair = new Pair<>("Age", 25);
String key = pair.getKey();
Integer value = pair.getValue();
```

---

## 3. Generic Method

```java
public class Util {
    // Generic method
    public static <T> void printArray(T[] array) {
        for (T element : array) {
            System.out.print(element + " ");
        }
        System.out.println();
    }
    
    // Generic method with return type
    public static <T> T getFirst(T[] array) {
        return array.length > 0 ? array[0] : null;
    }
}

// Usage
Integer[] intArray = {1, 2, 3, 4, 5};
String[] strArray = {"A", "B", "C"};

Util.printArray(intArray);  // 1 2 3 4 5
Util.printArray(strArray);  // A B C

Integer first = Util.getFirst(intArray);  // Type inference
```

---

## 4. Bounded Type Parameters

### Upper Bound (extends)

```java
// T must be Number or its subclass
public class NumberBox<T extends Number> {
    private T value;
    
    public void set(T value) {
        this.value = value;
    }
    
    public double getDoubleValue() {
        return value.doubleValue();  // Can call Number methods
    }
}

// Usage
NumberBox<Integer> intBox = new NumberBox<>();  // OK
NumberBox<Double> doubleBox = new NumberBox<>();  // OK
// NumberBox<String> strBox = new NumberBox<>();  // ERROR: String not a Number
```

### Multiple Bounds

```java
// T must extend Number AND implement Comparable
public class SortedBox<T extends Number & Comparable<T>> {
    private T value;
    
    public boolean isGreaterThan(T other) {
        return value.compareTo(other) > 0;  // Can use Comparable methods
    }
}
```

⭐ **Exam Fact:** In multiple bounds, **class must come first**, then interfaces.
```java
<T extends Number & Comparable>  // OK
<T extends Comparable & Number>  // ERROR
```

---

## 5. Wildcards

### Unbounded Wildcard (?)

```java
public static void printList(List<?> list) {
    for (Object obj : list) {
        System.out.println(obj);
    }
}

// Can accept any type
List<Integer> intList = Arrays.asList(1, 2, 3);
List<String> strList = Arrays.asList("A", "B", "C");
printList(intList);
printList(strList);
```

### Upper Bounded Wildcard (? extends)

```java
// Accepts Number or any subclass
public static double sum(List<? extends Number> list) {
    double total = 0;
    for (Number num : list) {
        total += num.doubleValue();
    }
    return total;
}

// Usage
List<Integer> intList = Arrays.asList(1, 2, 3);
List<Double> doubleList = Arrays.asList(1.5, 2.5, 3.5);
sum(intList);      // OK
sum(doubleList);   // OK
```

⚠️ **Cannot ADD** to upper bounded wildcard list:
```java
List<? extends Number> list = new ArrayList<Integer>();
// list.add(10);  // ERROR: Can't add to ? extends
// list.add(10.5); // ERROR
Number n = list.get(0);  // OK: Can read as Number
```

### Lower Bounded Wildcard (? super)

```java
// Accepts Integer or any superclass
public static void addNumbers(List<? super Integer> list) {
    list.add(10);   // OK: Can add Integer
    list.add(20);   // OK
    // list.add(10.5);  // ERROR: Double not guaranteed to be accepted
}

// Usage
List<Integer> intList = new ArrayList<>();
List<Number> numList = new ArrayList<>();
List<Object> objList = new ArrayList<>();

addNumbers(intList);  // OK
addNumbers(numList);  // OK
addNumbers(objList);  // OK
```

### PECS Principle (Producer Extends, Consumer Super)

| Use Case | Wildcard | Mnemonic |
|----------|----------|----------|
| **Reading** from structure | `<? extends T>` | Producer Extends |
| **Writing** to structure | `<? super T>` | Consumer Super |

```java
// Producer (read from source)
public static <T> void copy(List<? super T> dest, List<? extends T> src) {
    for (T item : src) {
        dest.add(item);
    }
}
```

---

## 6. Type Erasure

### What is Type Erasure?
Generics exist **only at compile time**. At runtime, generic type information is **erased** (replaced with Object or bound type).

```java
// Compile time
List<String> stringList = new ArrayList<>();
List<Integer> intList = new ArrayList<>();

// Runtime (after type erasure)
List stringList = new ArrayList();  // Both become raw List
List intList = new ArrayList();

// This is why:
System.out.println(stringList.getClass() == intList.getClass());  // true
```

### Type Erasure Examples

```java
// Before erasure
public class Box<T> {
    private T value;
    public T get() { return value; }
}

// After erasure
public class Box {
    private Object value;  // T → Object
    public Object get() { return value; }
}

// With bound
public class NumberBox<T extends Number> {
    private T value;
}

// After erasure
public class NumberBox {
    private Number value;  // T → Number (bound type)
}
```

⚠️ **Cannot do** because of type erasure:
```java
// ERROR: Cannot create generic array
T[] array = new T[10];

// ERROR: Cannot use instanceof with generics
if (obj instanceof List<String>) { }

// ERROR: Cannot create instance of type parameter
T obj = new T();

// ERROR: Cannot use in static context
class Box<T> {
    static T value;  // ERROR
}
```

---

## PART 2: COLLECTION FRAMEWORK

## 7. Collection Hierarchy

```
                    Iterable
                       |
                  Collection
                       |
        ┌──────────────┼──────────────┐
        |              |              |
      List           Set            Queue
        |              |              |
    ArrayList      HashSet      PriorityQueue
    LinkedList     TreeSet       ArrayDeque
    Vector      LinkedHashSet   LinkedList
    Stack
    
    
    (Separate hierarchy)
    
                     Map
                      |
            ┌─────────┼─────────┐
        HashMap   TreeMap  LinkedHashMap
        Hashtable
        Properties
```

### Key Interfaces

| Interface | Ordered | Sorted | Duplicates | Null |
|-----------|---------|--------|------------|------|
| **List** | ✅ | ❌ | ✅ | ✅ |
| **Set** | ❌ | ❌ | ❌ | Depends |
| **Queue** | ✅ | Depends | ✅ | Depends |
| **Map** | ❌ | ❌ | Keys: ❌, Values: ✅ | Depends |

---

## 8. List Interface

### ArrayList

```java
List<String> list = new ArrayList<>();

// Add elements
list.add("A");           // [A]
list.add("B");           // [A, B]
list.add(1, "C");        // [A, C, B] - insert at index
list.addAll(Arrays.asList("D", "E"));  // [A, C, B, D, E]

// Access elements
String first = list.get(0);       // A
String last = list.get(list.size() - 1);  // E

// Modify
list.set(1, "Z");        // [A, Z, B, D, E]

// Remove
list.remove(0);          // [Z, B, D, E] - by index
list.remove("B");        // [Z, D, E] - by object

// Search
boolean contains = list.contains("D");  // true
int index = list.indexOf("D");          // 1

// Size
int size = list.size();
boolean empty = list.isEmpty();

// Iterate
for (String s : list) {
    System.out.println(s);
}

// Clear
list.clear();
```

**Characteristics:**
- **Resizable array** implementation
- **Fast random access** O(1)
- **Slow insert/delete** O(n) in middle
- **Allows duplicates and null**
- **Not synchronized** (not thread-safe)

### LinkedList

```java
LinkedList<String> list = new LinkedList<>();

// All List operations
list.add("A");
list.get(0);

// Additional Queue operations
list.addFirst("First");
list.addLast("Last");
list.removeFirst();
list.removeLast();
list.getFirst();
list.getLast();

// Deque operations
list.push("Top");         // Stack-like
String top = list.pop();
```

**Characteristics:**
- **Doubly-linked list** implementation
- **Slow random access** O(n)
- **Fast insert/delete** O(1) at ends
- **Implements List and Deque**

### Vector (Legacy, Thread-Safe)

```java
Vector<Integer> vector = new Vector<>();
vector.add(10);
vector.add(20);

// Synchronized (thread-safe but slower)
```

**Characteristics:**
- **Synchronized** (thread-safe)
- **Slower** than ArrayList
- **Legacy class** (prefer ArrayList + synchronization if needed)

### ArrayList vs LinkedList vs Vector

| Feature | ArrayList | LinkedList | Vector |
|---------|-----------|------------|--------|
| **Structure** | Dynamic array | Doubly-linked list | Dynamic array |
| **Random Access** | O(1) | O(n) | O(1) |
| **Insert/Delete (middle)** | O(n) | O(1) | O(n) |
| **Insert/Delete (end)** | O(1) amortized | O(1) | O(1) |
| **Memory** | Less overhead | More (node objects) | Less |
| **Thread-Safe** | ❌ | ❌ | ✅ |
| **Use Case** | Frequent access | Frequent insert/delete | Legacy, thread-safe |

---

## 9. Set Interface

### HashSet

```java
Set<Integer> set = new HashSet<>();

// Add
set.add(1);
set.add(2);
set.add(1);  // Ignored (duplicate)
System.out.println(set);  // [1, 2] - no duplicates

// Contains
boolean exists = set.contains(1);  // true

// Remove
set.remove(1);

// Size
int size = set.size();

// Iterate (order not guaranteed)
for (Integer num : set) {
    System.out.println(num);
}
```

**Characteristics:**
- **Hash table** implementation
- **No duplicates**
- **Unordered** (no guaranteed order)
- **Allows one null**
- **Fast operations** O(1) add, remove, contains
- **Not synchronized**

### TreeSet

```java
Set<Integer> set = new TreeSet<>();
set.add(5);
set.add(1);
set.add(3);
System.out.println(set);  // [1, 3, 5] - sorted

// Custom comparator
Set<String> set2 = new TreeSet<>(Comparator.reverseOrder());
set2.add("C");
set2.add("A");
set2.add("B");
System.out.println(set2);  // [C, B, A]
```

**Characteristics:**
- **Red-Black tree** implementation
- **Sorted** (natural order or comparator)
- **No duplicates**
- **No null** (NullPointerException)
- **Operations** O(log n)

### LinkedHashSet

```java
Set<String> set = new LinkedHashSet<>();
set.add("C");
set.add("A");
set.add("B");
System.out.println(set);  // [C, A, B] - insertion order maintained
```

**Characteristics:**
- **Hash table + Linked list**
- **Maintains insertion order**
- **No duplicates**
- **Allows one null**
- **Slightly slower** than HashSet

### HashSet vs TreeSet vs LinkedHashSet

| Feature | HashSet | TreeSet | LinkedHashSet |
|---------|---------|---------|---------------|
| **Order** | Random | Sorted | Insertion order |
| **Performance** | O(1) | O(log n) | O(1) |
| **Null** | 1 null allowed | No null | 1 null allowed |
| **Implementation** | Hash table | Red-Black tree | Hash + Linked list |
| **Use Case** | Fast lookup | Sorted data | Ordered iteration |

---

## 10. Map Interface

### HashMap

```java
Map<String, Integer> map = new HashMap<>();

// Put
map.put("Alice", 25);
map.put("Bob", 30);
map.put("Charlie", 35);
map.put("Alice", 26);  // Updates value for existing key

// Get
Integer age = map.get("Alice");  // 26
Integer unknown = map.get("David");  // null

// getOrDefault
int age2 = map.getOrDefault("David", 0);  // 0

// Contains
boolean hasKey = map.containsKey("Alice");      // true
boolean hasValue = map.containsValue(30);       // true

// Remove
map.remove("Bob");

// Size
int size = map.size();

// Iterate - Method 1 (entrySet)
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + " = " + entry.getValue());
}

// Iterate - Method 2 (keySet)
for (String key : map.keySet()) {
    System.out.println(key + " = " + map.get(key));
}

// Iterate - Method 3 (values)
for (Integer value : map.values()) {
    System.out.println(value);
}

// Java 8+ forEach
map.forEach((key, value) -> System.out.println(key + " = " + value));
```

**Characteristics:**
- **Hash table** implementation
- **Key-value pairs**
- **No duplicate keys** (values can duplicate)
- **Allows one null key and multiple null values**
- **Unordered**
- **O(1)** operations
- **Not synchronized**

### TreeMap

```java
Map<String, Integer> map = new TreeMap<>();
map.put("C", 3);
map.put("A", 1);
map.put("B", 2);

// Iteration in sorted key order
for (String key : map.keySet()) {
    System.out.println(key);  // A, B, C
}

// Additional methods
String firstKey = ((TreeMap<String, Integer>) map).firstKey();  // A
String lastKey = ((TreeMap<String, Integer>) map).lastKey();    // C
```

**Characteristics:**
- **Red-Black tree** implementation
- **Sorted by keys**
- **No null key** (null values allowed)
- **O(log n)** operations

### LinkedHashMap

```java
Map<String, Integer> map = new LinkedHashMap<>();
map.put("C", 3);
map.put("A", 1);
map.put("B", 2);

// Iteration maintains insertion order
for (String key : map.keySet()) {
    System.out.println(key);  // C, A, B
}
```

### Hashtable (Legacy)

```java
Hashtable<String, Integer> table = new Hashtable<>();
table.put("A", 1);
// table.put(null, 1);  // NullPointerException
// table.put("B", null);  // NullPointerException
```

**Characteristics:**
- **Synchronized** (thread-safe)
- **No null** key or value
- **Legacy** (prefer HashMap or ConcurrentHashMap)

### HashMap vs TreeMap vs LinkedHashMap vs Hashtable

| Feature | HashMap | TreeMap | LinkedHashMap | Hashtable |
|---------|---------|---------|---------------|-----------|
| **Order** | Random | Sorted by key | Insertion | Random |
| **Performance** | O(1) | O(log n) | O(1) | O(1) |
| **Null Key** | 1 allowed | Not allowed | 1 allowed | Not allowed |
| **Null Value** | Allowed | Allowed | Allowed | Not allowed |
| **Thread-Safe** | ❌ | ❌ | ❌ | ✅ |
| **Use Case** | General purpose | Sorted keys | Order matters | Legacy |

---

## 11. Queue Interface

### PriorityQueue

```java
Queue<Integer> pq = new PriorityQueue<>();  // Min-heap by default

// Add
pq.offer(5);
pq.offer(1);
pq.offer(3);

// Peek (doesn't remove)
Integer min = pq.peek();  // 1

// Poll (removes and returns)
Integer removed = pq.poll();  // 1
System.out.println(pq);  // [3, 5]

// Max-heap
Queue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
maxHeap.offer(5);
maxHeap.offer(1);
maxHeap.offer(3);
System.out.println(maxHeap.poll());  // 5
```

**Characteristics:**
- **Heap** implementation
- **Priority-based** (min-heap by default)
- **Not ordered** for iteration
- **No null**

---

## 12. Iterator & ListIterator

### Iterator

```java
List<String> list = new ArrayList<>(Arrays.asList("A", "B", "C"));
Iterator<String> it = list.iterator();

while (it.hasNext()) {
    String s = it.next();
    if (s.equals("B")) {
        it.remove();  // Safe removal during iteration
    }
}
System.out.println(list);  // [A, C]
```

### ListIterator (Bidirectional)

```java
List<String> list = new ArrayList<>(Arrays.asList("A", "B", "C"));
ListIterator<String> lit = list.listIterator();

// Forward
while (lit.hasNext()) {
    System.out.println(lit.next());
}

// Backward
while (lit.hasPrevious()) {
    System.out.println(lit.previous());
}

// Modify during iteration
ListIterator<String> lit2 = list.listIterator();
while (lit2.hasNext()) {
    String s = lit2.next();
    if (s.equals("B")) {
        lit2.set("Z");  // Replace
    }
}
```

⚠️ **ConcurrentModificationException:**
```java
List<String> list = new ArrayList<>(Arrays.asList("A", "B", "C"));

// WRONG
for (String s : list) {
    list.remove(s);  // ConcurrentModificationException
}

// CORRECT
Iterator<String> it = list.iterator();
while (it.hasNext()) {
    it.next();
    it.remove();  // Use iterator.remove()
}
```

---

## 🔥 Top MCQs for Session 9-10

### MCQ 1: Generics Type Safety
**Q:** Main benefit of generics?
1. Performance
2. Compile-time type safety
3. Runtime optimization
4. Memory efficiency

**Answer:** 2. Compile-time type safety

---

### MCQ 2: Type Erasure
**Q:** After type erasure, `List<String>` becomes:
1. List<Object>
2. List
3. String[]
4. Compile error

**Answer:** 2. List (raw type)

---

### MCQ 3: Wildcard
**Q:** Which allows adding elements?
1. List<? extends Number>
2. List<? super Integer>
3. List<?>
4. None

**Answer:** 2. List<? super Integer>

---

### MCQ 4: HashMap Null
**Q:** HashMap allows:
1. No null key or value
2. One null key, multiple null values
3. Multiple null keys
4. One null key and value

**Answer:** 2. One null key, multiple null values

---

### MCQ 5: TreeSet Null
**Q:** TreeSet allows null?
1. Yes
2. No

**Answer:** 2. No (NullPointerException)

---

### MCQ 6: ArrayList vs LinkedList
**Q:** Which is faster for random access?
1. ArrayList
2. LinkedList

**Answer:** 1. ArrayList (O(1) vs O(n))

---

### MCQ 7: HashSet Order
**Q:** HashSet maintains:
1. Insertion order
2. Sorted order
3. No guaranteed order
4. Reverse order

**Answer:** 3. No guaranteed order

---

### MCQ 8: ConcurrentModificationException
**Q:** When does it occur?
```java
for (String s : list) {
    list.remove(s);
}
```
1. Compile time
2. Runtime
3. Never
4. Depends

**Answer:** 2. Runtime

---

### MCQ 9: TreeMap Sorting
**Q:** TreeMap sorts by:
1. Values
2. Keys
3. Both
4. Random

**Answer:** 2. Keys

---

### MCQ 10: Generic Array
**Q:** Is this valid?
```java
T[] array = new T[10];
```
1. Yes
2. No

**Answer:** 2. No (type erasure prevents generic array creation)

---

## ⚠️ Common Mistakes

1. **Modifying collection during iteration** (use iterator.remove())
2. **Expecting order from HashSet/HashMap**
3. **Adding null to TreeSet/TreeMap**
4. **Not overriding equals/hashCode** for custom objects in HashMap/HashSet
5. **Using == instead of equals()** for object comparison
6. **Type erasure limitations** (can't create generic arrays)
7. **Confusing ? extends vs ? super** wildcards

---

## ⭐ One-liner Exam Facts

1. Generics provide **compile-time type safety**
2. Type erasure occurs at **runtime**
3. **Cannot create generic arrays**
4. PECS: **Producer Extends, Consumer Super**
5. HashMap allows **one null key**
6. TreeSet/TreeMap **don't allow null**
7. HashSet maintains **no order**
8. LinkedHashSet maintains **insertion order**
9. TreeSet maintains **sorted order**
10. ArrayList is **fast for access**, LinkedList for **insert/delete**
11. Use **iterator.remove()** during iteration
12. TreeMap sorted by **keys**, not values
13. Hashtable is **synchronized**, HashMap is not
14. Vector is **legacy and synchronized**
15. PriorityQueue is **min-heap by default**

---

**End of Session 9-10**
