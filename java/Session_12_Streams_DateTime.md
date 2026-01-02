# Session 12 – Streams & Date Time API

**Topics Covered:** Streams vs Collections, Stream Lifecycle, Intermediate & Terminal Operations, map/filter/reduce/flatMap, Primitive Streams, Java 8/17 Date Time API

---

## PART 1: STREAMS

## 1. Stream vs Collection

| Aspect | Collection | Stream |
|--------|------------|--------|
| **Purpose** | Store data | Process data |
| **Nature** | Data structure | Pipeline of operations |
| **Modification** | Can modify elements | Cannot modify source |
| **Iteration** | External (for loop, iterator) | Internal (forEach) |
| **Consumption** | Reusable | One-time use |
| **Evaluation** | Eager | Lazy (intermediate ops) |
| **Size** | Finite | Can be infinite |

⭐ **Exam Fact:** Streams are **lazily evaluated** - intermediate operations don't execute until terminal operation is called.

---

## 2. Creating Streams

```java
// 1. From Collection
List<Integer> list = Arrays.asList(1, 2, 3, 4, 5);
Stream<Integer> stream1 = list.stream();

// 2. From Array
Integer[] arr = {1, 2, 3, 4, 5};
Stream<Integer> stream2 = Arrays.stream(arr);

// 3. Using Stream.of()
Stream<Integer> stream3 = Stream.of(1, 2, 3, 4, 5);

// 4. Infinite Streams
Stream<Integer> infinite1 = Stream.iterate(0, n -> n + 1);  // 0, 1, 2, 3...
Stream<Double> infinite2 = Stream.generate(Math::random);

// 5. Empty Stream
Stream<String> empty = Stream.empty();

// 6. Range (IntStream)
IntStream range = IntStream.range(1, 10);  // 1 to 9
IntStream rangeClosed = IntStream.rangeClosed(1, 10);  // 1 to 10
```

---

## 3. Intermediate Operations (Lazy Evaluation)

### filter() - Select elements

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

List<Integer> evenNumbers = numbers.stream()
    .filter(n -> n % 2 == 0)  // Keep only even numbers
    .collect(Collectors.toList());
// Result: [2, 4, 6, 8, 10]
```

### map() - Transform elements

```java
List<String> words = Arrays.asList("java", "python", "c++");

List<Integer> lengths = words.stream()
    .map(s -> s.length())  // Transform string to length
    .collect(Collectors.toList());
// Result: [4, 6, 3]

List<String> upperCase = words.stream()
    .map(String::toUpperCase)
    .collect(Collectors.toList());
// Result: [JAVA, PYTHON, C++]
```

### flatMap() - Flatten nested structures

```java
List<List<Integer>> nested = Arrays.asList(
    Arrays.asList(1, 2),
    Arrays.asList(3, 4),
    Arrays.asList(5, 6)
);

List<Integer> flattened = nested.stream()
    .flatMap(list -> list.stream())  // Flatten
    .collect(Collectors.toList());
// Result: [1, 2, 3, 4, 5, 6]

// Practical example
List<String> sentences = Arrays.asList("Hello World", "Java Streams");
List<String> words = sentences.stream()
    .flatMap(sentence -> Arrays.stream(sentence.split(" ")))
    .collect(Collectors.toList());
// Result: [Hello, World, Java, Streams]
```

### distinct() - Remove duplicates

```java
List<Integer> numbers = Arrays.asList(1, 2, 2, 3, 3, 3, 4, 5, 5);

List<Integer> unique = numbers.stream()
    .distinct()
    .collect(Collectors.toList());
// Result: [1, 2, 3, 4, 5]
```

### sorted() - Sort elements

```java
List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9);

// Natural order
List<Integer> sorted = numbers.stream()
    .sorted()
    .collect(Collectors.toList());
// Result: [1, 2, 5, 8, 9]

// Custom comparator
List<Integer> reversed = numbers.stream()
    .sorted(Comparator.reverseOrder())
    .collect(Collectors.toList());
// Result: [9, 8, 5, 2, 1]
```

### limit() & skip()

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// First 5 elements
List<Integer> first5 = numbers.stream()
    .limit(5)
    .collect(Collectors.toList());
// Result: [1, 2, 3, 4, 5]

// Skip first 5, take rest
List<Integer> after5 = numbers.stream()
    .skip(5)
    .collect(Collectors.toList());
// Result: [6, 7, 8, 9, 10]

// Pagination: Skip 5, take 3
List<Integer> page2 = numbers.stream()
    .skip(5)
    .limit(3)
    .collect(Collectors.toList());
// Result: [6, 7, 8]
```

---

## 4. Terminal Operations (Eager Evaluation)

### forEach() - Iterate

```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
names.stream().forEach(System.out::println);
```

### collect() - Accumulate to collection

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

// To List
List<Integer> list = numbers.stream().collect(Collectors.toList());

// To Set
Set<Integer> set = numbers.stream().collect(Collectors.toSet());

// To Map
Map<Integer, String> map = numbers.stream()
    .collect(Collectors.toMap(n -> n, n -> "Number: " + n));

// Joining strings
String joined = Arrays.asList("A", "B", "C").stream()
    .collect(Collectors.joining(", "));
// Result: "A, B, C"
```

### count() - Count elements

```java
long count = Stream.of(1, 2, 3, 4, 5).count();  // 5
```

### reduce() - Reduce to single value

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

// Sum
int sum = numbers.stream()
    .reduce(0, (a, b) -> a + b);  // 15

// Product
int product = numbers.stream()
    .reduce(1, (a, b) -> a * b);  // 120

// Max
Optional<Integer> max = numbers.stream()
    .reduce((a, b) -> a > b ? a : b);
```

### min() & max()

```java
List<Integer> numbers = Arrays.asList(5, 2, 8, 1, 9);

Optional<Integer> min = numbers.stream().min(Integer::compareTo);  // 1
Optional<Integer> max = numbers.stream().max(Integer::compareTo);  // 9
```

### anyMatch(), allMatch(), noneMatch()

```java
List<Integer> numbers = Arrays.asList(2, 4, 6, 8, 10);

boolean hasEven = numbers.stream().anyMatch(n -> n % 2 == 0);     // true
boolean allEven = numbers.stream().allMatch(n -> n % 2 == 0);     // true
boolean noneOdd = numbers.stream().noneMatch(n -> n % 2 != 0);    // true
```

### findFirst() & findAny()

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

Optional<Integer> first = numbers.stream().findFirst();  // Optional[1]
Optional<Integer> any = numbers.stream().findAny();      // Optional[1] (may vary in parallel)
```

---

## 5. Stream Cannot Be Reused

```java
Stream<Integer> stream = Stream.of(1, 2, 3);

stream.forEach(System.out::println);  // OK
stream.forEach(System.out::println);  // IllegalStateException: stream has already been operated upon or closed
```

⭐ **Exam Fact:** Streams are **single-use**. Once a terminal operation is called, stream is consumed.

---

## 6. Lazy Evaluation Example

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);

Stream<Integer> stream = numbers.stream()
    .filter(n -> {
        System.out.println("Filter: " + n);
        return n % 2 == 0;
    })
    .map(n -> {
        System.out.println("Map: " + n);
        return n * 2;
    });

// Nothing printed yet! (lazy)

List<Integer> result = stream.collect(Collectors.toList());
// Now prints:
// Filter: 1
// Filter: 2
// Map: 2
// Filter: 3
// Filter: 4
// Map: 4
// Filter: 5
```

---

## PART 2: DATE TIME API

## 1. Problems with Old Date API

```java
// Old (before Java 8) - AVOID
Date date = new Date();  // Mutable, not thread-safe
SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");  // Not thread-safe
```

**Problems:**
- Mutable (not thread-safe)
- Poor API design
- Month starts from 0 (confusing)
- Timezone handling complex

---

## 2. New Date Time API (java.time)

### LocalDate - Date without time

```java
import java.time.LocalDate;

// Current date
LocalDate today = LocalDate.now();

// Specific date
LocalDate date = LocalDate.of(2024, 1, 15);
LocalDate date2 = LocalDate.of(2024, Month.JANUARY, 15);

// Parsing
LocalDate parsed = LocalDate.parse("2024-01-15");

// Get components
int year = date.getYear();           // 2024
int month = date.getMonthValue();    // 1
Month monthEnum = date.getMonth();   // JANUARY
int day = date.getDayOfMonth();      // 15
DayOfWeek dayOfWeek = date.getDayOfWeek();  // MONDAY, etc.

// Operations (returns new LocalDate - immutable)
LocalDate tomorrow = today.plusDays(1);
LocalDate nextWeek = today.plusWeeks(1);
LocalDate nextMonth = today.plusMonths(1);
LocalDate nextYear = today.plusYears(1);

LocalDate yesterday = today.minusDays(1);
```

### LocalTime - Time without date

```java
import java.time.LocalTime;

// Current time
LocalTime now = LocalTime.now();

// Specific time
LocalTime time = LocalTime.of(14, 30);        // 14:30
LocalTime time2 = LocalTime.of(14, 30, 45);   // 14:30:45

// Get components
int hour = time.getHour();      // 14
int minute = time.getMinute();  // 30
int second = time.getSecond();  // 0

// Operations
LocalTime later = time.plusHours(2);
LocalTime earlier = time.minusMinutes(30);
```

### LocalDateTime - Date + Time (no timezone)

```java
import java.time.LocalDateTime;

// Current date-time
LocalDateTime now = LocalDateTime.now();

// Specific date-time
LocalDateTime dt = LocalDateTime.of(2024, 1, 15, 14, 30);

// Combining Local Date and LocalTime
LocalDate date = LocalDate.of(2024, 1, 15);
LocalTime time = LocalTime.of(14, 30);
LocalDateTime combined = LocalDateTime.of(date, time);

// Operations
LocalDateTime future = now.plusDays(1).plusHours(2);
```

### Period - Date-based duration

```java
import java.time.Period;

LocalDate start = LocalDate.of(2023, 1, 1);
LocalDate end = LocalDate.of(2024, 3, 15);

Period period = Period.between(start, end);

int years = period.getYears();    // 1
int months = period.getMonths();  // 2
int days = period.getDays();      // 14

// Create period
Period twoMonths = Period.ofMonths(2);
Period oneWeek = Period.ofWeeks(1);
LocalDate future = LocalDate.now().plus(twoMonths);
```

### Duration - Time-based duration

```java
import java.time.Duration;

LocalTime start = LocalTime.of(10, 0);
LocalTime end = LocalTime.of(14, 30);

Duration duration = Duration.between(start, end);

long hours = duration.toHours();      // 4
long minutes = duration.toMinutes();  // 270
long seconds = duration.toSeconds();  // 16200

// Create duration
Duration fiveHours = Duration.ofHours(5);
LocalTime later = LocalTime.now().plus(fiveHours);
```

### Formatting & Parsing

```java
import java.time.format.DateTimeFormatter;

LocalDate date = LocalDate.now();

// Predefined formatters
String iso = date.format(DateTimeFormatter.ISO_DATE);  // 2024-01-15

// Custom formatter
DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MM-yyyy");
String formatted = date.format(formatter);  // 15-01-2024

// Parsing
LocalDate parsed = LocalDate.parse("15-01-2024", formatter);

// Common patterns
// dd-MM-yyyy → 15-01-2024
// MM/dd/yyyy → 01/15/2024
// yyyy-MM-dd HH:mm:ss → 2024-01-15 14:30:45
```

⭐ **Exam Fact:** New Date Time API classes are **immutable** and **thread-safe**.

---

## 🔥 Top MCQs for Session 12

**MCQ 1:** Can stream be reused?
- Answer: No (IllegalStateException)

**MCQ 2:** Which is terminal operation?
- Answer: collect, forEach, reduce (NOT filter, map, sorted)

**MCQ 3:** Streams are evaluated:
- Answer: Lazily (intermediate ops don't execute until terminal op)

**MCQ 4:** LocalDate is:
- Answer: Immutable and thread-safe

**MCQ 5:** Period is for:
- Answer: Date-based duration (years, months, days)

**MCQ 6:** What is output?
```java
Stream.of(1,2,3).filter(n -> n > 2).count();
```
- Answer: 1 (only 3 passes filter)

**MCQ 7:** flatMap() is used for:
- Answer: Flattening nested structures

**MCQ 8:** Duration measures:
- Answer: Time-based (hours, minutes, seconds)

**MCQ 9:** reduce() is:
- Answer: Terminal operation

**MCQ 10:** Old Date class is:
- Answer: Mutable (not thread-safe)

---

## ⚠️ Common Mistakes

1. **Reusing streams** after terminal operation
2. **Confusing intermediate vs terminal** operations
3. **Forgetting lazy evaluation** (intermediate ops don't run without terminal)
4. **Using old Date/Calendar** instead of new API
5. **Confusing Period** (date-based) **vs Duration** (time-based)
6. **Not handling** Optional from findFirst/findAny

---

## ⭐ One-liner Exam Facts

1. Streams are **lazily evaluated**
2. Stream **cannot be reused** (single-use)
3. **Intermediate**: filter, map, flatMap, sorted, distinct, limit, skip
4. **Terminal**: collect, forEach, reduce, count, anyMatch, findFirst
5. flatMap() **flattens** nested structures
6. New Date API classes **immutable** and **thread-safe**
7. **LocalDate** = date only, **LocalTime** = time only, **LocalDateTime** = both
8. **Period** for dates (years/months/days)
9. **Duration** for time (hours/minutes/seconds)
10. reduce() **accumulates** to single value

---

**End of Session 12**
