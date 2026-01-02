# Sessions 13-14 – Concurrency & Multithreading

**Topics Covered:** Thread Basics, Creating Threads, Thread Lifecycle, Thread Methods, Synchronization, Race Conditions, Deadlock, Inter-thread Communication

---

## 1. What is Concurrency?

**Concurrency** = Multiple tasks making progress (appear to run simultaneously).  
**Parallelism** = Multiple tasks actually running at the same time (requires multiple cores).

### Why Concurrency?
- Better resource utilization
- Improved responsiveness (UI doesn't freeze)
- Faster execution for independent tasks
- Better system throughput

---

## 2. Process vs Thread

| Aspect | Process | Thread |
|--------|---------|--------|
| **Definition** | Independent program | Lightweight subprocess |
| **Memory** | Separate memory space | Shares process memory |
| **Communication** | IPC (complex) | Direct (shared memory) |
| **Creation** | Expensive | Lightweight |
| **Example** | Running app | Task within app |

⭐ **Exam Fact:** Threads share **heap** and **method area**, but each has its own **stack**.

---

## 3. Thread Lifecycle

```
     NEW
      ↓
   start()
      ↓
  RUNNABLE ←→ RUNNING
      ↓          ↓
   BLOCKED   WAITING/TIMED_WAITING
                  ↓
             TERMINATED
```

### States Explained

| State | Description | How to Enter |
|-------|-------------|--------------|
| **NEW** | Thread created but not started | `new Thread()` |
| **RUNNABLE** | Ready to run, waiting for CPU | `start()` called |
| **RUNNING** | Executing | Thread scheduler picks it |
| **BLOCKED** | Waiting for monitor lock | Trying to enter synchronized block |
| **WAITING** | Waiting indefinitely | `wait()`, `join()` |
| **TIMED_WAITING** | Waiting for specified time | `sleep(ms)`, `wait(ms)` |
| **TERMINATED** | Completed execution | `run()` method completes |

---

## 4. Creating Threads

### Method 1: Extend Thread Class

```java
class MyThread extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(Thread.currentThread().getName() + ": " + i);
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

// Usage
MyThread t1 = new MyThread();
MyThread t2 = new MyThread();
t1.start();  // Starts new thread
t2.start();
```

### Method 2: Implement Runnable Interface (Preferred)

```java
class MyRunnable implements Runnable {
    @Override
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(Thread.currentThread().getName() + ": " + i);
        }
    }
}

// Usage
Thread t1 = new Thread(new MyRunnable());
Thread t2 = new Thread(new MyRunnable());
t1.start();
t2.start();

// With lambda (Java 8+)
Thread t3 = new Thread(() -> {
    System.out.println("Lambda thread");
});
t3.start();
```

### Why Runnable is Preferred?

1. **Can extend another class** (Java has single inheritance)
2. **Separation of concerns** (task vs thread)
3. **Better for thread pools** (ExecutorService)
4. **More flexible**

⚠️ **Common Mistake:**
```java
Thread t = new Thread();
t.run();  // WRONG! Runs in current thread, doesn't start new thread
t.start();  // CORRECT! Starts new thread
```

---

## 5. Thread Methods

### start() vs run()

```java
Thread t = new Thread(() -> System.out.println("Running in: " + Thread.currentThread().getName()));

t.run();   // Runs in main thread: "Running in: main"
t.start(); // Runs in new thread: "Running in: Thread-0"
```

### join() - Wait for Thread Completion

```java
Thread t1 = new Thread(() -> {
    for (int i = 0; i < 5; i++) {
        System.out.println("T1: " + i);
    }
});

t1.start();
t1.join();  // Main thread waits for t1 to complete
System.out.println("T1 completed");
```

### sleep() - Pause Execution

```java
try {
    Thread.sleep(1000);  // Sleep for 1 second
} catch (InterruptedException e) {
    e.printStackTrace();
}
```

⭐ **Exam Fact:** `sleep()` throws **InterruptedException** (checked).

### getName() & setName()

```java
Thread t = new Thread(() -> System.out.println("Running"));
t.setName("MyWorker");
System.out.println(t.getName());  // MyWorker

// Current thread
String name = Thread.currentThread().getName();  // main
```

### Priority (1-10)

```java
Thread t = new Thread();
t.setPriority(Thread.MAX_PRIORITY);   // 10
t.setPriority(Thread.MIN_PRIORITY);   // 1
t.setPriority(Thread.NORM_PRIORITY);  // 5 (default)

int priority = t.getPriority();
```

⚠️ **Note:** Priority is just a **hint** to thread scheduler, not a guarantee.

### Daemon Threads

```java
Thread t = new Thread(() -> {
    while (true) {
        System.out.println("Background task");
    }
});
t.setDaemon(true);  // Make daemon BEFORE starting
t.start();

// JVM exits when only daemon threads remain
```

⭐ **Exam Fact:** **Daemon threads don't prevent JVM from exiting**.

---

## 6. Synchronization

### Problem: Race Condition

```java
class Counter {
    private int count = 0;
    
    // NOT thread-safe
    public void increment() {
        count++;  // 3 operations: read, increment, write (not atomic)
    }
    
    public int getCount() {
        return count;
    }
}

// Multiple threads
Counter counter = new Counter();
Thread t1 = new Thread(() -> {
    for (int i = 0; i < 1000; i++) counter.increment();
});
Thread t2 = new Thread(() -> {
    for (int i = 0; i < 1000; i++) counter.increment();
});

t1.start();
t2.start();
t1.join();
t2.join();

System.out.println(counter.getCount());  // Expected: 2000, Actual: Less (e.g., 1850)
```

### Solution 1: synchronized Method

```java
class Counter {
    private int count = 0;
    
    // Thread-safe
    public synchronized void increment() {
        count++;  // Only one thread can execute at a time
    }
    
    public synchronized int getCount() {
        return count;
    }
}

// Now guaranteed to be 2000
```

### Solution 2: synchronized Block

```java
class Counter {
    private int count = 0;
    private Object lock = new Object();
    
    public void increment() {
        synchronized(lock) {  // Lock on specific object
            count++;
        }
    }
    
    public void method() {
        synchronized(this) {  // Lock on current instance
            // Critical section
        }
    }
}
```

### synchronized on Different Objects

```java
class BankAccount {
    private int balance = 1000;
    
    public synchronized void withdraw(int amount) {
        if (balance >= amount) {
            balance -= amount;
        }
    }
    
    public synchronized void deposit(int amount) {
        balance += amount;
    }
}

// If two threads call withdraw() on SAME account → synchronized
// If two threads call withdraw() on DIFFERENT accounts → NOT synchronized (different locks)
```

⭐ **Exam Fact:** synchronized acquires lock on **object** (instance method) or **class** (static method).

---

## 7. Deadlock

### What is Deadlock?
Two or more threads waiting for each other to release locks, creating circular dependency.

### Classic Deadlock Example

```java
class Resource1 { }
class Resource2 { }

Resource1 r1 = new Resource1();
Resource2 r2 = new Resource2();

// Thread 1
Thread t1 = new Thread(() -> {
    synchronized(r1) {
        System.out.println("T1: Locked R1");
        try { Thread.sleep(100); } catch (InterruptedException e) { }
        
        synchronized(r2) {  // Waiting for R2 (held by T2)
            System.out.println("T1: Locked R2");
        }
    }
});

// Thread 2
Thread t2 = new Thread(() -> {
    synchronized(r2) {
        System.out.println("T2: Locked R2");
        try { Thread.sleep(100); } catch (InterruptedException e) { }
        
        synchronized(r1) {  // Waiting for R1 (held by T1)
            System.out.println("T2: Locked R1");
        }
    }
});

t1.start();
t2.start();
// DEADLOCK! Both threads wait forever
```

### Deadlock Prevention

**Solution: Always acquire locks in same order**

```java
// Both threads acquire locks in same order
Thread t1 = new Thread(() -> {
    synchronized(r1) {
        synchronized(r2) {
            System.out.println("T1: Done");
        }
    }
});

Thread t2 = new Thread(() -> {
    synchronized(r1) {  // Same order as T1
        synchronized(r2) {
            System.out.println("T2: Done");
        }
    }
});
```

---

## 8. Inter-thread Communication

### wait(), notify(), notifyAll()

Must be called from **synchronized** context.

```java
class SharedResource {
    private boolean dataReady = false;
    
    public synchronized void produce() {
        // Produce data
        dataReady = true;
        System.out.println("Data produced");
        notify();  // Wake up waiting consumer
    }
    
    public synchronized void consume() throws InterruptedException {
        while (!dataReady) {
            wait();  // Release lock and wait
        }
        System.out.println("Data consumed");
        dataReady = false;
    }
}
```

### Producer-Consumer Example

```java
class Queue {
    private LinkedList<Integer> queue = new LinkedList<>();
    private int capacity;
    
    public Queue(int capacity) {
        this.capacity = capacity;
    }
    
    public synchronized void produce(int value) throws InterruptedException {
        while (queue.size() == capacity) {
            wait();  // Queue full, wait for consumer
        }
        queue.add(value);
        System.out.println("Produced: " + value);
        notify();  // Notify consumer
    }
    
    public synchronized int consume() throws InterruptedException {
        while (queue.isEmpty()) {
            wait();  // Queue empty, wait for producer
        }
        int value = queue.removeFirst();
        System.out.println("Consumed: " + value);
        notify();  // Notify producer
        return value;
    }
}
```

⭐ **Exam Fact:** wait() **releases lock**, sleep() **does NOT release lock**.

---

## 🔥 Top MCQs for Sessions 13-14

### MCQ 1: start() vs run()
**Q:** What happens if you call run() instead of start()?
1. Same as start()
2. Compile error
3. Runs in current thread (doesn't create new thread)
4. RuntimeException

**Answer:** 3. Runs in current thread  
**Explanation:** run() is a normal method call, doesn't start new thread.

---

### MCQ 2: Runnable vs Thread
**Q:** Why is Runnable preferred over Thread?
1. Faster
2. Can extend another class
3. Uses less memory
4. Automatically synchronized

**Answer:** 2. Can extend another class  
**Explanation:** Java has single inheritance, Runnable allows extending other classes.

---

### MCQ 3: synchronized
**Q:** synchronized prevents:
1. Compilation error
2. Race condition
3. Memory leak
4. Stack overflow

**Answer:** 2. Race condition  
**Explanation:** synchronized ensures only one thread accesses critical section.

---

### MCQ 4: Deadlock
**Q:** Deadlock occurs when:
1. Thread sleeps forever
2. Circular lock dependency
3. Too many threads
4. Out of memory

**Answer:** 2. Circular lock dependency  
**Explanation:** Threads waiting for each other's locks create deadlock.

---

### MCQ 5: wait() vs sleep()
**Q:** Major difference between wait() and sleep()?
1. wait() throws exception
2. wait() releases lock, sleep() doesn't
3. sleep() is slower
4. No difference

**Answer:** 2. wait() releases lock, sleep() doesn't  
**Explanation:** wait() releases monitor lock, sleep() just pauses thread.

---

### MCQ 6: Thread Priority
**Q:** Thread priority range?
1. 0-9
2. 1-10
3. 1-5
4. 0-10

**Answer:** 2. 1-10  
**Explanation:** MIN_PRIORITY=1, NORM_PRIORITY=5, MAX_PRIORITY=10.

---

### MCQ 7: Daemon Thread
**Q:** Daemon threads prevent JVM from exiting?
1. Yes
2. No

**Answer:** 2. No  
**Explanation:** JVM exits when only daemon threads remain.

---

### MCQ 8: join()
**Q:** Thread.join() does what?
1. Starts thread
2. Waits for thread to complete
3. Stops thread
4. Pauses thread

**Answer:** 2. Waits for thread to complete  
**Explanation:** Current thread waits for the joined thread to finish.

---

### MCQ 9: InterruptedException
**Q:** Which throws InterruptedException?
1. start()
2. run()
3. sleep()
4. getName()

**Answer:** 3. sleep()  
**Explanation:** sleep(), wait(), join() throw InterruptedException.

---

### MCQ 10: synchronized Block
**Q:** What can synchronized lock on?
1. Only this
2. Any object
3. Only class
4. Primitives

**Answer:** 2. Any object  
**Explanation:** synchronized can lock on any object reference.

---

## ⚠️ Common Mistakes

1. **Calling run() instead of start()**
2. **Not handling InterruptedException**
3. **Forgetting synchronization** for shared data
4. **Deadlock from wrong lock order**
5. **Calling wait/notify** outside synchronized block
6. **Using sleep() when wait() is needed**
7. **Setting daemon** after starting thread

---

## ⭐ One-liner Exam Facts

1. **start()** starts new thread, **run()** doesn't
2. **Runnable** preferred (can extend other classes)
3. Thread shares **heap**, separate **stack**
4. **synchronized** prevents race condition
5. Thread priority: **1 (MIN) to 10 (MAX)**, default **5**
6. **Daemon threads** don't prevent JVM exit
7. **join()** waits for thread completion
8. **sleep()** throws **InterruptedException**
9. **wait()** releases lock, **sleep()** doesn't
10. **wait/notify/notifyAll** must be in **synchronized** block
11. Deadlock = **circular lock dependency**
12. Thread states: NEW → RUNNABLE → RUNNING → TERMINATED
13. **BLOCKED** = waiting for lock
14. **WAITING** = waiting indefinitely (wait, join)
15. **TIMED_WAITING** = waiting with timeout (sleep, wait with timeout)

---

**End of Sessions 13-14**
