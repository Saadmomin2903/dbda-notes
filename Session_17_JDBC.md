# Session 17 – JDBC (Java Database Connectivity)

**Topics Covered:** JDBC Introduction, Driver Types, Connection Steps, Statement vs PreparedStatement, ResultSet, SQL Injection Prevention, Batch Processing, Transactions

---

## 1. What is JDBC?

**JDBC** = Java Database Connectivity  
**Purpose:** Standard API for connecting Java applications to relational databases.

### JDBC Architecture

```
┌──────────────────┐
│  Java Application │
└────────┬─────────┘
         │ JDBC API
┌────────▼─────────┐
│   JDBC Driver    │
└────────┬─────────┘
         │ Database Protocol
┌────────▼─────────┐
│     Database     │
└──────────────────┘
```

---

## 2. JDBC Driver Types

| Type | Name | Description | Use Case |
|------|------|-------------|----------|
| **Type 1** | JDBC-ODBC Bridge | Uses ODBC driver | **Deprecated** |
| **Type 2** | Native-API Driver | Converts JDBC → DB native calls | Requires client-side libs |
| **Type 3** | Network Protocol Driver | Pure Java, middleware server | 3-tier architecture |
| **Type 4** | Thin Driver | **Pure Java, direct DB communication** | **Most common** (preferred) |

⭐ **Exam Fact:** **Type 4** (thin driver) is **most commonly used** (pure Java, platform-independent).

---

## 3. JDBC Steps (6 Steps)

```java
1. Load Driver (optional in JDBC 4.0+)
2. Establish Connection
3. Create Statement
4. Execute Query
5. Process Result
6. Close Resources
```

---

## 4. Step-by-Step JDBC Example

### Step 1: Load Driver (Optional in JDBC 4.0+)

```java
// Old way (JDBC 3.0 and earlier)
Class.forName("com.mysql.cj.jdbc.Driver");

// Modern way (JDBC 4.0+)
// Driver auto-loaded from classpath, no need for Class.forName()
```

⭐**Exam Fact:** JDBC 4.0+ **auto-loads drivers**, Class.forName() not required.

### Step 2: Establish Connection

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

String url = "jdbc:mysql://localhost:3306/mydb";
String username = "root";
String password = "password";

Connection conn = DriverManager.getConnection(url, username, password);
```

### JDBC URL Format

```
jdbc:<subprotocol>://<host>:<port>/<database>

Examples:
jdbc:mysql://localhost:3306/mydb
jdbc:postgresql://localhost:5432/testdb
jdbc:oracle:thin:@localhost:1521:orcl
jdbc:sqlserver://localhost:1433;databaseName=mydb
```

### Step 3: Create Statement

```java
import java.sql.Statement;

Statement stmt = conn.createStatement();
```

### Step 4: Execute Query

```java
import java.sql.ResultSet;

// SELECT query
ResultSet rs = stmt.executeQuery("SELECT * FROM users");

// INSERT/UPDATE/DELETE
int rows = stmt.executeUpdate("INSERT INTO users VALUES (1, 'Alice', 25)");
```

### Step 5: Process Result

```java
while (rs.next()) {
    int id = rs.getInt("id");          // or rs.getInt(1)
    String name = rs.getString("name"); // or rs.getString(2)
    int age = rs.getInt("age");        // or rs.getInt(3)
    
    System.out.println(id + " - " + name + " - " + age);
}
```

### Step 6: Close Resources

```java
rs.close();
stmt.close();
conn.close();
```

---

## 5. Complete Example

```java
import java.sql.*;

public class JDBCDemo {
    public static void main(String[] args) {
        String url = "jdbc:mysql://localhost:3306/mydb";
        String user = "root";
        String password = "password";
        
        try {
            // 2. Connection
            Connection conn = DriverManager.getConnection(url, user, password);
            
            // 3. Statement
            Statement stmt = conn.createStatement();
            
            // 4. Execute
            ResultSet rs = stmt.executeQuery("SELECT * FROM users");
            
            // 5. Process
            while (rs.next()) {
                System.out.println(rs.getInt("id") + " - " + rs.getString("name"));
            }
            
            // 6. Close
            rs.close();
            stmt.close();
            conn.close();
            
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```

---

## 6. Statement vs PreparedStatement

### Statement (SQL Injection Vulnerable)

```java
String username = "admin";
Statement stmt = conn.createStatement();

// SQL Injection vulnerable!
String sql = "SELECT * FROM users WHERE username = '" + username + "'";
ResultSet rs = stmt.executeQuery(sql);
```

### SQL Injection Attack

```java
String username = "admin' OR '1'='1";  // Malicious input
String sql = "SELECT * FROM users WHERE username = '" + username + "'";
// Becomes: SELECT * FROM users WHERE username = 'admin' OR '1'='1'
// Returns ALL users! (Security breach)
```

### PreparedStatement (Safe)

```java
String username = "admin' OR '1'='1";  // Same malicious input

PreparedStatement pstmt = conn.prepareStatement(
    "SELECT * FROM users WHERE username = ?"
);
pstmt.setString(1, username);  // Treats entire string as literal value
ResultSet rs = pstmt.executeQuery();

// Safe! Entire input treated as username string, not SQL code
```

⭐ **Exam Fact:** **Always use PreparedStatement** to prevent SQL injection.

---

## 7. Statement vs PreparedStatement Comparison

| Aspect | Statement | PreparedStatement |
|--------|-----------|-------------------|
| **SQL Injection** | Vulnerable | Safe (parameterized) |
| **Performance** | Compiled every time | Pre-compiled (faster) |
| **Use Case** | Static SQL (rare) | Dynamic SQL with parameters |
| **Readability** | Concatenation messy | Clean with placeholders |
| **Caching** | Not cached | Cached by DB |

---

## 8. PreparedStatement Examples

### INSERT

```java
PreparedStatement pstmt = conn.prepareStatement(
    "INSERT INTO users (id, name, age) VALUES (?, ?, ?)"
);
pstmt.setInt(1, 1);
pstmt.setString(2, "Alice");
pstmt.setInt(3, 25);

int rows = pstmt.executeUpdate();
System.out.println(rows + " row(s) inserted");
```

### UPDATE

```java
PreparedStatement pstmt = conn.prepareStatement(
    "UPDATE users SET name = ?, age = ? WHERE id = ?"
);
pstmt.setString(1, "Bob");
pstmt.setInt(2, 30);
pstmt.setInt(3, 1);

int rows = pstmt.executeUpdate();
System.out.println(rows + " row(s) updated");
```

### DELETE

```java
PreparedStatement pstmt = conn.prepareStatement(
    "DELETE FROM users WHERE id = ?"
);
pstmt.setInt(1, 1);

int rows = pstmt.executeUpdate();
System.out.println(rows + " row(s) deleted");
```

---

## 9. Execute Methods

| Method | Return Type | Use For | Description |
|--------|-------------|---------|-------------|
| `executeQuery()` | ResultSet | SELECT | Returns data |
| `executeUpdate()` | int | INSERT/UPDATE/DELETE | Returns row count |
| `execute()` | boolean | Any SQL | Returns true if ResultSet |

```java
// executeQuery() - SELECT
ResultSet rs = stmt.executeQuery("SELECT * FROM users");

// executeUpdate() - INSERT/UPDATE/DELETE
int rows = stmt.executeUpdate("INSERT INTO users VALUES (1, 'Alice', 25)");

// execute() - Generic (any SQL)
boolean hasResultSet = stmt.execute("SELECT * FROM users");
if (hasResultSet) {
    ResultSet rs = stmt.getResultSet();
}
```

---

## 10. ResultSet Methods

```java
ResultSet rs = stmt.executeQuery("SELECT id, name, age FROM users");

while (rs.next()) {  // Move to next row
    // Get by column name
    int id = rs.getInt("id");
    String name = rs.getString("name");
    int age = rs.getInt("age");
    
    // Get by column index (1-based)
    int id2 = rs.getInt(1);
    String name2 = rs.getString(2);
    int age2 = rs.getInt(3);
}

// Metadata
ResultSetMetaData metaData = rs.getMetaData();
int columnCount = metaData.getColumnCount();
String columnName = metaData.getColumnName(1);
```

---

## 11. Batch Processing

Execute multiple statements in one batch (better performance).

```java
PreparedStatement pstmt = conn.prepareStatement(
    "INSERT INTO users (id, name, age) VALUES (?, ?, ?)"
);

// Add to batch
pstmt.setInt(1, 1);
pstmt.setString(2, "Alice");
pstmt.setInt(3, 25);
pstmt.addBatch();

pstmt.setInt(1, 2);
pstmt.setString(2, "Bob");
pstmt.setInt(3, 30);
pstmt.addBatch();

pstmt.setInt(1, 3);
pstmt.setString(2, "Charlie");
pstmt.setInt(3, 35);
pstmt.addBatch();

// Execute all at once
int[] results = pstmt.executeBatch();
System.out.println("Inserted " + results.length + " rows");
```

---

## 12. Transactions

### ACID Properties

- **Atomicity**: All or nothing
- **Consistency**: Data remains consistent
- **Isolation**: Transactions isolated from each other
- **Durability**: Committed data is permanent

### Transaction Example

```java
try {
    conn.setAutoCommit(false);  // Disable auto-commit
    
    // Transaction operations
    Statement stmt = conn.createStatement();
    stmt.executeUpdate("UPDATE account SET balance = balance - 1000 WHERE id = 1");
    stmt.executeUpdate("UPDATE account SET balance = balance + 1000 WHERE id = 2");
    
    conn.commit();  // Commit transaction
    System.out.println("Transaction successful");
    
} catch (SQLException e) {
    try {
        conn.rollback();  // Rollback on error
        System.out.println("Transaction rolled back");
    } catch (SQLException ex) {
        ex.printStackTrace();
    }
} finally {
    try {
        conn.setAutoCommit(true);  // Restore auto-commit
    } catch (SQLException e) {
        e.printStackTrace();
    }
}
```

---

## 13. try-with-resources (Automatic Close)

```java
String url = "jdbc:mysql://localhost:3306/mydb";

try (Connection conn = DriverManager.getConnection(url, "root", "password");
     Statement stmt = conn.createStatement();
     ResultSet rs = stmt.executeQuery("SELECT * FROM users")) {
    
    while (rs.next()) {
        System.out.println(rs.getString("name"));
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
// All resources auto-closed
```

⭐ **Exam Fact:** Use **try-with-resources** to automatically close JDBC resources.

---

## 🔥 Top MCQs for Session 17

### MCQ 1: Driver Type
**Q:** Most commonly used JDBC driver type?
1. Type 1
2. Type 2
3. Type 3
4. Type 4

**Answer:** 4. Type 4 (Thin driver)  
**Explanation:** Pure Java, platform-independent, most widely used.

---

### MCQ 2: SQL Injection
**Q:** Which prevents SQL injection?
1. Statement
2. PreparedStatement
3. CallableStatement
4. None

**Answer:** 2. PreparedStatement  
**Explanation:** Parameterized queries prevent SQL injection.

---

### MCQ 3: executeQuery()
**Q:** executeQuery() returns:
1. int
2. boolean
3. ResultSet
4. void

**Answer:** 3. ResultSet  
**Explanation:** executeQuery() returns ResultSet for SELECT queries.

---

### MCQ 4: executeUpdate()
**Q:** executeUpdate() returns:
1. ResultSet
2. int (row count)
3. boolean
4. void

**Answer:** 2. int (row count)  
**Explanation:** Returns number of affected rows for INSERT/UPDATE/DELETE.

---

### MCQ 5: JDBC URL
**Q:** Correct JDBC URL format?
1. jdbc:mysql:localhost:3306/mydb
2. jdbc://mysql://localhost:3306/mydb
3. jdbc:mysql://localhost:3306/mydb
4. mysql://localhost:3306/mydb

**Answer:** 3. jdbc:mysql://localhost:3306/mydb  
**Explanation:** Format: jdbc:<subprotocol>://<host>:<port>/<database>

---

### MCQ 6: Auto-Commit
**Q:** Default auto-commit mode?
1. true (enabled)
2. false (disabled)

**Answer:** 1. true (enabled)  
**Explanation:** By default, each statement auto-commits.

---

### MCQ 7: ResultSet Column Index
**Q:** ResultSet column index starts from:
1. 0
2. 1

**Answer:** 2. 1  
**Explanation:** ResultSet is 1-indexed (not 0-indexed like arrays).

---

### MCQ 8: Batch Processing
**Q:** Batch processing improves:
1. Security
2. Performance
3. Readability
4. Compatibility

**Answer:** 2. Performance  
**Explanation:** Executes multiple statements in one round-trip to DB.

---

### MCQ 9: Class.forName()
**Q:** In JDBC 4.0+, Class.forName() is:
1. Required
2. Optional
3. Deprecated
4. Removed

**Answer:** 2. Optional  
**Explanation:** JDBC 4.0+ auto-loads drivers, Class.forName() not needed.

---

### MCQ 10: Transaction Rollback
**Q:** rollback() is used to:
1. Commit changes
2. Undo uncommitted changes
3. Close connection
4. Execute query

**Answer:** 2. Undo uncommitted changes  
**Explanation:** Rollback reverts transaction changes on error.

---

## ⚠️ Common Mistakes

1. **Using Statement** instead of PreparedStatement (SQL injection)
2. **Not closing resources** (connection leaks)
3. **Forgetting setAutoCommit(false)** for transactions
4. **Using 0-based indexing** for ResultSet (it's 1-based)
5. **Concatenating SQL** strings (use PreparedStatement)
6. **Not handling SQLException** properly
7. **Confusing executeQuery** vs **executeUpdate**

---

## ⭐ One-liner Exam Facts

1. **Type 4** driver most common (pure Java)
2. **PreparedStatement** prevents SQL injection
3. **executeQuery()** → ResultSet (SELECT)
4. **executeUpdate()** → int (INSERT/UPDATE/DELETE)
5. JDBC URL format: **jdbc:subprotocol://host:port/database**
6. JDBC 4.0+ **auto-loads** drivers (Class.forName optional)
7. ResultSet is **1-indexed** (not 0-indexed)
8. **setAutoCommit(false)** for manual transactions
9. **commit()** saves changes, **rollback()** undoes
10. **try-with-resources** auto-closes JDBC resources
11. **Batch processing** improves performance
12. PreparedStatement is **pre-compiled** (faster)
13. Statement **compiled every time** (slower)
14. Always **close** Connection, Statement, ResultSet
15. **ACID**: Atomicity, Consistency, Isolation, Durability

---

**End of Session 17**

---

# 🎓 ALL 17 SESSIONS COMPLETE!

You now have comprehensive, exam-ready Java Programming notes covering:
- ✅ Sessions 1-2: Java Basics & JVM
- ✅ Session 3: Object Lifecycle & Operators
- ✅ Session 4: Arrays, Strings & Encapsulation
- ✅ Session 5: Inheritance & Polymorphism
- ✅ Session 6: Exception Handling
- ✅ Session 7: Enum, Autoboxing & Annotations
- ✅ Session 8: java.lang & java.util
- ✅ Session 9-10: Generics & Collections
- ✅ Session 11: Functional Programming
- ✅ Session 12: Streams & Date/Time API
- ✅ Session 13-14: Concurrency
- ✅ Session 15: IO & Serialization
- ✅ Session 16: JVM Internals & Reflection
- ✅ Session 17: JDBC

**Total: 100+ MCQs, 300+ code examples, 20+ diagrams, 30+ comparison tables**

**Good luck with your PG-DBDA Java Programming exam! 🚀**
