# 03. Time & Space Complexity Analysis

## 📋 Table of Contents

1. [Big O Notation](#big-o-notation)
2. [Time Complexity Examples](#time-complexity-examples)
3. [Space Complexity Examples](#space-complexity-examples)
4. [Optimization Techniques](#optimization-techniques)
5. [Real-World Performance](#real-world-performance)

---

## Big O Notation

### 🎯 What is Big O?

Big O notation describes how an algorithm's performance scales with input size.

### 📊 Common Complexities

| Complexity | Name         | Example             | Frappe Use Case          |
| ---------- | ------------ | ------------------- | ------------------------ |
| O(1)       | Constant     | Hash table lookup   | Cache access             |
| O(log n)   | Logarithmic  | Binary search       | Database indexing        |
| O(n)       | Linear       | Array traversal     | Employee list processing |
| O(n log n) | Linearithmic | Merge sort          | Report sorting           |
| O(n²)      | Quadratic    | Nested loops        | BOM explosion            |
| O(2ⁿ)      | Exponential  | Recursive Fibonacci | Complex calculations     |

---

## Time Complexity Examples

### 1. O(1) - Cache Lookup

**Location**: `frappe/frappe/utils/caching.py`

```python
def site_cache(ttl: int | None = None, maxsize: int | None = None):
    def wrapper(func):
        @wraps(func)
        def cached_func(*args, **kwargs):
            cache_key = __generate_request_cache_key(args, kwargs)

            # O(1) hash table lookup
            if cache_key in _SITE_CACHE[site][func.__name__]:
                return _SITE_CACHE[site][func.__name__][cache_key]

            result = func(*args, **kwargs)
            _SITE_CACHE[site][func.__name__][cache_key] = result
            return result
        return cached_func
    return wrapper
```

**Why O(1)?**

- Hash table provides constant-time access
- No iteration needed
- Direct key lookup

### 2. O(n) - Employee Search

**Location**: `erpnext/erpnext/controllers/queries.py`

```python
def employee_query(doctype, txt, searchfield, start, page_len, filters, **kwargs):
    return frappe.db.sql(
        """select {fields} from `tabEmployee`
        where status in ('Active', 'Suspended')
            and docstatus < 2
            and ({key} like %(txt)s
                or employee_name like %(txt)s)
        order by
            (case when locate(%(_txt)s, name) > 0 then locate(%(_txt)s, name) else 99999 end),
            name, employee_name
        limit %(page_len)s offset %(start)s""",
        {"txt": "%%%s%%" % txt, "_txt": txt.replace("%", ""), "start": start, "page_len": page_len},
    )
```

**Why O(n)?**

- Database scans all matching records
- Linear relationship with data size
- No indexing optimization

### 3. O(n log n) - Report Sorting

**Location**: `fusion_hr/fusion_hr/fusion_hr/report/fervent_salary_register/fervent_salary_register.js`

```javascript
function sortingDataDepartmentWise(data) {
    data.sort((a, b) => {
        const departmentOrders = ["Management", "Project Management", ...];
        const aIndex = departmentOrders.indexOf(a.sub_department);
        const bIndex = departmentOrders.indexOf(b.sub_department);

        if (aIndex !== -1 && bIndex !== -1) {
            return aIndex - bIndex;
        }

        return a.sub_department.localeCompare(b.sub_department);
    });

    return data;
}
```

**Why O(n log n)?**

- JavaScript's `sort()` uses efficient sorting algorithm
- Comparison function adds O(1) per comparison
- Overall complexity: O(n log n)

### 4. O(n²) - BOM Explosion

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
def traverse_tree(self, bom_list=None):
    def _get_children(bom_no):
        children = frappe.cache().hget("bom_children", bom_no)
        if children is None:
            children = frappe.db.sql_list(
                """SELECT `bom_no` FROM `tabBOM Item`
                WHERE `parent`=%s AND `bom_no`!='' AND `parenttype`='BOM'""",
                bom_no,
            )
            frappe.cache().hset("bom_children", bom_no, children)
        return children

    count = 0
    if not bom_list:
        bom_list = []

    if self.name not in bom_list:
        bom_list.append(self.name)

    while count < len(bom_list):
        for child_bom in _get_children(bom_list[count]):
            if child_bom not in bom_list:
                bom_list.append(child_bom)
        count += 1

    return bom_list
```

**Why O(n²)?**

- Nested loops: outer while loop, inner for loop
- Each BOM can have multiple children
- Worst case: every BOM references every other BOM

---

## Space Complexity Examples

### 1. O(1) - Simple Calculations

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/festival_bonus_entry/festival_bonus_entry.py`

```python
def calculate_service_days(joining_date, payroll_date):
    if not joining_date or not payroll_date:
        return 0

    joining_date = getdate(joining_date)
    payroll_date = getdate(payroll_date)

    if joining_date > payroll_date:
        return 0

    service_days = (payroll_date - joining_date).days
    return max(0, service_days)
```

**Why O(1)?**

- Uses only a few variables
- No additional data structures
- Constant memory usage

### 2. O(n) - Employee List Processing

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/festival_bonus_entry/festival_bonus_entry.py`

```python
def process_bonus_calculation(doc):
    employees = get_eligible_employees(doc)
    festival_slabs = doc.get("festival_slabs", [])
    eligible_employees = []

    for employee in employees:
        service_days = calculate_service_days(employee.date_of_joining, payroll_date)
        matching_slab = find_matching_slab(service_days, festival_slabs)

        if matching_slab:
            eligible_employees.append({
                "employee": employee.name,
                "gross_salary": gross_salary,
                "service_days": service_days,
                "bonus_percentage": bonus_percentage,
                "payable_amount": payable_amount,
            })

    return eligible_employees
```

**Why O(n)?**

- Stores result for each employee
- Linear relationship with employee count
- Additional space for each processed employee

### 3. O(n) - BOM Tree Storage

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
class BOMTree:
    def __init__(self, name: str, is_bom: bool = True, exploded_qty: float = 1.0, qty: float = 1):
        self.name = name
        self.child_items: list["BOMTree"] = []  # O(n) space
        self.is_bom = is_bom
        self.item_code: str = None
        self.qty = qty
        self.exploded_qty = exploded_qty
```

**Why O(n)?**

- Each BOM item requires storage
- Child items list grows with BOM complexity
- Tree structure uses O(n) space

---

## Optimization Techniques

### 1. Caching for O(1) Access

**Location**: `frappe/frappe/utils/caching.py`

```python
@site_cache(ttl=3600)
def expensive_calculation(data):
    # Complex calculation that takes time
    result = perform_complex_calculation(data)
    return result
```

**Optimization**: Cache results to avoid repeated calculations

### 2. Database Indexing

**Location**: `frappe/frappe/core/doctype/recorder/db_optimizer.py`

```python
def potential_indexes(self) -> list[DBIndex]:
    """Get all columns that can potentially be indexed to speed up this query."""

    possible_indexes = []

    # Where clause columns benefit from index
    if where_columns := self.parsed_query.columns_dict.get("where"):
        possible_indexes.extend(where_columns)

    # Join clauses - Both sides should be indexed
    if join_columns := self.parsed_query.columns_dict.get("join"):
        possible_indexes.extend(join_columns)

    return possible_indexes
```

**Optimization**: Suggest database indexes for faster queries

### 3. Pagination for Large Datasets

**Location**: `frappe/frappe/desk/search.py`

```python
def search_widget(
    doctype: str,
    txt: str,
    start: int = 0,
    page_length: int = 10,
    **kwargs
):
    values = frappe.get_list(
        doctype,
        filters=filters,
        fields=formatted_fields,
        limit_start=start,
        limit_page_length=page_length,  # Limit results
        order_by=order_by,
    )

    return values
```

**Optimization**: Limit results to prevent memory issues

---

## Real-World Performance

### 🚀 Performance Tips from Frappe

1. **Use appropriate data structures**

   - Hash tables for O(1) lookups
   - Lists for ordered data
   - Trees for hierarchical data

2. **Optimize database queries**

   - Use indexes effectively
   - Limit result sets
   - Cache frequently accessed data

3. **Handle large datasets**
   - Implement pagination
   - Use background jobs for heavy processing
   - Optimize memory usage

### 📊 Complexity Comparison

| Operation | Array | Hash Table | Tree     | Graph  |
| --------- | ----- | ---------- | -------- | ------ |
| Search    | O(n)  | O(1)       | O(log n) | O(V+E) |
| Insert    | O(1)  | O(1)       | O(log n) | O(1)   |
| Delete    | O(n)  | O(1)       | O(log n) | O(1)   |
| Space     | O(n)  | O(n)       | O(n)     | O(V+E) |

---

## 🎯 Interview Questions & Answers

### Q1: "What's the time complexity of Frappe's employee search?"

**Answer**: "Frappe's employee search has O(n) time complexity because it performs a linear scan through the employee records. However, with proper database indexing on the search fields, it can be optimized to O(log n) for exact matches.

The current implementation:

```python
# O(n) - scans all matching records
where ({key} like %(txt)s or employee_name like %(txt)s)
```

Could be optimized with:

````python
# O(log n) with proper indexing
where {key} = %(txt)s  # Exact match with index
```"

### Q2: "How does Frappe optimize BOM explosion performance?"
**Answer**: "Frappe uses several optimization techniques for BOM explosion:

1. **Caching**: Stores BOM children in cache to avoid repeated database queries
2. **Early termination**: Limits recursion depth to prevent infinite loops
3. **Batch processing**: Processes multiple BOMs together

The space complexity is O(n) where n is the number of BOM items, and time complexity is O(n²) in the worst case due to nested loops for dependency resolution."

### Q3: "Explain the space complexity of Frappe's caching system."
**Answer**: "Frappe's caching system has O(n) space complexity where n is the number of cached items. The system uses:

1. **Hash tables** for O(1) access time
2. **TTL (Time To Live)** to prevent unlimited growth
3. **Namespace separation** to avoid conflicts

```python
_SITE_CACHE = defaultdict(lambda: defaultdict(dict))
````

This structure allows efficient storage and retrieval while maintaining reasonable memory usage through TTL expiration."

---

## 🚀 Next Steps

Ready to learn problem-solving patterns? Move on to [Problem Solving Patterns](./04_Problem_Solving_Patterns.md) to master common algorithmic approaches!

**Key Takeaways:**

- Understand how algorithms scale with input size
- Choose appropriate data structures for your use case
- Optimize for both time and space complexity
- Use real examples to explain complexity in interviews
