# 02. Algorithms in Frappe Ecosystem

## 📋 Table of Contents

1. [Sorting Algorithms](#sorting-algorithms)
2. [Searching Algorithms](#searching-algorithms)
3. [Recursion](#recursion)
4. [Dynamic Programming](#dynamic-programming)
5. [Greedy Algorithms](#greedy-algorithms)
6. [Graph Algorithms](#graph-algorithms)

---

## Sorting Algorithms

### 🎯 What is Sorting?

Arranging data in a particular order (ascending or descending) based on some criteria.

### 💼 Real Examples from Frappe

#### 1. Employee Search with Relevance Sorting

**Location**: `erpnext/erpnext/controllers/queries.py`

```python
@frappe.whitelist()
@frappe.validate_and_sanitize_search_inputs
def employee_query(doctype, txt, searchfield, start, page_len, filters, **kwargs):
    return frappe.db.sql(
        """select {fields} from `tabEmployee`
        where status in ('Active', 'Suspended')
            and docstatus < 2
            and ({key} like %(txt)s
                or employee_name like %(txt)s)
            {fcond} {mcond}
        order by
            (case when locate(%(_txt)s, name) > 0 then locate(%(_txt)s, name) else 99999 end),
            (case when locate(%(_txt)s, employee_name) > 0 then locate(%(_txt)s, employee_name) else 99999 end),
            idx desc,
            name, employee_name
        limit %(page_len)s offset %(start)s""".format(
            **{
                "fields": ", ".join(fields),
                "key": searchfield,
                "fcond": get_filters_cond(doctype, filters, conditions),
                "mcond": mcond,
            }
        ),
        {"txt": "%%%s%%" % txt, "_txt": txt.replace("%", ""), "start": start, "page_len": page_len},
    )
```

**Algorithm Used**: Custom sorting with relevance scoring
**Why This Approach?**

- Prioritizes exact matches
- Uses `LOCATE()` function for position-based sorting
- Falls back to alphabetical sorting

#### 2. Item Search with Multiple Criteria

**Location**: `erpnext/erpnext/controllers/queries.py`

```python
def item_query(doctype, txt, searchfield, start, page_len, filters, as_dict=False):
    return frappe.db.sql(
        """select
            tabItem.name {columns}
        from tabItem
        where tabItem.docstatus < 2
            and tabItem.disabled=0
            and tabItem.has_variants=0
            and (tabItem.end_of_life > %(today)s or ifnull(tabItem.end_of_life, '0000-00-00')='0000-00-00')
            and ({scond} or tabItem.item_code IN (select parent from `tabItem Barcode` where barcode LIKE %(txt)s)
                {description_cond})
            {fcond} {mcond}
        order by
            if(locate(%(_txt)s, name), locate(%(_txt)s, name), 99999),
            if(locate(%(_txt)s, item_name), locate(%(_txt)s, item_name), 99999),
            idx desc,
            name, item_name
        limit %(start)s, %(page_len)s """.format(
            columns=columns,
            scond=searchfields,
            fcond=get_filters_cond(doctype, filters, conditions).replace("%", "%%"),
            mcond=get_match_cond(doctype).replace("%", "%%"),
            description_cond=description_cond,
        ),
        {
            "today": nowdate(),
            "txt": "%%%s%%" % txt,
            "_txt": txt.replace("%", ""),
            "start": start,
            "page_len": page_len,
        },
        as_dict=as_dict,
    )
```

**Algorithm Used**: Multi-criteria sorting with relevance
**Why This Approach?**

- Handles multiple search fields
- Prioritizes exact matches
- Efficient database-level sorting

#### 3. Department-wise Salary Register Sorting

**Location**: `fusion_hr/fusion_hr/fusion_hr/report/fervent_salary_register/fervent_salary_register.js`

```javascript
function sortingDataDepartmentWise(data) {
  data.sort((a, b) => {
    const departmentOrders = [
      "Management (C)",
      "Management (F)",
      "Project Management (C)",
      "Project Management (F)",
      // ... more departments
    ];
    const aIndex = departmentOrders.indexOf(a.sub_department);
    const bIndex = departmentOrders.indexOf(b.sub_department);

    if (aIndex !== -1 && bIndex !== -1) {
      return aIndex - bIndex;
    }

    if (aIndex !== -1) return -1;
    if (bIndex !== -1) return 1;

    return a.sub_department.localeCompare(b.sub_department);
  });

  return data;
}
```

**Algorithm Used**: Custom comparator sorting
**Why This Approach?**

- Predefined department hierarchy
- Fallback to alphabetical sorting
- Maintains business logic order

### 🔧 Common Sorting Patterns

```python
# Simple sorting
employees.sort(key=lambda x: x['name'])
employees.sort(key=lambda x: x['salary'], reverse=True)

# Multi-criteria sorting
employees.sort(key=lambda x: (x['department'], x['salary']))

# Custom sorting with relevance
def relevance_sort(items, search_term):
    def sort_key(item):
        name = item.get('name', '').lower()
        search = search_term.lower()

        if name.startswith(search):
            return (0, name)  # Exact prefix match
        elif search in name:
            return (1, name)  # Contains match
        else:
            return (2, name)  # No match

    return sorted(items, key=sort_key)
```

---

## Searching Algorithms

### 🎯 What is Searching?

Finding specific elements or patterns within a collection of data.

### 💼 Real Examples from Frappe

#### 1. Global Search with Full-Text Search

**Location**: `frappe/frappe/utils/global_search.py`

```python
@frappe.whitelist()
def search(text, start=0, limit=20, doctype=""):
    """Search for given text in __global_search"""
    results = []
    sorted_results = []

    allowed_doctypes = set(get_doctypes_for_global_search()) & set(frappe.get_user().get_can_read())
    if not allowed_doctypes or (doctype and doctype not in allowed_doctypes):
        return []

    for word in set(text.split("&")):
        word = word.strip()
        if not word:
            continue

        global_search = frappe.qb.Table("__global_search")
        rank = Match(global_search.content).Against(word)
        query = (
            frappe.qb.from_(global_search)
            .select(global_search.doctype, global_search.name, global_search.content, rank.as_("rank"))
            .where(rank)
            .orderby("rank", order=frappe.qb.desc)
            .limit(limit)
        )

        if doctype:
            query = query.where(global_search.doctype == doctype)
        else:
            query = query.where(global_search.doctype.isin(allowed_doctypes))

        if cint(start) > 0:
            query = query.offset(start)

        result = query.run(as_dict=True)
        results.extend(result)

    # Sort results based on doctype priority
    for doctype in allowed_doctypes:
        for r in results:
            if r.doctype == doctype and r.rank > 0.0:
                sorted_results.extend([r])

    return sorted_results
```

**Algorithm Used**: Full-text search with ranking
**Why This Approach?**

- Uses MySQL's `MATCH() AGAINST()` for relevance scoring
- Handles multiple search terms
- Prioritizes results by doctype importance

#### 2. Point of Sale Item Search

**Location**: `erpnext/erpnext/selling/page/point_of_sale/point_of_sale.py`

```python
def search_by_term(search_term, warehouse, price_list):
    result = frappe.db.sql(
        """select
            tabItem.name, tabItem.item_name, tabItem.description, tabItem.image,
            tabItem.is_stock_item, tabItem.has_variants, tabItem.variant_of,
            tabItem.item_group, tabItem.has_batch_no, tabItem.has_serial_no,
            tabItem.sales_uom, tabItem.stock_uom
        from tabItem
        where tabItem.docstatus < 2
            and tabItem.disabled=0
            and tabItem.has_variants=0
            and (tabItem.end_of_life > %(today)s or ifnull(tabItem.end_of_life, '0000-00-00')='0000-00-00')
            and (tabItem.name like %(txt)s
                or tabItem.item_name like %(txt)s
                or tabItem.description like %(txt)s
                or tabItem.item_code IN (select parent from `tabItem Barcode` where barcode LIKE %(txt)s))
        order by
            if(locate(%(_txt)s, tabItem.name), locate(%(_txt)s, tabItem.name), 99999),
            if(locate(%(_txt)s, tabItem.item_name), locate(%(_txt)s, tabItem.item_name), 99999),
            tabItem.idx desc, tabItem.name, tabItem.item_name
        limit 20""",
        {
            "today": nowdate(),
            "txt": "%%%s%%" % search_term,
            "_txt": search_term.replace("%", ""),
        },
        as_dict=1,
    )

    return result
```

**Algorithm Used**: Multi-field search with relevance ranking
**Why This Approach?**

- Searches across multiple fields
- Uses `LOCATE()` for position-based ranking
- Handles barcode searches

#### 3. Festival Bonus Slab Matching

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/festival_bonus_entry/festival_bonus_entry.py`

```python
def find_matching_slab(service_days, festival_slabs):
    """
    Find the matching slab based on service days range

    This function matches an employee to the appropriate bonus slab based on:
    1. Service days meeting or exceeding the slab's minimum_days requirement
    2. Service days being less than or equal to the slab's maximum_days requirement
    3. If maximum_days is 0, it means infinity (no upper limit)

    If multiple slabs match, it returns the one with the highest bonus percentage.
    """
    matching_slabs = []

    for slab in festival_slabs:
        min_days = slab.get('minimum_days', 0) or 0
        max_days = slab.get('maximum_days', 0) or 0

        # Check if service days meet the range requirements
        if service_days >= min_days and (max_days == 0 or service_days <= max_days):
            matching_slabs.append(slab)

    # Return the slab with the highest bonus percentage if multiple matches
    if matching_slabs:
        return max(matching_slabs, key=lambda x: x.get('bonus_percentage', 0) or 0)

    return None
```

**Algorithm Used**: Range-based search with optimization
**Why This Approach?**

- Efficient range matching
- Handles edge cases (infinite upper bound)
- Optimizes for best match (highest bonus)

### 🔧 Common Searching Patterns

```python
# Linear search
def linear_search(items, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# Binary search (for sorted arrays)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Search with multiple criteria
def multi_criteria_search(items, criteria):
    results = []
    for item in items:
        match = True
        for key, value in criteria.items():
            if item.get(key) != value:
                match = False
                break
        if match:
            results.append(item)
    return results
```

---

## Recursion

### 🎯 What is Recursion?

A technique where a function calls itself to solve smaller instances of the same problem.

### 💼 Real Examples from Frappe

#### 1. BOM Tree Traversal

**Location**: `erpnext/erpnext/manufacturing/doctype/bom/bom.py`

```python
class BOMTree:
    def __init__(self, name: str, is_bom: bool = True, exploded_qty: float = 1.0, qty: float = 1):
        self.name = name
        self.child_items: list["BOMTree"] = []
        self.is_bom = is_bom
        self.item_code: str = None
        self.qty = qty
        self.exploded_qty = exploded_qty

        if not self.is_bom:
            self.item_code = self.name
        else:
            self.__create_tree()

    def __create_tree(self):
        bom = frappe.get_cached_doc("BOM", self.name)
        self.item_code = bom.item
        self.bom_qty = bom.quantity

        for item in bom.get("items", []):
            qty = item.stock_qty / bom.quantity  # quantity per unit
            exploded_qty = self.exploded_qty * qty
            if item.bom_no:
                # Recursive call to create child BOM tree
                child = BOMTree(item.bom_no, exploded_qty=exploded_qty, qty=qty)
                self.child_items.append(child)
            else:
                self.child_items.append(
                    BOMTree(item.item_code, is_bom=False, exploded_qty=exploded_qty, qty=qty)
                )
```

**Why Recursion?**

- Natural fit for tree structures
- Handles nested BOMs elegantly
- Calculates exploded quantities recursively

#### 2. Organizational Hierarchy Building

**Location**: `fusion_hr/fusion_hr/fusion_hr/page/organogram_chart/organogram_chart.py`

```python
def build_hierarchy(data, parent=None):
    hierarchy = []
    for row in data:
        if row["parent_ref"] == parent:
            # Recursively find children of the current row
            children = build_hierarchy(data, row["name_code"])
            hierarchy.append({
                "id": row["name_code"],
                "name": row["name"],
                "type": row.type,
                "children": children
            })
    return hierarchy
```

**Why Recursion?**

- Builds hierarchical structures naturally
- Handles arbitrary depth levels
- Clean, readable code

#### 3. Account Tree Building

**Location**: `erpnext/erpnext/accounts/doctype/account/chart_of_accounts/chart_of_accounts.py`

```python
def build_account_tree(tree, parent, all_accounts):
    # find children
    parent_account = parent.name if parent else ""
    children = [acc for acc in all_accounts if cstr(acc.parent_account) == parent_account]

    # if no children, but a group account
    if not children and parent.is_group:
        tree["is_group"] = 1
        tree["account_number"] = parent.account_number

    # build a subtree for each child
    for child in children:
        # start new subtree
        tree[child.account_name] = {}

        # assign account_type and root_type
        if child.account_number:
            tree[child.account_name]["account_number"] = child.account_number
        if child.account_type:
            tree[child.account_name]["account_type"] = child.account_type
        if child.tax_rate:
            tree[child.account_name]["tax_rate"] = child.tax_rate
        if not parent:
            tree[child.account_name]["root_type"] = child.root_type

        # call recursively to build a subtree for current account
        build_account_tree(tree[child.account_name], child, all_accounts)
```

**Why Recursion?**

- Handles nested account structures
- Maintains parent-child relationships
- Efficient tree building

### 🔧 Common Recursion Patterns

```python
# Tree traversal
def traverse_tree(node):
    if not node:
        return

    print(node.value)  # Process current node
    for child in node.children:
        traverse_tree(child)  # Recursive call

# Factorial calculation
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Directory traversal
def list_files(directory):
    files = []
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            files.extend(list_files(path))  # Recursive call
        else:
            files.append(path)
    return files
```

---

## Dynamic Programming

### 🎯 What is Dynamic Programming?

A method for solving complex problems by breaking them down into simpler subproblems and storing the results to avoid redundant calculations.

### 💼 Real Examples from Frappe

#### 1. Overtime Calculation with Slabs

**Location**: `fusion_hr/fusion_hr/fusion_hr/doctype/overtime_process/overtime_process.py`

```python
def _calculate_overtime_amounts_for_value(hours_value, ledger_entry, follow_labour_law):
    """Calculate overtime amounts for a given hours value using slab-based calculation."""

    if hours_value <= 0:
        return {
            "total_amount": 0.0,
            "labour_law_hours": 0.0,
            "labour_law_amount": 0.0,
            "non_labour_law_hours": 0.0,
            "non_labour_law_amount": 0.0
        }

    if not ledger_entry.get("overtime_slab"):
        return {
            "total_amount": 0.0,
            "labour_law_hours": 0.0,
            "labour_law_amount": 0.0,
            "non_labour_law_hours": 0.0,
            "non_labour_law_amount": 0.0
        }

    # Get the overtime slab
    slab = frappe.get_doc("Overtime Slab", ledger_entry["overtime_slab"])
    basis = (getattr(slab, "formula_based_on", "") or "").strip()

    # Use tiers in given order
    tiers = slab.get("overtime_tiers", [])

    remaining = hours_value
    total_amount = 0.0
    labour_law_hours = 0.0
    labour_law_amount = 0.0
    non_labour_law_hours = 0.0
    non_labour_law_amount = 0.0

    for tier in tiers:
        segment_hours = float(tier.get("ot_hour_max") or 0)
        if segment_hours <= 0:
            continue

        chunk = min(segment_hours, remaining)
        if chunk <= 0:
            continue

        # Derive rate per hour
        rate = 0.0
        if basis:
            rate = _safe_eval_formula(
                tier.get("ot_rate_formula"),
                float(ledger_entry.get("basic") or 0),
                float(ledger_entry.get("gross_pay") or 0)
            )
        else:
            rate = float(tier.get("ot_rate_fixed") or 0)

        amount = round(chunk * float(rate or 0), 2)
        total_amount += amount

        # Categorize by labour law compliance
        according_to_labour_law = tier.get("according_to_labour_law", 0)
        if according_to_labour_law:
            labour_law_hours += chunk
            labour_law_amount += amount
        else:
            non_labour_law_hours += chunk
            non_labour_law_amount += amount

        remaining -= chunk
        if remaining <= 0:
            break

    return {
        "total_amount": round(total_amount, 2),
        "labour_law_hours": round(labour_law_hours, 2),
        "labour_law_amount": round(labour_law_amount, 2),
        "non_labour_law_hours": round(non_labour_law_hours, 2),
        "non_labour_law_amount": round(non_labour_law_amount, 2)
    }
```

**Why Dynamic Programming?**

- Breaks down overtime calculation into slabs
- Avoids redundant calculations
- Handles complex tier-based calculations

#### 2. FIFO Stock Valuation

**Location**: `erpnext/erpnext/stock/valuation.py`

```python
class FIFOValuation(BinWiseValuation):
    """Valuation method where a queue of all the incoming stock is maintained.

    New stock is added at end of the queue.
    Qty consumption happens on First In First Out basis.

    Queue is implemented using "bins" of [qty, rate].
    """

    def __init__(self, state: list[StockBin] | None):
        self.queue: list[StockBin] = state if state is not None else []

    def add_stock(self, qty: float, rate: float) -> None:
        """Update fifo queue with new stock."""
        if not len(self.queue):
            self.queue.append([0, 0])

        # last row has the same rate, merge new bin.
        if self.queue[-1][RATE] == rate:
            self.queue[-1][QTY] += qty
        else:
            # Item has a positive balance qty, add new entry
            if self.queue[-1][QTY] > 0:
                self.queue.append([qty, rate])
            else:  # negative balance qty
                qty = self.queue[-1][QTY] + qty
                if qty > 0:  # new balance qty is positive
                    self.queue[-1] = [qty, rate]
                else:  # new balance qty is still negative, maintain same rate
                    self.queue[-1][QTY] = qty

    def remove_stock(self, qty: float, outgoing_rate: float = 0.0, rate_generator: Callable[[], float] | None = None) -> list[StockBin]:
        """Remove stock from FIFO queue."""
        # Implementation continues...
```

**Why Dynamic Programming?**

- Maintains state of stock bins
- Optimizes stock valuation calculations
- Handles complex stock movements

### 🔧 Common Dynamic Programming Patterns

```python
# Memoization decorator
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

# Bottom-up DP example
def fibonacci_dp(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]

# Top-down DP with memoization
@memoize
def fibonacci_memo(n):
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
```

---

## Greedy Algorithms

### 🎯 What are Greedy Algorithms?

Algorithms that make locally optimal choices at each step, hoping to find a global optimum.

### 💼 Real Examples from Frappe

#### 1. Payment Allocation

**Location**: `erpnext/erpnext/accounts/doctype/payment_entry/payment_entry.py`

```python
def allocate_open_payment_requests_to_references(references=None, precision=None):
    """
    Allocate unpaid Payment Requests to the references.

    Allocation based on below factors:
    - Reference Allocated Amount
    - Reference Outstanding Amount (With Payment Terms or without Payment Terms)
    - Reference Payment Request's outstanding amount

    Allocation based on below scenarios:
    - Reference's Allocated Amount == Payment Request's Outstanding Amount
        - Allocate the Payment Request to the reference
        - This PR will not be allocated further
    - Reference's Allocated Amount < Payment Request's Outstanding Amount
        - Allocate the Payment Request to the reference
        - Reduce the PR's outstanding amount by the allocated amount
        - This PR can be allocated further
    - Reference's Allocated Amount > Payment Request's Outstanding Amount
        - Allocate the Payment Request to the reference
        - Reduce Allocated Amount of the reference by the PR's outstanding amount
        - Create a new row for the remaining amount until the Allocated Amount is 0
            - Allocate PR if available
    """
    if not references:
        return

    # get all unpaid payment requests for the references
    references_open_payment_requests = get_open_payment_requests_for_references(references)

    if not references_open_payment_requests:
        return

    if not precision:
        precision = references[0].precision("allocated_amount")

    # to manage new rows
    row_number = 1
    MOVE_TO_NEXT_ROW = 1
    TO_SKIP_NEW_ROW = 2

    while row_number <= len(references):
        # Implementation continues...
```

**Why Greedy?**

- Makes locally optimal allocation decisions
- Processes payments in order
- Maximizes allocation efficiency

#### 2. Route Optimization

**Location**: `erpnext/erpnext/stock/doctype/delivery_trip/delivery_trip.py`

```python
@frappe.whitelist()
def process_route(self, optimize):
    """
    Estimate the arrival times for each stop in the Delivery Trip.
    If `optimize` is True, the stops will be re-arranged, based
    on the optimized order, before estimating the arrival times.
    """
    departure_datetime = get_datetime(self.departure_time)
    route_list = self.form_route_list(optimize)

    # For locks, maintain idx count while looping through route list
    idx = 0
    for route in route_list:
        directions = self.get_directions(route, optimize)

        if directions:
            if optimize and len(directions.get("waypoint_order")) > 1:
                self.rearrange_stops(directions.get("waypoint_order"), start=idx)

            # Google Maps returns the legs in the optimized order
            for leg in directions.get("legs"):
                delivery_stop = self.delivery_stops[idx]

                delivery_stop.lat, delivery_stop.lng = leg.get("end_location", {}).values()
                delivery_stop.uom = self.default_distance_uom
                distance = leg.get("distance", {}).get("value", 0.0)  # in meters
                delivery_stop.distance = distance * self.uom_conversion_factor

                duration = leg.get("duration", {}).get("value", 0)
                estimated_arrival = departure_datetime + datetime.timedelta(seconds=duration)
                delivery_stop.estimated_arrival = estimated_arrival

                stop_delay = frappe.db.get_single_value("Delivery Settings", "stop_delay")
                departure_datetime = estimated_arrival + datetime.timedelta(minutes=cint(stop_delay))
                idx += 1
```

**Why Greedy?**

- Optimizes route based on current best option
- Minimizes total travel time
- Uses Google Maps optimization

### 🔧 Common Greedy Patterns

```python
# Activity Selection Problem
def activity_selection(activities):
    # Sort by finish time
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_finish = activities[0][1]

    for start, finish in activities[1:]:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected

# Fractional Knapsack
def fractional_knapsack(items, capacity):
    # Sort by value/weight ratio
    items.sort(key=lambda x: x[1]/x[2], reverse=True)

    total_value = 0
    knapsack = []

    for item, value, weight in items:
        if capacity >= weight:
            knapsack.append((item, weight))
            total_value += value
            capacity -= weight
        else:
            fraction = capacity / weight
            knapsack.append((item, fraction * weight))
            total_value += fraction * value
            break

    return knapsack, total_value
```

---

## Graph Algorithms

### 🎯 What are Graph Algorithms?

Algorithms designed to work with graphs (nodes connected by edges) to solve problems like pathfinding, cycle detection, and connectivity.

### 💼 Real Examples from Frappe

#### 1. Circular Dependency Detection

**Location**: `erpnext/erpnext/projects/doctype/task/task.py`

```python
def check_recursion(self):
    if self.flags.ignore_recursion_check:
        return
    check_list = [["task", "parent"], ["parent", "task"]]
    for d in check_list:
        task_list, count = [self.name], 0
        while len(task_list) > count:
            tasks = frappe.db.sql(
                " select {} from `tabTask Depends On` where {} = {} ".format(d[0], d[1], "%s"),
                cstr(task_list[count]),
            )
            count = count + 1
            for b in tasks:
                if b[0] == self.name:
                    frappe.throw(_("Circular Reference Error"), CircularReferenceError)
                if b[0]:
                    task_list.append(b[0])

            if count == 15:
                break
```

**Algorithm Used**: Depth-First Search for cycle detection
**Why This Approach?**

- Detects circular dependencies in task graphs
- Prevents infinite loops
- Ensures workflow integrity

#### 2. BOM Dependency Resolution

**Location**: `erpnext/erpnext/manufacturing/doctype/bom_update_log/bom_updation_utils.py`

```python
def get_next_higher_level_boms(child_boms: list[str], processed_boms: dict[str, bool]) -> list[str]:
    """Generate immediate higher level dependants with no unresolved dependencies."""

    def _all_children_are_processed(parent_bom):
        child_boms = dependency_map.get(parent_bom)
        return all(processed_boms.get(bom) for bom in child_boms)

    dependants_map, dependency_map = _generate_dependence_map()

    dependants = []
    for bom in child_boms:
        # generate list of immediate dependants
        parents = dependants_map.get(bom) or []
        dependants.extend(parents)

    dependants = set(dependants)  # remove duplicates
    resolved_dependants = set()

    # consider only if children are all resolved
    for parent_bom in dependants:
        if _all_children_are_processed(parent_bom):
            resolved_dependants.add(parent_bom)

    return list(resolved_dependants)
```

**Algorithm Used**: Topological sorting
**Why This Approach?**

- Resolves BOM update dependencies
- Ensures proper update order
- Handles complex dependency graphs

#### 3. Hierarchy Chart Processing

**Location**: `hrms/hrms/utils/hierarchy_chart.py`

```python
@frappe.whitelist()
def get_all_nodes(method, company):
    """Recursively gets all data from nodes"""
    method = frappe.get_attr(method)

    root_nodes = method(company=company)
    result = []
    nodes_to_expand = []

    for root in root_nodes:
        data = method(root.id, company)
        result.append(dict(parent=root.id, parent_name=root.name, data=data))
        nodes_to_expand.extend(
            [{"id": d.get("id"), "name": d.get("name")} for d in data if d.get("expandable")]
        )

    while nodes_to_expand:
        parent = nodes_to_expand.pop(0)
        data = method(parent.get("id"), company)
        result.append(dict(parent=parent.get("id"), parent_name=parent.get("name"), data=data))
        for d in data:
            if d.get("expandable"):
                nodes_to_expand.append({"id": d.get("id"), "name": d.get("name")})

    return result
```

**Algorithm Used**: Breadth-First Search
**Why This Approach?**

- Processes hierarchy level by level
- Handles large organizational structures
- Efficient memory usage

### 🔧 Common Graph Algorithms

```python
# Depth-First Search
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Breadth-First Search
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Detect Cycle in Directed Graph
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        if color[node] == GRAY:
            return True  # Cycle detected
        if color[node] == BLACK:
            return False

        color[node] = GRAY
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False

# Topological Sort
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}

    # Calculate in-degrees
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Find nodes with no incoming edges
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else []  # Empty if cycle exists
```

---

## 🎯 Interview Questions & Answers

### Q1: "How does Frappe implement efficient searching?"

**Answer**: "Frappe uses multiple searching strategies depending on the use case:

1. **Database-level search** with relevance scoring using `LOCATE()` function
2. **Full-text search** using MySQL's `MATCH() AGAINST()` for global search
3. **Multi-field search** across different columns with fallback mechanisms

For example, in employee search:

```python
order by
    (case when locate(%(_txt)s, name) > 0 then locate(%(_txt)s, name) else 99999 end),
    (case when locate(%(_txt)s, employee_name) > 0 then locate(%(_txt)s, employee_name) else 99999 end),
    idx desc, name, employee_name
```

This prioritizes exact matches and provides fallback sorting."

### Q2: "Explain how Frappe handles recursive data structures."

**Answer**: "Frappe uses recursion extensively for hierarchical data:

1. **BOM Trees**: Recursive BOM explosion for material requirements
2. **Organizational Charts**: Building employee hierarchies
3. **Account Trees**: Chart of accounts structure

The key is to have proper base cases and avoid infinite recursion. For example, in BOM trees:

```python
def __create_tree(self):
    bom = frappe.get_cached_doc("BOM", self.name)
    for item in bom.get("items", []):
        if item.bom_no:
            # Recursive call for sub-BOMs
            child = BOMTree(item.bom_no, exploded_qty=exploded_qty, qty=qty)
            self.child_items.append(child)
```

This naturally handles nested structures while maintaining performance."

### Q3: "How does Frappe optimize payment allocation?"

**Answer**: "Frappe uses a greedy algorithm for payment allocation:

1. **Local optimization**: Allocates payments to references based on outstanding amounts
2. **Priority handling**: Processes payment requests in order
3. **Efficient matching**: Matches payments to references optimally

The algorithm makes locally optimal choices at each step:

- If allocated amount equals payment request amount → allocate completely
- If allocated amount is less → partial allocation, continue with remaining
- If allocated amount is more → allocate fully, create new row for remaining

This ensures maximum allocation efficiency while maintaining business logic."

---

## 🚀 Next Steps

Ready to understand performance implications? Move on to [Time & Space Complexity](./03_Complexity_Analysis.md) to learn how to analyze algorithm efficiency!

**Key Takeaways:**

- Choose algorithms based on problem requirements
- Understand trade-offs between different approaches
- Use real examples to explain algorithm choices in interviews
- Practice implementing these patterns in your own projects
