"""A minimal in-memory stand-in for the real supabase-py client, used so the
test suite never makes a real network call to Supabase. It implements just
enough of the query-builder chain (table/insert/select/update/eq/order/limit/
execute) to support what database.py actually calls — it isn't a general
Supabase mock, just enough of one for these tests.
"""


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, table_name, rows):
        self._table_name = table_name
        self._rows = rows  # reference to the client's list for this table
        self._filters = []
        self._insert_rows = None
        self._update_values = None
        self._order_key = None
        self._order_desc = False
        self._limit = None

    def select(self, columns="*"):
        return self

    def insert(self, row):
        self._insert_rows = row if isinstance(row, list) else [row]
        return self

    def update(self, values):
        self._update_values = values
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self._order_key = key
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _matching(self):
        return [r for r in self._rows if all(r.get(k) == v for k, v in self._filters)]

    def execute(self):
        if self._insert_rows is not None:
            inserted = []
            for row in self._insert_rows:
                new_row = dict(row)
                # Mirrors the real schema's column default, since a fresh insert
                # here doesn't go through Postgres's own DEFAULT FALSE.
                if self._table_name == "feedback" and "used_in_training" not in new_row:
                    new_row["used_in_training"] = False
                new_row["id"] = len(self._rows) + 1
                self._rows.append(new_row)
                inserted.append(new_row)
            return _FakeResult(inserted)

        if self._update_values is not None:
            matched = self._matching()
            for row in matched:
                row.update(self._update_values)
            return _FakeResult(matched)

        matched = self._matching()
        if self._order_key:
            matched = sorted(matched, key=lambda r: r.get(self._order_key), reverse=self._order_desc)
        if self._limit is not None:
            matched = matched[: self._limit]
        return _FakeResult(matched)


class FakeSupabaseClient:
    def __init__(self):
        self.tables = {}

    def table(self, name):
        self.tables.setdefault(name, [])
        return _FakeQuery(name, self.tables[name])
