// Probe: is opencog::combo::TimedValue::operator< (table.h:482) a valid
// strict weak ordering? It is the comparator for
// Counter<TimedValue, count_t> = std::map<TimedValue, ...>, which every
// CTable cell is when a timestamp column is present (table.cc:548).
#include <moses/comboreduct/table/table.h>
#include <cstdio>
using namespace opencog;
using namespace opencog::combo;

int main() {
  // Two values differing in BOTH fields, in opposite directions.
  TimedValue a(contin_t(1.0), TTable::value_type(boost::gregorian::date(2020, 1, 5)));
  TimedValue b(contin_t(2.0), TTable::value_type(boost::gregorian::date(2020, 1, 3)));
  printf("a<b = %d\n", (int)(a < b));
  printf("b<a = %d\n", (int)(b < a));
  printf("asymmetry violated (both true) = %d\n", (int)((a < b) && (b < a)));
  printf("a==b = %d\n", (int)(a == b));
  return 0;
}
