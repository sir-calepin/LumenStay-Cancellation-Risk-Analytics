# Raw Data

The original LumenStay dataset is not included in this repository unless explicit
permission to share it has been obtained.

To reproduce the analysis, place an authorized source file in this folder. The
notebooks expect a CSV file with a binary target column named:

```text
cancelled_flag
```

The target definition is:

- `1`: reservation cancelled before arrival
- `0`: reservation remained active through arrival

Do not commit confidential guest records, booking identifiers, payment information,
or restricted course/project data to a public repository.
