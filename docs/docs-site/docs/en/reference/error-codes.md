# Error codes

UCM uses the following error codes in its underlying C++ stores. These codes appear in log messages and exception outputs.

| Error Code | Name | Meaning | Common Cause |
|------------|------|---------|--------------|
| **0** | `OK` | Success | Operation completed normally |
| **-1** | `Error` | General error | Unclassified error, check the attached message |
| **-50000** | `InvalidParam` | Invalid parameter | Parameter is illegal, empty, or has wrong format |
| **-50001** | `OutOfMemory` | Out of memory | Memory allocation failed |
| **-50002** | `OsApiError` | OS API error | System call failed (e.g. file I/O, thread operation) |
| **-50003** | `DuplicateKey` | Duplicate key | Attempting to insert a key that already exists |
| **-50004** | `Retry` | Retry required | Transient error, retry recommended |
| **-50005** | `NotFound` | Not found | Requested object or resource does not exist |
| **-50006** | `SerializeFailed` | Serialization failed | Error during data serialization |
| **-50007** | `DeserializeFailed` | Deserialization failed | Error during data deserialization |
| **-50008** | `Unsupported` | Unsupported operation | Feature or operation not supported in current context |
| **-50009** | `NoSpace` | No space | Storage space (disk/cache) is full |
| **-50010** | `Timeout` | Timeout | Operation timed out |

For diagnosis by symptom, see [troubleshooting](troubleshooting.md).
