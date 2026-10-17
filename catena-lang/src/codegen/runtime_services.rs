//! Host services shared by every code generator: metered device allocation,
//! exported under the names in [`crate::runtime::module_abi`].

use crate::runtime::{GpuDialect, module_abi};

/// Render the services for `dialect`. They are host-only and must follow the
/// standard headers; `catena_runtime_allocate` is the entry point generated host
/// code calls.
pub(crate) fn render(dialect: GpuDialect) -> String {
    format!(
        r#"#ifndef {guard}
__host__ static inline void catena_runtime_reject(const char *reason) {{
    fprintf(stderr, "catena runtime rejected generated code: %s\n", reason);
    fflush(stderr);
    __builtin_trap();
}}

// Generated allocations are charged cumulatively within one entry-point
// invocation. Freeing a buffer does not refund budget: this deliberately caps
// allocation churn as well as peak live bytes.
static thread_local uint64_t catena_runtime_allocation_budget = UINT64_MAX;
static thread_local uint64_t catena_runtime_allocated_bytes = 0;

extern "C" __host__ void {budget_begin}(uint64_t byte_limit) {{
    catena_runtime_allocation_budget = byte_limit;
    catena_runtime_allocated_bytes = 0;
}}

extern "C" __host__ void {budget_end}() {{
    catena_runtime_allocation_budget = UINT64_MAX;
    catena_runtime_allocated_bytes = 0;
}}

// Allocate `element_count * element_size` device bytes; null when that is zero.
__host__ static inline void *catena_runtime_allocate(uint64_t element_count, uint64_t element_size) {{
    if (element_size != 0 && element_count > UINT64_MAX / element_size) {{
        catena_runtime_reject("allocation byte count overflowed");
    }}
    uint64_t byte_len = element_count * element_size;
    if (byte_len > SIZE_MAX) {{
        catena_runtime_reject("allocation byte count exceeds size_t");
    }}
    if (catena_runtime_allocated_bytes > catena_runtime_allocation_budget
        || byte_len > catena_runtime_allocation_budget - catena_runtime_allocated_bytes) {{
        catena_runtime_reject("allocation budget exceeded");
    }}
    void *data = nullptr;
    if (byte_len == 0) {{
        return data;
    }}
    {error_type} error = {device_alloc}(&data, (size_t)byte_len);
    if (error != {success}) {{
        fprintf(stderr, "catena GPU error: %s\n", {error_string}(error));
        fflush(stderr);
        __builtin_trap();
    }}
    catena_runtime_allocated_bytes += byte_len;
    return data;
}}

#endif
"#,
        guard = dialect.device_compile_guard(),
        budget_begin = module_abi::ALLOCATION_BUDGET_BEGIN,
        budget_end = module_abi::ALLOCATION_BUDGET_END,
        error_type = dialect.error_type(),
        device_alloc = dialect.device_alloc_fn(),
        success = dialect.success_value(),
        error_string = dialect.error_string_fn(),
    )
}
